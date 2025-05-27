#!/usr/bin/env python
# coding: utf-8

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps,
visualizes how data transforms during forward and reverse diffusion,
and evaluates the quality of generated samples.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DiffusionModel
from src.utils import set_seed


def visualize_noise_prediction(
    model, batch, timesteps=[100, 500, 900], output_dir=None
):
    """Visualize true vs predicted noise at different timesteps."""
    model.eval()
    device = next(model.parameters()).device
    batch = batch.to(device)

    fig, axes = plt.subplots(len(timesteps), 2, figsize=(12, 4 * len(timesteps)))

    for i, t in enumerate(timesteps):
        # Create timestep tensor
        t_tensor = torch.full((batch.shape[0],), t, device=device, dtype=torch.long)

        # Generate true noise
        true_noise = torch.randn_like(batch)

        # Add noise to data at timestep t
        noisy_batch = model.forward_diffusion.sample(batch, t_tensor, true_noise)

        # Predict noise
        with torch.no_grad():
            pred_noise = model.predict_added_noise(noisy_batch, t_tensor)

        # Plot true noise
        axes[i, 0].hist(true_noise.cpu().flatten().numpy(), bins=50, alpha=0.7)
        axes[i, 0].set_title(f"True Noise (t={t})")

        # Plot predicted noise
        axes[i, 1].hist(pred_noise.cpu().flatten().numpy(), bins=50, alpha=0.7)
        axes[i, 1].set_title(f"Predicted Noise (t={t})")

        # Add statistics
        axes[i, 0].text(
            0.05,
            0.95,
            f"Mean: {true_noise.mean():.4f}\nStd: {true_noise.std():.4f}",
            transform=axes[i, 0].transAxes,
            verticalalignment="top",
        )
        axes[i, 1].text(
            0.05,
            0.95,
            f"Mean: {pred_noise.mean():.4f}\nStd: {pred_noise.std():.4f}",
            transform=axes[i, 1].transAxes,
            verticalalignment="top",
        )

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "noise_prediction.png", dpi=300, bbox_inches="tight")
    return fig


def generate_and_evaluate_samples(
    model, real_data, num_samples=100, denoise_step=10, output_dir=None
):
    """Generate samples from the model and evaluate their quality compared to real data.

    Args:
        model: The trained diffusion model
        real_data: Real data samples for comparison
        num_samples: Number of samples to generate
        denoise_step: Step size for the reverse diffusion process
        output_dir: Directory to save visualizations

    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\nGenerating and evaluating samples...")
    model.eval()
    device = next(model.parameters()).device

    # Ensure real_data has the right shape and is on the correct device
    real_data = real_data.to(device)
    if real_data.dim() == 2:
        real_data = real_data.unsqueeze(1)  # Add channel dimension if needed

    # Generate samples
    print(f"Generating {num_samples} samples with denoise_step={denoise_step}...")
    with torch.no_grad():
        generated_samples = model.generate_samples(
            num_samples=num_samples,
            denoise_step=denoise_step,
            discretize=True,
            seed=42,
            device=device,
        )

    print(f"Generated samples shape: {generated_samples.shape}")

    # Convert to numpy for analysis
    gen_np = generated_samples.cpu().numpy()
    real_np = real_data.cpu().numpy()

    # Calculate metrics
    metrics = {}

    # 1. Calculate MAF (Minor Allele Frequency) correlation
    metrics.update(calculate_maf_correlation(gen_np, real_np, output_dir))

    # 2. Compare value distributions
    metrics.update(compare_value_distributions(gen_np, real_np, output_dir))

    # 3. Visualize samples
    visualize_samples(gen_np, real_np, output_dir)

    return metrics


def calculate_maf_correlation(generated_samples, real_samples, output_dir=None):
    """Calculate Minor Allele Frequency correlation between generated and real samples.

    Args:
        generated_samples: Generated samples array of shape [num_samples, channels, seq_len]
        real_samples: Real samples array of shape [num_real, channels, seq_len]
        output_dir: Directory to save visualizations

    Returns:
        dict: Dictionary with MAF correlation metrics
    """
    print("\nCalculating MAF correlation...")

    # Reshape if needed (assuming channel dim is 1 for SNP data)
    if generated_samples.ndim == 3:
        generated_samples = generated_samples.squeeze(1)  # [num_samples, seq_len]
    if real_samples.ndim == 3:
        real_samples = real_samples.squeeze(1)  # [num_real, seq_len]

    # Calculate MAF for each marker position
    # For SNP data scaled to [0, 0.5], we multiply by 2 to get [0, 1]
    # Then MAF is the mean across samples
    gen_maf = np.mean(generated_samples * 2, axis=0)  # [seq_len]
    real_maf = np.mean(real_samples * 2, axis=0)  # [seq_len]

    # Calculate correlation
    maf_corr, p_value = stats.pearsonr(gen_maf, real_maf)
    print(f"MAF Correlation: {maf_corr:.4f} (p-value: {p_value:.4e})")

    # Visualize MAF correlation
    if output_dir:
        plt.figure(figsize=(8, 8))
        plt.scatter(real_maf, gen_maf, alpha=0.5)
        plt.plot([0, 0.5], [0, 0.5], "r--")  # Identity line
        plt.xlabel("Real Data MAF")
        plt.ylabel("Generated Data MAF")
        plt.title(f"Minor Allele Frequency Correlation: {maf_corr:.4f}")
        plt.savefig(output_dir / "maf_correlation.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {"maf_correlation": maf_corr, "maf_p_value": p_value}


def compare_value_distributions(generated_samples, real_samples, output_dir=None):
    """Compare the distribution of values between generated and real samples.

    Args:
        generated_samples: Generated samples array
        real_samples: Real samples array
        output_dir: Directory to save visualizations

    Returns:
        dict: Dictionary with distribution comparison metrics
    """
    print("\nComparing value distributions...")

    # Flatten arrays for distribution analysis
    gen_flat = generated_samples.flatten()
    real_flat = real_samples.flatten()

    # Calculate KL divergence between distributions
    # First, create histograms
    bins = np.linspace(0, 0.5, 101)  # 100 bins from 0 to 0.5
    gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)
    real_hist, _ = np.histogram(real_flat, bins=bins, density=True)

    # Add small constant to avoid division by zero
    gen_hist = gen_hist + 1e-10
    real_hist = real_hist + 1e-10

    # Normalize
    gen_hist = gen_hist / gen_hist.sum()
    real_hist = real_hist / real_hist.sum()

    # Calculate KL divergence: KL(real || gen)
    kl_div = np.sum(real_hist * np.log(real_hist / gen_hist))
    print(f"KL Divergence (real || gen): {kl_div:.4f}")

    # Visualize distributions
    if output_dir:
        plt.figure(figsize=(10, 6))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, real_hist, width=0.005, alpha=0.6, label="Real Data")
        plt.bar(bin_centers, gen_hist, width=0.005, alpha=0.6, label="Generated Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Value Distribution Comparison")
        plt.legend()
        plt.savefig(output_dir / "value_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Also plot as a histogram for easier comparison
        plt.figure(figsize=(10, 6))
        plt.hist(real_flat, bins=bins, alpha=0.6, density=True, label="Real Data")
        plt.hist(gen_flat, bins=bins, alpha=0.6, density=True, label="Generated Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Value Distribution Comparison (Histogram)")
        plt.legend()
        plt.savefig(output_dir / "value_histogram.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {"kl_divergence": kl_div}


def visualize_samples(generated_samples, real_samples, output_dir=None, num_to_show=5):
    """Visualize a few generated samples compared to real samples.

    Args:
        generated_samples: Generated samples array
        real_samples: Real samples array
        output_dir: Directory to save visualizations
        num_to_show: Number of samples to visualize
    """
    print("\nVisualizing samples...")

    # Ensure we don't try to show more samples than we have
    num_to_show = min(num_to_show, generated_samples.shape[0], real_samples.shape[0])

    # Reshape if needed
    if generated_samples.ndim == 3:
        generated_samples = generated_samples.squeeze(1)  # [num_samples, seq_len]
    if real_samples.ndim == 3:
        real_samples = real_samples.squeeze(1)  # [num_real, seq_len]

    # Create figure
    fig, axes = plt.subplots(num_to_show, 2, figsize=(12, 3 * num_to_show))

    # Plot samples
    for i in range(num_to_show):
        # Plot real sample
        im_real = axes[i, 0].imshow(
            real_samples[i : i + 1], aspect="auto", cmap="viridis"
        )
        axes[i, 0].set_title(f"Real Sample {i+1}")
        axes[i, 0].set_yticks([])

        # Plot generated sample
        im_gen = axes[i, 1].imshow(
            generated_samples[i : i + 1], aspect="auto", cmap="viridis"
        )
        axes[i, 1].set_title(f"Generated Sample {i+1}")
        axes[i, 1].set_yticks([])

    # Add colorbar
    fig.colorbar(im_real, ax=axes[:, 0].ravel().tolist(), shrink=0.8, label="Value")
    fig.colorbar(im_gen, ax=axes[:, 1].ravel().tolist(), shrink=0.8, label="Value")

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "sample_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Also visualize a larger batch of samples as a heatmap
    num_for_heatmap = min(50, generated_samples.shape[0])
    plt.figure(figsize=(12, 8))
    plt.imshow(generated_samples[:num_for_heatmap], aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    plt.title(f"Generated Samples Heatmap (n={num_for_heatmap})")
    plt.xlabel("Marker Position")
    plt.ylabel("Sample")

    if output_dir:
        plt.savefig(output_dir / "generated_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()


def visualize_diffusion_process(
    model, batch, timesteps=[100, 500, 900], output_dir=None
):
    """Visualize the forward and reverse diffusion process at different timesteps.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
    """
    print("\nVisualizing diffusion process at different timesteps...")
    model.eval()
    device = next(model.parameters()).device

    # Ensure batch has the right shape and is on the correct device
    batch = batch.to(device)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)  # Add channel dimension if needed

    # Use only the first sample for visualization
    x0 = batch[0:1]  # Shape: [1, C, seq_len]

    # Get the sequence length for proper reshaping
    seq_len = x0.shape[-1]

    # Create a figure with 5 columns (original, noise, noisy, pred_noise, denoised)
    # and one row per timestep
    fig, axes = plt.subplots(len(timesteps), 5, figsize=(20, 4 * len(timesteps)))

    # Handle the case of a single timestep
    if len(timesteps) == 1:
        axes = axes.reshape(1, -1)  # Reshape to 2D array with 1 row

    # Set column titles
    axes[0, 0].set_title("Original Sample")
    axes[0, 1].set_title("Added Noise")
    axes[0, 2].set_title("Noisy Sample")
    axes[0, 3].set_title("Predicted Noise")
    axes[0, 4].set_title("Denoised Sample")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # 1. Original sample (x0)
            # Already have x0

            # 2. Generate noise
            eps = torch.randn_like(x0)

            # 3. Forward diffusion: Add noise to get x_t
            x_t = model.forward_diffusion.sample(x0, t_tensor, eps)

            # 4. Predict noise
            pred_eps = model.predict_added_noise(x_t, t_tensor)

            # 5. Reverse diffusion: Denoise x_t to get x_0_pred
            # For a single step denoising, we'll use the reverse_diffusion_step method
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)

            # Prepare data for visualization - reshape to 2D
            # For 1D data, we'll reshape to [1, seq_len] for imshow
            x0_vis = x0.cpu().squeeze().numpy().reshape(1, -1)
            eps_vis = eps.cpu().squeeze().numpy().reshape(1, -1)
            x_t_vis = x_t.cpu().squeeze().numpy().reshape(1, -1)
            pred_eps_vis = pred_eps.cpu().squeeze().numpy().reshape(1, -1)
            x_0_pred_vis = x_0_pred.cpu().squeeze().numpy().reshape(1, -1)

            # Plot results
            # 1. Original
            im0 = axes[i, 0].imshow(x0_vis, aspect="auto", cmap="viridis")
            axes[i, 0].set_ylabel(f"t={t}")
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            # 2. Noise
            im1 = axes[i, 1].imshow(eps_vis, aspect="auto", cmap="viridis")
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

            # 3. Noisy
            im2 = axes[i, 2].imshow(x_t_vis, aspect="auto", cmap="viridis")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])

            # 4. Predicted Noise
            im3 = axes[i, 3].imshow(pred_eps_vis, aspect="auto", cmap="viridis")
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])

            # 5. Denoised
            im4 = axes[i, 4].imshow(x_0_pred_vis, aspect="auto", cmap="viridis")
            axes[i, 4].set_xticks([])
            axes[i, 4].set_yticks([])

            # Add text with stats
            axes[i, 0].text(
                0.5,
                -0.15,
                f"Mean: {x0.mean():.3f}\nStd: {x0.std():.3f}",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 1].text(
                0.5,
                -0.15,
                f"Mean: {eps.mean():.3f}\nStd: {eps.std():.3f}",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 2].text(
                0.5,
                -0.15,
                f"Mean: {x_t.mean():.3f}\nStd: {x_t.std():.3f}",
                transform=axes[i, 2].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 3].text(
                0.5,
                -0.15,
                f"Mean: {pred_eps.mean():.3f}\nStd: {pred_eps.std():.3f}",
                transform=axes[i, 3].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 4].text(
                0.5,
                -0.15,
                f"Mean: {x_0_pred.mean():.3f}\nStd: {x_0_pred.std():.3f}",
                transform=axes[i, 4].transAxes,
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    if output_dir:
        plt.savefig(
            output_dir / "diffusion_process_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    return fig


def visualize_diffusion_process_lineplot(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the forward and reverse diffusion process at different timesteps using line plots.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization (to avoid overcrowded plots)
    """
    print("\nVisualizing diffusion process using line plots...")
    model.eval()
    device = next(model.parameters()).device

    # Ensure batch has the right shape and is on the correct device
    batch = batch.to(device)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)  # Add channel dimension if needed

    # Use only the first sample for visualization
    x0 = batch[0:1]  # Shape: [1, C, seq_len]

    # Get the sequence length
    seq_len = x0.shape[-1]

    # If sequence is too long, sample points for visualization
    if seq_len > sample_points:
        indices = np.linspace(0, seq_len - 1, sample_points, dtype=int)
    else:
        indices = np.arange(seq_len)

    # Create a figure with 5 columns (original, noise, noisy, pred_noise, denoised)
    # and one row per timestep
    fig, axes = plt.subplots(len(timesteps), 5, figsize=(20, 4 * len(timesteps)))

    # Handle the case of a single timestep
    if len(timesteps) == 1:
        axes = axes.reshape(1, -1)  # Reshape to 2D array with 1 row

    # Set column titles
    axes[0, 0].set_title("Original Sample")
    axes[0, 1].set_title("Added Noise")
    axes[0, 2].set_title("Noisy Sample")
    axes[0, 3].set_title("Predicted Noise")
    axes[0, 4].set_title("Denoised Sample")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # 1. Original sample (x0)
            # Already have x0

            # 2. Generate noise
            eps = torch.randn_like(x0)

            # 3. Forward diffusion: Add noise to get x_t
            x_t = model.forward_diffusion.sample(x0, t_tensor, eps)

            # 4. Predict noise
            pred_eps = model.predict_added_noise(x_t, t_tensor)

            # 5. Reverse diffusion: Denoise x_t to get x_0_pred
            # For a single step denoising, we'll use the reverse_diffusion_step method
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)

            # Prepare data for visualization - convert to numpy and sample points
            x0_vis = x0.cpu().squeeze().numpy()[indices]
            eps_vis = eps.cpu().squeeze().numpy()[indices]
            x_t_vis = x_t.cpu().squeeze().numpy()[indices]
            pred_eps_vis = pred_eps.cpu().squeeze().numpy()[indices]
            x_0_pred_vis = x_0_pred.cpu().squeeze().numpy()[indices]

            # Create x-axis for plotting
            x_axis = np.arange(len(indices))

            # Plot results
            # 1. Original
            axes[i, 0].plot(x_axis, x0_vis, "b-", linewidth=1)
            axes[i, 0].set_ylabel(f"t={t}")
            axes[i, 0].set_ylim(-0.1, 0.6)  # Set consistent y-axis limits for SNP data

            # 2. Noise
            axes[i, 1].plot(x_axis, eps_vis, "r-", linewidth=1)
            axes[i, 1].set_ylim(-3, 3)  # Typical range for noise

            # 3. Noisy
            axes[i, 2].plot(x_axis, x_t_vis, "g-", linewidth=1)
            axes[i, 2].set_ylim(-3, 3)  # Allow wider range for noisy data

            # 4. Predicted Noise
            axes[i, 3].plot(x_axis, pred_eps_vis, "m-", linewidth=1)
            axes[i, 3].set_ylim(-3, 3)  # Typical range for noise

            # 5. Denoised
            axes[i, 4].plot(x_axis, x_0_pred_vis, "c-", linewidth=1)
            axes[i, 4].set_ylim(-0.1, 0.6)  # Set consistent y-axis limits for SNP data

            # Add text with stats
            axes[i, 0].text(
                0.5,
                -0.15,
                f"Mean: {x0.mean():.3f}\nStd: {x0.std():.3f}",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 1].text(
                0.5,
                -0.15,
                f"Mean: {eps.mean():.3f}\nStd: {eps.std():.3f}",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 2].text(
                0.5,
                -0.15,
                f"Mean: {x_t.mean():.3f}\nStd: {x_t.std():.3f}",
                transform=axes[i, 2].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 3].text(
                0.5,
                -0.15,
                f"Mean: {pred_eps.mean():.3f}\nStd: {pred_eps.std():.3f}",
                transform=axes[i, 3].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 4].text(
                0.5,
                -0.15,
                f"Mean: {x_0_pred.mean():.3f}\nStd: {x_0_pred.std():.3f}",
                transform=axes[i, 4].transAxes,
                ha="center",
                fontsize=8,
            )

            # Hide x-axis ticks except for the last row
            if i < len(timesteps) - 1:
                for j in range(5):
                    axes[i, j].set_xticks([])

    plt.tight_layout()
    if output_dir:
        plt.savefig(
            output_dir / "diffusion_process_lineplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    return fig


# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test diffusion model and evaluate generated samples"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--snp_index",
        type=int,
        default=50,
        help="Index of the SNP to monitor (default: 50)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--denoise_step",
        type=int,
        default=10,
        help="Step size for reverse diffusion (default: 10)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs for variance estimation (default: 3)",
    )
    return parser.parse_args()


def main():
    # Set global seed for reproducibility
    set_seed(42)

    # Parse Arguments
    args = parse_args()

    try:
        # Load the model from checkpoint
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        # Get model config and move to device
        config = model.hparams
        model = model.to(device)
        model.eval()

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)

    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Setup output directory structure
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    results_dir = base_dir / "evaluation_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load test dataset
    print("\nLoading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Get a batch of test data
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    print(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # Run analyses
    # 1. Visualize noise prediction
    print("\n1. Visualizing noise prediction...")
    visualize_noise_prediction(model, x0, output_dir=results_dir)

    # 2. Visualize the diffusion process (heatmap)
    print("\n2. Visualizing diffusion process (heatmap)...")
    visualize_diffusion_process(
        model=model,
        batch=x0,
        timesteps=[100, 500, 900],  # Low, medium, and high noise levels
        output_dir=results_dir,
    )

    # 3. Visualize the diffusion process (line plot)
    print("\n3. Visualizing diffusion process (line plot)...")
    visualize_diffusion_process_lineplot(
        model=model,
        batch=x0,
        timesteps=[100, 500, 900],  # Low, medium, and high noise levels
        output_dir=results_dir,
        sample_points=200,  # Sample 200 points for clearer visualization
    )

    # 3. Generate and evaluate samples
    print("\n3. Generating and evaluating samples...")
    metrics = generate_and_evaluate_samples(
        model=model,
        real_data=x0,
        num_samples=100,  # Generate 100 samples for evaluation
        denoise_step=args.denoise_step,
        output_dir=results_dir,
    )

    # Print summary of metrics
    print("\nEvaluation Metrics Summary:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

    print(f"\nAll evaluation results saved to: {results_dir}")


if __name__ == "__main__":
    main()
