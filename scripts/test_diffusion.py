#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

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
from scipy import stats

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DiffusionModel
from src.utils import set_seed


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


def visualize_marker_trajectory(
    model, sample, marker_idx=50, timesteps=None, output_dir=None
):
    """
    Visualize the value of a single SNP marker as noise is added and removed at different timesteps.
    Args:
        model: The diffusion model
        sample: Tensor of shape (channels, seq_len) or (1, channels, seq_len)
        marker_idx: Index of the marker/SNP to track
        timesteps: List of timesteps to plot (default: [1, 100, 200, 500, 900])
        output_dir: Optional path to save the figure
    """
    if timesteps is None:
        timesteps = [1, 100, 200, 500, 900]
    model.eval()
    device = next(model.parameters()).device
    if sample.dim() == 2:
        sample = sample.unsqueeze(0)  # Add batch dimension if needed
    sample = sample.to(device)

    original_value = sample[0, ..., marker_idx].item()
    noisy_values = []
    denoised_values = []

    for t in timesteps:
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        true_noise = torch.randn_like(sample)
        noisy_sample = model.forward_diffusion.sample(sample, t_tensor, true_noise)
        with torch.no_grad():
            pred_noise = model.predict_added_noise(noisy_sample, t_tensor)

        # Denoise (reverse process, simplified for DDPM-like models)
        alpha_bar = model.forward_diffusion.alphas_bar(t_tensor)
        denoised = (
            noisy_sample - (1 - alpha_bar).sqrt() * pred_noise
        ) / alpha_bar.sqrt()
        noisy_values.append(noisy_sample[0, ..., marker_idx].item())
        denoised_values.append(denoised[0, ..., marker_idx].item())

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, [original_value] * len(timesteps), "k--", label="Original")
    plt.plot(timesteps, noisy_values, "ro-", label="Noisy")
    plt.plot(timesteps, denoised_values, "bo-", label="Denoised")
    plt.xlabel("Timestep")
    plt.ylabel(f"Marker {marker_idx} Value")
    plt.title(f"Diffusion Process for Marker {marker_idx}")
    plt.legend()
    plt.grid(True)
    if output_dir is not None:
        plt.savefig(
            str(output_dir / f"marker_{marker_idx}_trajectory.png"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.show()


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
    output_dir = base_dir / "test_diffusion"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load test dataset
    print("\nLoading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Get a batch of test data
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    print(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # Shared timesteps for all visualizations
    timesteps = [1, 2, 3, 4, 5]

    # 3. Visualize marker trajectory for a single sample and marker
    print("\n3. Visualizing marker trajectory...")
    sample = x0[0]  # Take the first sample in the batch
    visualize_marker_trajectory(
        model,
        sample,
        marker_idx=args.snp_index,
        timesteps=timesteps,
        output_dir=output_dir,
    )

    # 4. Generate and evaluate samples
    print("\n4. Generating and evaluating samples...")
    metrics = generate_and_evaluate_samples(
        model=model,
        real_data=x0,
        num_samples=args.num_samples,
        denoise_step=args.denoise_step,
        output_dir=output_dir,
    )

    # Print summary of metrics
    print("\nEvaluation Metrics Summary:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

    print(f"\nAll evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
