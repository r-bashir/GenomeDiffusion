#!/usr/bin/env python
# coding: utf-8

"""Script for generating and analyzing samples from trained SNP diffusion models.

Examples:
    # Generate default number of samples (from config)
    python inference.py --config config.yaml --checkpoint path/to/checkpoint.ckpt

    # Generate specific number of samples
    python inference.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --num_samples 100

Generated outputs are saved in the 'inference' directory, including:
- Generated samples (.pt file)
- Sample comparisons with real data
- Statistical analysis
- Visualization plots
"""

import argparse
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA
from diffusion.diffusion_model import DiffusionModel


def plot_sample_grid(samples, save_path, title, timesteps=None):
    """Plot a grid of samples showing the diffusion process.

    Args:
        samples: Tensor of samples to plot
        save_path: Path to save the plot
        title: Title for the plot
        timesteps: List of timesteps corresponding to each sample
    """
    n_samples = min(samples.shape[0], 500)  # Show at most 500 timesteps
    seq_length = samples.shape[-1]

    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 2 * n_samples))
    fig.suptitle(title)

    # If only one sample, axes won't be an array
    if n_samples == 1:
        axes = [axes]

    # Plot each sample
    for i in range(n_samples):
        axes[i].imshow(samples[i, 0, :].reshape(1, -1), aspect="auto", cmap="viridis")
        if timesteps is not None:
            axes[i].set_title(f"t = {timesteps[i]}")
        else:
            axes[i].set_title(f"Step {i}")
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(real_samples, generated_samples, save_path):
    """Plot comparison between real and generated samples."""
    # Convert to numpy for plotting
    real = real_samples.cpu().numpy()
    gen = generated_samples.cpu().numpy()

    # Print statistics for debugging
    # print("\nData Statistics:")
    # print(
    #    f"Real - shape: {real.shape}, range: [{real.min():.3f}, {real.max():.3f}], mean: {real.mean():.3f}, std: {real.std():.3f}"
    # )
    # print(
    #    f"Generated - shape: {gen.shape}, range: [{gen.min():.3f}, {gen.max():.3f}], mean: {gen.mean():.3f}, std: {gen.std():.3f}"
    # )

    # Check for NaN values
    if np.isnan(gen).any():
        print("Warning: Generated samples contain NaN values!")
        gen = np.nan_to_num(gen, nan=0.0)  # Replace NaN with 0

    # Add channel dimension if missing
    if len(real.shape) == 2:
        real = real.reshape(real.shape[0], 1, -1)
    if len(gen.shape) == 2:
        gen = gen.reshape(gen.shape[0], 1, -1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Real vs Generated Samples Comparison")

    try:
        # Plot sample sequences
        axes[0, 0].plot(real[0].flatten(), label="Real")
        axes[0, 0].plot(gen[0].flatten(), label="Generated")
        axes[0, 0].set_title("Sample Sequence Comparison")
        axes[0, 0].set_xlabel("Position")
        axes[0, 0].set_ylabel("Value")
        axes[0, 0].legend()

        # Plot heatmaps
        first_100 = min(100, real.shape[-1])

        # Custom colormap: 0 → blue, 0.5 → green, 1 → red
        cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])

        # Plot real data
        axes[0, 1].imshow(
            real[0].reshape(1, -1)[:, :first_100],
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, 1].set_title("Real Data Pattern (first 100 positions)")
        axes[0, 1].set_yticks([])

        # Plot generated data
        axes[1, 0].imshow(
            gen[0].reshape(1, -1)[:, :first_100],
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, 0].set_title("Generated Data Pattern (first 100 positions)")
        axes[1, 0].set_yticks([])

        # Plot value distributions
        axes[1, 1].hist(real.flatten(), bins=50, alpha=0.5, label="Real", density=True)
        axes[1, 1].hist(
            gen.flatten(), bins=50, alpha=0.5, label="Generated", density=True
        )
        axes[1, 1].set_title("Value Distribution")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Error during plotting: {e}")
        plt.close()


def compute_genomic_metrics(real_samples, generated_samples, output_dir):
    """Compute genomic-specific metrics for generated samples.

    Args:
        real_samples: Tensor of real SNP data [batch_size, channels, seq_len]
        generated_samples: Tensor of generated SNP data [batch_size, channels, seq_len]
        output_dir: Directory to save plots

    Returns:
        dict: Dictionary of genomic metrics
    """
    metrics = {}

    # Ensure tensors are on CPU
    real = real_samples.cpu()
    gen = generated_samples.cpu()

    # Flatten channel dimension if present
    if len(real.shape) > 2 and real.shape[1] == 1:
        real = real.squeeze(1)
    if len(gen.shape) > 2 and gen.shape[1] == 1:
        gen = gen.squeeze(1)

    print("\nComputing genomic-specific metrics...")

    # 1. Genotype distribution analysis
    try:
        # Round to nearest genotype (0, 0.5, 1.0)
        real_genotypes = torch.round(real * 2) / 2
        gen_genotypes = torch.round(gen * 2) / 2

        # Count occurrences of each genotype
        real_counts = torch.bincount(
            real_genotypes.flatten().long() * 2, minlength=3
        ).float()
        gen_counts = torch.bincount(
            gen_genotypes.flatten().long() * 2, minlength=3
        ).float()

        # Normalize to get frequencies
        real_freq = real_counts / real_counts.sum()
        gen_freq = gen_counts / gen_counts.sum()

        # Calculate Jensen-Shannon divergence (symmetric KL divergence)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        real_freq_safe = real_freq + epsilon
        gen_freq_safe = gen_freq + epsilon
        m = 0.5 * (real_freq_safe + gen_freq_safe)

        # Normalize to ensure they sum to 1 after adding epsilon
        real_freq_safe = real_freq_safe / real_freq_safe.sum()
        gen_freq_safe = gen_freq_safe / gen_freq_safe.sum()
        m = m / m.sum()

        # Calculate KL divergences manually to avoid numerical issues
        kl_real_m = torch.sum(real_freq_safe * torch.log(real_freq_safe / m))
        kl_gen_m = torch.sum(gen_freq_safe * torch.log(gen_freq_safe / m))
        js_div = 0.5 * (kl_real_m + kl_gen_m)

        metrics["genotype_js_div"] = js_div.item()
        print(f"Genotype distribution JS divergence: {js_div.item():.4f}")

        # Plot genotype distribution
        plt.figure(figsize=(10, 6))
        labels = ["0.0", "0.5", "1.0"]
        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, real_freq.numpy(), width, label="Real")
        plt.bar(x + width / 2, gen_freq.numpy(), width, label="Generated")

        plt.xlabel("Genotype")
        plt.ylabel("Frequency")
        plt.title("Genotype Distribution Comparison")
        plt.xticks(x, labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "genotype_distribution.png"))
        plt.close()
        print(
            f"Genotype distribution plot saved to: {os.path.join(output_dir, 'genotype_distribution.png')}"
        )

    except Exception as e:
        print(f"Warning: Could not compute genotype distribution: {str(e)}")

    # 2. Minor Allele Frequency (MAF) analysis
    try:
        # Calculate MAF for each SNP (column)
        real_maf = real.mean(dim=0)  # Average across samples
        gen_maf = gen.mean(dim=0)

        # Calculate correlation
        maf_corr = torch.corrcoef(torch.stack([real_maf.flatten(), gen_maf.flatten()]))[
            0, 1
        ]
        metrics["maf_correlation"] = maf_corr.item()
        print(f"MAF correlation: {maf_corr.item():.4f}")

        # Plot MAF comparison
        plt.figure(figsize=(8, 8))
        plt.scatter(real_maf.numpy(), gen_maf.numpy(), alpha=0.5)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("Real MAF")
        plt.ylabel("Generated MAF")
        plt.title("Minor Allele Frequency Comparison")
        plt.savefig(os.path.join(output_dir, "maf_comparison.png"))
        plt.close()
        print(
            f"MAF comparison plot saved to: {os.path.join(output_dir, 'maf_comparison.png')}"
        )

    except Exception as e:
        print(f"Warning: Could not compute MAF correlation: {str(e)}")

    # 3. Population structure analysis via PCA
    try:
        # Perform PCA on real data
        n_components = min(5, min(real.shape[0], real.shape[1]) - 1)
        pca = PCA(n_components=n_components)
        real_pca = pca.fit_transform(real.numpy())

        # Project generated data onto same PC space
        gen_pca = pca.transform(gen.numpy())

        # Calculate correlation between PC coordinates
        pc_corrs = []
        for i in range(n_components):
            corr = np.corrcoef(real_pca[:, i], gen_pca[:, i])[0, 1]
            pc_corrs.append(corr)

        metrics["pca_correlation_mean"] = np.mean(pc_corrs)
        print(
            f"PCA correlation (mean across {n_components} PCs): {np.mean(pc_corrs):.4f}"
        )

        # Plot PCA comparison (first 2 PCs)
        plt.figure(figsize=(10, 8))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.7, label="Real")
        plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.7, label="Generated")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Comparison of Real vs Generated Data")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pca_comparison.png"))
        plt.close()
        print(
            f"PCA comparison plot saved to: {os.path.join(output_dir, 'pca_comparison.png')}"
        )

    except Exception as e:
        print(f"Warning: Could not compute PCA analysis: {str(e)}")

    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate samples using trained SNP diffusion model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (overrides config)",
    )

    return parser.parse_args()


def main():
    """Main function."""

    # Parse arguments
    args = parse_args()

    try:
        print("\nLoading model from checkpoint...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model from checkpoint
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        config = model.hparams  # model config used during training
        model = model.to(device)  # move model to device
        model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    if "checkpoints" in str(checkpoint_path):
        base_dir = checkpoint_path.parent.parent
    else:
        base_dir = checkpoint_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInference results will be saved to: {output_dir}")

    # Load all real SNP data from test split (ground truth)
    print("\nLoading full test dataset...")
    model.setup("test")  # Ensure test dataset is initialized
    test_loader = model.test_dataloader()

    # Collect all test samples
    real_samples = []
    print("Loading all test batches...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to same device as model
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            real_samples.append(batch)
    real_samples = torch.cat(real_samples, dim=0)

    print(f"\nReal samples shape: {real_samples.shape}")
    print(f"Real samples unique values: {torch.unique(real_samples)}")

    # Generate synthetic sequences
    try:
        with torch.no_grad():

            # Match to real sample shape
            num_samples = real_samples.shape[0]
            print(f"\nGenerating {num_samples} synthetic sequences...")
            gen_samples = model.generate_samples(
                num_samples=num_samples, discretize=False
            )

            # Check for NaN values in generated samples
            if torch.isnan(gen_samples).any():
                print(
                    "Warning: Generated samples contain NaN values. Attempting to fix..."
                )
                # gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

            # Save samples
            torch.save(gen_samples, output_dir / "synthetic_sequences.pt")

            # Print statistics
            print(f"\nGen samples shape: {gen_samples.shape}")
            print(f"Gen samples unique values: {torch.unique(gen_samples)}")
            print(f"First gen samples: {gen_samples[:, :1]}")

            # ----------------------------------------------------------------------

            # Generate comparison plots
            print("\nGenerating comparison plots...")
            plot_path = output_dir / "sample_comparison.png"
            plot_comparison(real_samples.cpu(), gen_samples.cpu(), plot_path)
            print(f"Comparison plots saved to: {plot_path}")

            # Compute genomic metrics
            print("\nComputing genomic metrics...")
            metrics = compute_genomic_metrics(
                real_samples.cpu(), gen_samples.cpu(), output_dir
            )

            # Save metrics to JSON file
            metrics_path = output_dir / "genomic_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics.items()
                    },
                    f,
                    indent=4,
                )
            print(f"Genomic metrics saved to: {metrics_path}")

            # Print summary of metrics
            print("\nGenome Diffusion Model Evaluation Metrics:")
            print("-" * 50)
            for k, v in metrics.items():
                print(
                    f"{k}: {v:.4f}"
                    if isinstance(v, (float, np.float32, np.float64))
                    else f"{k}: {v}"
                )
            print("-" * 50)

            # Generate reverse diffusion visualization
            print("\nGenerating reverse diffusion visualization...")
            reverse_samples = []
            timesteps = []

            # Start with noise (t=T)
            x = torch.randn((1,) + model._data_shape, device=model.device)
            reverse_samples.append(x)
            timesteps.append(model._forward_diffusion.tmax)

            # Reverse diffusion process
            for t in range(model._forward_diffusion.tmax, 0, -100):
                x = model._reverse_process_step(x, t)
                x = torch.clamp(x, -5.0, 5.0)  # Prevent explosion
                reverse_samples.append(x)
                timesteps.append(t)

            # Final step (t=0)
            x = torch.clamp(x, 0, 1)
            reverse_samples.append(x)
            timesteps.append(0)

            # Plot reverse diffusion
            reverse_samples = torch.cat(reverse_samples, dim=0)
            plot_path = output_dir / "reverse_diffusion.png"
            plot_sample_grid(
                reverse_samples.cpu(),
                plot_path,
                "Reverse Diffusion Process",
                timesteps=timesteps,
            )
            print(f"Reverse diffusion visualization saved to: {plot_path}")

    except Exception as e:
        raise RuntimeError(f"Sample generation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
