#!/usr/bin/env python
# coding: utf-8

"""Script for analyzing Minor Allele Frequency (MAF) distribution in SNP data.

Example:
    python infer.py --checkpoint path/to/last.ckpt

Outputs:
- MAF distribution plots for real and generated data
- MAF statistics in JSON format
- Correlation between real and generated MAF distributions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from diffusion.diffusion_model import DiffusionModel


def calculate_maf_stats(maf):
    """Calculate MAF statistics.
    Args:
        maf: numpy array of MAF values
    Returns:
        dict: Dictionary of MAF statistics
    """
    # Explicitly convert numpy values to Python floats
    mean_val = float(np.mean(maf))
    median_val = float(np.median(maf))
    std_val = float(np.std(maf))

    return {"mean": mean_val, "median": median_val, "std": std_val}


def analyze_maf_distribution(samples, save_path, bin_width=0.001):
    """Analyze and plot the Minor Allele Frequency (MAF) distribution.

    Args:
        samples: Tensor of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]
        save_path: Path to save the MAF histogram plot
        bin_width: Width of histogram bins (default: 0.001 for fine-grained analysis)
    Returns:
        numpy.ndarray: Array of MAF values
    """
    # Convert tensor to numpy array on CPU
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    # Ensure 2D shape [batch_size, seq_len]
    if len(samples.shape) == 3:
        samples = samples.squeeze(1)

    # Calculate allele frequencies
    freq = np.mean(samples, axis=0)

    # Print frequency distribution details
    print(f"\nFrequency analysis for {save_path}:")
    print(f"Raw frequency range: [{freq.min():.3f}, {freq.max():.3f}]")
    print(f"Number of 0.5 frequencies: {np.sum(np.abs(freq - 0.5) < 0.001)}")

    # Convert to minor allele frequency
    maf = np.minimum(freq, 1 - freq)
    print(f"MAF range: [{maf.min():.3f}, {maf.max():.3f}]")
    print(f"Number of MAF = 0.5: {np.sum(np.abs(maf - 0.5) < 0.001)}\n")

    # Plot setup
    plt.figure(figsize=(15, 8))
    bins = np.arange(0, 0.5 + bin_width, bin_width)
    plt.hist(maf, bins=bins, alpha=0.7, density=True)

    # Calculate expected peak spacing
    num_samples = samples.shape[0]
    has_half = 0.5 in np.unique(samples)

    # If we have 0.5 values, spacing is 1/(2*num_samples)
    # If no 0.5 values, spacing is 1/num_samples
    peak_spacing = 1.0 / (2 * num_samples) if has_half else 1.0 / num_samples
    num_peaks = int(0.5 / peak_spacing)

    # Add vertical lines for expected peaks
    for i in range(1, num_peaks + 1):
        peak = i * peak_spacing
        plt.axvline(x=peak, color="r", linestyle="--", alpha=0.3)

    # Plot formatting
    spacing_text = f"1/{2*num_samples}" if has_half else f"1/{num_samples}"
    plt.title(
        f"Minor Allele Frequency Distribution (spacing: {spacing_text})", fontsize=12
    )
    plt.xlabel("MAF", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Calculate peak statistics
    peak_counts = np.zeros(num_peaks)
    for i in range(num_peaks):
        peak_center = (i + 1) * peak_spacing
        peak_range = (peak_center - bin_width / 2, peak_center + bin_width / 2)
        peak_counts[i] = np.sum((maf >= peak_range[0]) & (maf < peak_range[1]))

    # Add statistics text box
    stats = calculate_maf_stats(maf)
    # Convert peak values to a list of floats
    peak_indices = np.argsort(peak_counts)[-3:][::-1]
    peak_values = [float((i + 1) * peak_spacing) for i in peak_indices]

    stats_text = (
        f"Statistics:\n"
        f'Mean MAF: {stats["mean"]:.4f}\n'
        f'Median MAF: {stats["median"]:.4f}\n'
        f'Std MAF: {stats["std"]:.4f}\n'
        f"Peak spacing: {peak_spacing:.6f}\n"
        f"Strongest peaks at: {', '.join(f'{v:.6f}' for v in peak_values)}\n"
        f"Num positions: {len(maf)}"
    )
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return maf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Minor Allele Frequency (MAF) distribution in SNP data"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
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
        # Match to real sample shape
        num_samples = real_samples.shape[0]
        print(f"\nGenerating {num_samples} synthetic sequences...")
        gen_samples = model.generate_samples(num_samples=num_samples, discretize=False)

        # Check for NaN values in generated samples
        if torch.isnan(gen_samples).any():
            print("Warning: Generated samples contain NaN values. Attempting to fix...")
            # gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

        # Save samples
        torch.save(gen_samples, output_dir / "synthetic_sequences.pt")

        # Print statistics
        print(f"\nGen samples shape: {gen_samples.shape}")
        print(f"Gen samples unique values: {torch.unique(gen_samples)}")
        print(f"First gen samples: {gen_samples[:, :1]}")

        # ----------------------------------------------------------------------

        # Analyze MAF distribution
        print("\nAnalyzing MAF distribution...")
        real_maf = analyze_maf_distribution(
            real_samples, output_dir / "real_maf_distribution.png"
        )
        gen_maf = analyze_maf_distribution(
            gen_samples, output_dir / "gen_maf_distribution.png"
        )

        # Calculate MAF correlation
        maf_corr = torch.corrcoef(
            torch.stack([torch.tensor(real_maf), torch.tensor(gen_maf)])
        )[0, 1]
        print(f"\nMAF correlation between real and generated data: {maf_corr:.4f}")

        # Save MAF statistics
        maf_stats = {
            "real": calculate_maf_stats(real_maf),
            "generated": calculate_maf_stats(gen_maf),
            "correlation": float(maf_corr),
        }

        with open(output_dir / "maf_statistics.json", "w") as f:
            json.dump(maf_stats, f, indent=4)
            print(f"MAF statistics saved to: {output_dir / 'maf_statistics.json'}")

    except Exception as e:
        raise RuntimeError(f"Sample generation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
