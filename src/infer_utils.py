#!/usr/bin/env python
# coding: utf-8

"""Utility functions for inference and evaluation of diffusion models.

This module contains reusable functions for:
1. MAF (Minor Allele Frequency) analysis
2. Visualization of samples and diffusion process
3. Genomic metrics calculation
4. PCA analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from .utils import set_seed

# === Helper Functions ===


def validate_samples(real_samples, generated_samples, flatten=False):
    """
    Validate shape, convert tensors to numpy, and (optionally) flatten
    both real and generated samples.
    Returns (real_out, gen_out)
    """
    try:
        if real_samples.shape != generated_samples.shape:
            raise ValueError(
                f"Input tensors must have same shape. Got {real_samples.shape} and {generated_samples.shape}"
            )
        if len(real_samples.shape) != 3:
            raise ValueError(
                f"Samples must be 3D tensors with shape [batch_size, channels, seq_len]. "
                f"Got shape {real_samples.shape}"
            )
    except Exception as e:
        print(f"Warning: Error in validate_samples: {e}")
        return None, None

    real = real_samples.cpu().numpy()
    gen = generated_samples.cpu().numpy()
    if flatten:
        real = real.flatten()
        gen = gen.flatten()
    return real, gen


def count_genotype_values(arr, genotype_values=[0.0, 0.5, 1.0]):
    """
    Count occurrences of each genotype value in arr (with float tolerance).

    Args:
        arr: Array-like object containing genotype values
        genotype_values: List of expected genotype values, default [0.0, 0.25, 0.5]
                         (scaled from original [0.0, 0.5, 1.0])

    Returns:
        tuple: Counts for each genotype value in the order provided
    """
    arr = np.asarray(arr)
    counts = []

    for value in genotype_values:
        count = np.sum(np.isclose(arr, value, atol=1e-3))
        counts.append(int(count))

    return tuple(counts)


# === Sample Analysis ===


import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_comparison(real, gen, save_path):
    """Plot real and generated sample side by side as line plots."""
    real_np = real.flatten().cpu().numpy()
    gen_np = gen.flatten().cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(real_np, label="Real", color="tab:blue")
    axes[0].set_title("Real Sample")
    axes[1].plot(gen_np, label="Generated", color="tab:orange")
    axes[1].set_title("Generated Sample")
    for ax in axes:
        ax.set_ylabel("Value")
        ax.set_ylim(0, 0.5)
    axes[1].set_xlabel("Position")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved sample comparison plot to {save_path}")


def print_sample_stats(sample, label="Sample"):
    """Print unique values, mean, std, and min/max."""
    sample_np = sample.flatten().cpu().numpy()
    print(f"\n{label} statistics:")
    print(f"  Unique values: {np.unique(sample_np)}")
    print(f"  Mean: {np.mean(sample_np):.4f}")
    print(f"  Std: {np.std(sample_np):.4f}")
    print(f"  Min: {np.min(sample_np):.4f}, Max: {np.max(sample_np):.4f}")


def compare_samples(
    real_samples, generated_samples, save_path, genotype_values=[0.0, 0.5, 1.0]
):
    """
    Compare and plot the genotype distributions for real and generated samples.
    Prints stats and saves a bar plot to save_path.

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        save_path (str): Path to save the plot
        genotype_values (list): Expected genotype values, default [0.0, 0.25, 0.5]

    Returns:
        None
    """

    # Validate and Flatten samples
    real_flat, gen_flat = validate_samples(
        real_samples, generated_samples, flatten=True
    )

    # Count genotype values
    real_counts = count_genotype_values(real_flat, genotype_values)
    gen_counts = count_genotype_values(gen_flat, genotype_values)

    # Print stats
    print("Real samples:")
    for i, value in enumerate(genotype_values):
        print(f"  {value:.2f}:   {real_counts[i]}")
    print("Generated samples:")
    for i, value in enumerate(genotype_values):
        print(f"  {value:.2f}:   {gen_counts[i]}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot value distributions as density with more robust binning
    # Use fewer bins and explicit range to avoid binning errors with extreme values
    hist_params = {
        "bins": 20,  # Reduced from 50 to be more robust
        "alpha": 0.5,
        "density": True,
        "range": (0.0, 0.5),  # Explicitly set range to expected SNP values
    }
    axes[0].hist(real_flat, **hist_params, label="Real", color="tab:blue")
    axes[0].hist(gen_flat, **hist_params, label="Generated", color="tab:orange")
    axes[0].set_title("Value Distribution (Density)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Bar plot
    labels = [f"{value:.2f}" for value in genotype_values]

    x = np.arange(len(labels))
    width = 0.35

    axes[1].bar(x - width / 2, real_counts, width, label="Real", color="tab:blue")
    axes[1].bar(x + width / 2, gen_counts, width, label="Generated", color="tab:orange")

    # Annotate bars
    for bar in axes[1].patches:
        height = bar.get_height()
        axes[1].annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes[1].set_xlabel("Genotype Value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Genotype Distribution (0, 0.5, 1.0)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Genotype distribution plot saved to: {save_path}")


def visualize_samples(
    real_samples,
    generated_samples,
    save_path,
    max_seq_len=10000,
    genotype_values=[0.0, 0.5, 1.0],
):
    """Plot comparison between real and generated samples.

    Creates a 2x2 grid showing:
    1. Real data pattern (first 100 positions)
    2. Generated data pattern (first 100 positions)
    3. Value distributions for both datasets

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        save_path (str): Path to save the plot
        max_seq_len (int, optional): Max sequence length to plot in upper plots. Default is 10000.
        genotype_values (list): Expected genotype values, default [0.0, 0.25, 0.5]

    Returns:
        None
    """

    # Flatten samples
    real, gen = validate_samples(real_samples, generated_samples)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Real and Generated Samples Comparison")

    # Limit sequence length for upper plots
    real_seq = real[0].flatten()[:max_seq_len]
    gen_seq = gen[0].flatten()[:max_seq_len]

    # Plot real sample sequence (limited length)
    axes[0, 0].plot(real_seq, color="tab:blue")
    axes[0, 0].set_title(f"Real Sample Sequence (first {len(real_seq)})")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Value")

    # Plot generated sample sequence (limited length)
    axes[0, 1].plot(gen_seq, color="tab:orange")
    axes[0, 1].set_title(f"Generated Sample Sequence (first {len(gen_seq)})")
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_ylabel("Value")

    # Plot heatmaps
    first_100 = min(100, real_samples.shape[-1])

    # Custom colormap: 0 → blue, 0.5 → green, 1 → red
    # cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
    cmap = plt.cm.viridis

    # Set min/max values based on genotype values
    vmin = min(genotype_values)
    vmax = max(genotype_values)

    # Bottom left: Real data heatmap (first channel, first 100 positions)
    im_real = axes[1, 0].imshow(
        real[0].reshape(1, -1)[:, :first_100],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title("Real Data Pattern (first 100 positions)")
    axes[1, 0].set_yticks([])
    plt.colorbar(
        im_real,
        ax=axes[1, 0],
        orientation="vertical",
        fraction=0.05,
        pad=0.04,
        label="Value",
    )

    # Bottom right: Generated data heatmap (first channel, first 100 positions)
    im_gen = axes[1, 1].imshow(
        gen[0].reshape(1, -1)[:, :first_100],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title("Generated Data Pattern (first 100 positions)")
    axes[1, 1].set_yticks([])
    plt.colorbar(
        im_gen,
        ax=axes[1, 1],
        orientation="vertical",
        fraction=0.05,
        pad=0.04,
        label="Value",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Genotype visualization plot saved to: {save_path}")


# === MAF Analysis Functions ===
# These functions handle the calculation and analysis of Minor Allele Frequency (MAF),
# which is a key metric in genomic data analysis. MAF represents the frequency of the
# less common allele at a specific position in the genome.


def calculate_maf(samples, max_value=1.0):
    """Calculate Minor Allele Frequency (MAF) for a batch of samples.

    This is the core function for MAF calculation. It processes the input data to extract
    both raw allele frequencies and their corresponding MAF values. For SNP data, MAF is
    calculated as min(freq, max_value-freq) where freq is the mean value across samples at each position.

    Args:
        samples: Input samples tensor of shape [batch_size, channels, seq_len]
        max_value: Maximum possible value for genotypes (default: 0.5 for scaled data)

    Returns:
        tuple: (maf, freq) where:
            - maf: Minor allele frequencies (tensor)
            - freq: Raw allele frequencies (tensor)
    """
    # Convert to tensor if numpy
    if not torch.is_tensor(samples):
        samples = torch.from_numpy(samples)

    # Ensure 3D shape [batch_size, channels, seq_len]
    if len(samples.shape) != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {samples.shape}")

    # Calculate raw frequencies (mean across batch dimension)
    freq = torch.mean(samples, dim=0).squeeze()

    # Convert to minor allele frequency
    maf = torch.minimum(freq, max_value - freq)

    return maf, freq


def calculate_maf_stats(maf, genotype_values=[0.0, 0.5, 1.0]):
    """Calculate basic statistical measures for MAF values.

    This function computes summary statistics (mean, median, standard deviation)
    for a set of MAF values, which is useful for comparing distributions between
    real and generated data.

    Args:
        maf: Array/Tensor of MAF values
        genotype_values: List of expected genotype values, default [0.0, 0.25, 0.5]

    Returns:
        dict: Dictionary containing:
            - mean: Mean MAF value
            - median: Median MAF value
            - std: Standard deviation of MAF values
            - min_maf: Minimum MAF value
            - max_maf: Maximum MAF value
            - num_half_maf: Number of MAF values equal to middle genotype value
            - min_freq: Minimum frequency value (same as min_maf)
            - max_freq: Maximum frequency value (same as max_maf)
            - num_half_freq: Number of frequency values equal to middle genotype value
    """
    # Convert to tensor if numpy
    if not torch.is_tensor(maf):
        maf = torch.from_numpy(maf)

    # Calculate min/max values
    min_val = float(torch.min(maf).item())
    max_val = float(torch.max(maf).item())

    # Count middle genotype values with tolerance
    # Use the middle value from genotype_values (typically 0.25 for scaled data)
    middle_value = genotype_values[len(genotype_values) // 2]
    half_count = torch.sum(
        torch.isclose(maf, torch.tensor(middle_value), atol=1e-3)
    ).item()

    return {
        "mean": float(torch.mean(maf).item()),
        "median": float(torch.median(maf).item()),
        "std": float(torch.std(maf).item()),
        "min_maf": min_val,
        "max_maf": max_val,
        "num_half_maf": int(half_count),
        "min_freq": min_val,  # For backward compatibility
        "max_freq": max_val,  # For backward compatibility
        "num_half_freq": int(half_count),  # For backward compatibility
    }


def analyze_maf_distribution(
    samples, save_path, bin_width=0.001, genotype_values=[0.0, 0.5, 1.0], max_value=1.0
):
    """Analyze and visualize the Minor Allele Frequency (MAF) distribution for a dataset.

    This function calculates MAF values, prints summary statistics, and creates a
    histogram visualization of the MAF distribution. It also analyzes the expected
    peak spacing based on the number of samples and presence of heterozygous values.

    The visualization shows the theoretical peaks that should appear in the MAF distribution
    based on population genetics principles, which is useful for assessing the quality of
    both real and generated genomic data.

    Args:
        samples: Tensor of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]
        save_path: Path to save the MAF histogram plot
        bin_width: Width of histogram bins (default: 0.001)
        genotype_values: List of expected genotype values, default [0.0, 0.25, 0.5]
        max_value: Maximum possible value for genotypes (default: 0.5 for scaled data)

    Returns:
        tuple: (maf, freq) where:
            - maf: numpy.ndarray of Minor Allele Frequency values
            - freq: numpy.ndarray of raw allele frequencies
    """
    # Calculate MAF and raw frequencies using the dedicated function
    maf, freq = calculate_maf(samples, max_value=max_value)

    # Convert samples to numpy for shape information (needed for peak spacing calculation)
    # We need to do this here since we need the original sample shape, not the processed MAF
    samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples
    if len(samples_np.shape) == 3:
        samples_np = samples_np.squeeze(1)

    # Print frequency distribution details
    print(f"\nFrequency analysis for {save_path}:")
    min_freq = freq.min().item() if torch.is_tensor(freq) else float(np.min(freq))
    max_freq = freq.max().item() if torch.is_tensor(freq) else float(np.max(freq))
    min_maf = maf.min().item() if torch.is_tensor(maf) else float(np.min(maf))
    max_maf = maf.max().item() if torch.is_tensor(maf) else float(np.max(maf))

    # Count 0.5 values
    num_half_freq = (
        torch.sum((freq - 0.5).abs() < 0.001).item()
        if torch.is_tensor(freq)
        else int(np.sum(np.abs(freq - 0.5) < 0.001))
    )
    num_half_maf = (
        torch.sum((maf - 0.5).abs() < 0.001).item()
        if torch.is_tensor(maf)
        else int(np.sum(np.abs(maf - 0.5) < 0.001))
    )

    print(f"Raw frequency range: [{min_freq:.3f}, {max_freq:.3f}]")
    print(f"Number of 0.5 frequencies: {num_half_freq}")
    print(f"MAF range: [{min_maf:.3f}, {max_maf:.3f}]")
    print(f"Number of MAF = 0.5: {num_half_maf}\n")

    # Plot setup - convert to numpy for plotting
    plt.figure(figsize=(15, 8))
    bins = np.arange(0, 0.5 + bin_width, bin_width)
    maf_np = maf.cpu().numpy() if torch.is_tensor(maf) else maf
    plt.hist(maf_np, bins=bins, alpha=0.7, density=True)

    # Calculate expected peak spacing
    num_samples = samples_np.shape[0]
    has_half = 0.5 in np.unique(samples_np)

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

        # Convert to numpy if needed for peak counting
        if torch.is_tensor(maf):
            maf_np = maf.cpu().numpy()
        else:
            maf_np = maf
        peak_counts[i] = np.sum((maf_np >= peak_range[0]) & (maf_np < peak_range[1]))

    # Calculate MAF statistics
    stats = calculate_maf_stats(maf, genotype_values=genotype_values)
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

    return maf, freq


def compare_maf_distributions(
    real_maf,
    gen_maf,
    output_dir,
    bin_width=0.001,
    genotype_values=[0.0, 0.5, 1.0],
    max_value=1.0,
):
    """Plot and compare MAF distributions between real and generated genomic data.

    This function creates two key visualizations:
    1. A histogram comparing the MAF distributions of real and generated data
    2. A scatter plot showing the correlation between real and generated MAF values

    These visualizations are important for assessing how well the generated data
    captures the allele frequency patterns of the real data, which is a key metric
    in genomic data generation.

    Args:
        real_maf: MAF values for real data
        gen_maf: MAF values for generated data
        output_dir: Directory to save output plots
        bin_width: Width of histogram bins (default: 0.001)
        genotype_values: List of expected genotype values, default [0.0, 0.25, 0.5]
        max_value: Maximum possible value for genotypes (default: 0.5 for scaled data)

    Returns:
        float: Correlation coefficient between real and generated MAF
    """
    # Basic path handling
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy only for plotting and correlation

    # Plot real MAF distribution
    plt.figure(figsize=(15, 8))
    bins = np.arange(0, 0.5 + bin_width, bin_width)

    real_maf_np = real_maf.cpu().numpy() if torch.is_tensor(real_maf) else real_maf
    gen_maf_np = gen_maf.cpu().numpy() if torch.is_tensor(gen_maf) else gen_maf

    plt.hist(real_maf_np, bins=bins, alpha=0.7, density=True, label="Real")
    plt.hist(gen_maf_np, bins=bins, alpha=0.7, density=True, label="Generated")

    # Calculate correlation
    maf_corr = np.corrcoef(real_maf_np, gen_maf_np)[0, 1]

    # Plot formatting
    plt.title(f"Minor Allele Frequency Distribution Comparison (r = {maf_corr:.4f})")
    plt.xlabel("MAF")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add statistics text box
    real_stats = calculate_maf_stats(real_maf)
    gen_stats = calculate_maf_stats(gen_maf)

    stats_text = (
        f"Real MAF Stats:\n"
        f'Mean: {real_stats["mean"]:.4f}\n'
        f'Median: {real_stats["median"]:.4f}\n'
        f'Std: {real_stats["std"]:.4f}\n\n'
        f"Generated MAF Stats:\n"
        f'Mean: {gen_stats["mean"]:.4f}\n'
        f'Median: {gen_stats["median"]:.4f}\n'
        f'Std: {gen_stats["std"]:.4f}\n'
        f"Correlation: {maf_corr:.4f}"
    )

    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "maf_comparison.png", dpi=300)
    plt.close()

    # Also create a scatter plot of real vs generated MAF
    plt.figure(figsize=(10, 10))

    # Convert to numpy for plotting if needed
    real_maf_np = real_maf.cpu().numpy() if torch.is_tensor(real_maf) else real_maf
    gen_maf_np = gen_maf.cpu().numpy() if torch.is_tensor(gen_maf) else gen_maf

    plt.scatter(real_maf_np, gen_maf_np, alpha=0.5, s=5)
    plt.plot([0, 0.5], [0, 0.5], "r--")  # Diagonal line
    plt.title(f"MAF Correlation (r = {maf_corr:.4f})")
    plt.xlabel("Real MAF")
    plt.ylabel("Generated MAF")
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "maf_correlation.png", dpi=300)
    plt.close()

    return maf_corr


# === Genomic Metrics Functions ===
# These functions handle the calculation and analysis of genomic metrics.
# It calculates allele frequency correlation, perform PCA, and visualize.
def compute_genomic_metrics(real_samples, generated_samples, output_dir):
    """Compute genomic-specific metrics for generated samples.

    Args:
        real_samples: Tensor of real SNP data [batch_size, channels, seq_len]
        generated_samples: Tensor of generated SNP data [batch_size, channels, seq_len]
        output_dir: Directory to save plots

    Returns:
        dict: Dictionary of genomic metrics
    """
    # Convert tensors to numpy arrays
    real = real_samples.cpu().numpy()
    gen = generated_samples.cpu().numpy()

    # Ensure 2D shape [batch_size, seq_len]
    if len(real.shape) == 3:
        real = real.squeeze(1)
    if len(gen.shape) == 3:
        gen = gen.squeeze(1)

    metrics = {}

    # 1. Calculate allele frequency correlation
    real_af = np.mean(real, axis=0)
    gen_af = np.mean(gen, axis=0)
    af_corr = np.corrcoef(real_af, gen_af)[0, 1]
    metrics["allele_frequency_correlation"] = af_corr

    # 2. Calculate minor allele frequency correlation
    real_maf = np.minimum(real_af, 1 - real_af)
    gen_maf = np.minimum(gen_af, 1 - gen_af)
    maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]
    metrics["maf_correlation"] = maf_corr

    # 3. Calculate heterozygosity metrics
    real_het = 2 * real_af * (1 - real_af)
    gen_het = 2 * gen_af * (1 - gen_af)
    het_corr = np.corrcoef(real_het, gen_het)[0, 1]
    metrics["heterozygosity_correlation"] = het_corr
    metrics["real_mean_heterozygosity"] = float(np.mean(real_het))
    metrics["gen_mean_heterozygosity"] = float(np.mean(gen_het))

    # 4. Run PCA analysis
    try:
        # Combine data for PCA
        combined = np.vstack([real, gen])

        # Fit PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)

        # Split back into real and generated
        real_pca = pca_result[: real.shape[0]]
        gen_pca = pca_result[real.shape[0] :]

        # Plot PCA results
        plt.figure(figsize=(10, 8))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label="Real")
        plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label="Generated")

        # Add explained variance
        explained_var = pca.explained_variance_ratio_
        plt.title("PCA of Real and Generated SNP Data")
        plt.xlabel(f"PC1 ({explained_var[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({explained_var[1]:.2%} variance)")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pca_analysis.png", dpi=300)
        plt.close()

        # Calculate distance between centroids
        real_centroid = np.mean(real_pca, axis=0)
        gen_centroid = np.mean(gen_pca, axis=0)
        centroid_distance = np.linalg.norm(real_centroid - gen_centroid)
        metrics["pca_centroid_distance"] = float(centroid_distance)

    except Exception as e:
        print(f"Warning: PCA analysis failed: {e}")
        metrics["pca_centroid_distance"] = None

    # 5. Plot MAF comparison scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(real_maf, gen_maf, alpha=0.5, s=5)
    plt.plot([0, 0.5], [0, 0.5], "r--")  # Diagonal line
    plt.title(f"MAF Correlation (r = {maf_corr:.4f})")
    plt.xlabel("Real MAF")
    plt.ylabel("Generated MAF")
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "maf_correlation.png", dpi=300)
    plt.close()

    return metrics


# === Standard Diffusion Process Functions ===
# These functions handle the generation and visualization of the standard (full-noise)
# diffusion process, leveraging the DiffusionModel's built-in functionality.
def visualize_diffusion(samples, save_path, title, timesteps=None):
    """Plot a grid of samples showing the diffusion process.

    Creates a grid visualization showing how samples evolve during the diffusion process.
    Each row represents a different timestep, allowing us to see how the noise level
    changes throughout the process.

    Args:
        samples: Tensor of samples to plot [num_steps, batch_size, channels, seq_len]
        save_path: Path to save the plot
        title: Title for the plot
        timesteps: List of timesteps corresponding to each sample
    """
    n_samples = min(samples.shape[0], 20)  # Show at most 20 timesteps
    seq_length = samples.shape[-1]

    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 2 * n_samples))
    fig.suptitle(title)

    # If only one sample, axes won't be an array
    if n_samples == 1:
        axes = [axes]

    # Plot each sample
    for i in range(n_samples):
        axes[i].imshow(
            samples[i, 0, :].detach().cpu().numpy().reshape(1, -1),
            aspect="auto",
            cmap="viridis",
        )
        if timesteps is not None:
            axes[i].set_title(f"t = {timesteps[i]}")
        else:
            axes[i].set_title(f"Step {i}")
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# === Mid-Noise Diffusion Functions ===
# These functions handle the generation and visualization of samples starting from
# mid-noise levels (t < tmax). This allows testing model behavior with less noisy
# starting points and can be useful for analyzing the denoising process.


def generate_samples_mid_step(
    model, num_samples, mid_timestep=500, denoise_step=10, discretize=False, seed=42
):
    """Generate samples starting from a mid-noise level.

    Args:
        model: The diffusion model to use for generation
        num_samples: Number of samples to generate
        mid_timestep: Timestep to start the reverse diffusion from (default: 500)
        denoise_step: Number of timesteps to skip in reverse diffusion (default: 10)
        discretize: Whether to discretize the final output to 0, 0.5, 1.0 values

    Returns:
        torch.Tensor: Generated samples starting from mid-noise level
    """
    if mid_timestep >= model.forward_diffusion.tmax:
        raise ValueError(
            f"mid_timestep ({mid_timestep}) must be less than tmax ({model.forward_diffusion.tmax})"
        )

    # Create noise for the specified number of samples
    dummy_batch = torch.zeros((num_samples,) + model._data_shape, device=model.device)

    # Start from mid-noise level
    with torch.no_grad():
        # Create noise tensor for the specified timestep
        # Set random seed for reproducible noise initialization
        set_seed(seed)

        # Initialize noise tensor
        x = torch.randn_like(dummy_batch)

        t = torch.full(
            (num_samples,), mid_timestep, device=model.device, dtype=torch.long
        )

        # Sample noise at mid_timestep
        x = model.forward_diffusion.sample(dummy_batch, t, x)

        # Print initial statistics
        # print(f"Initial noise stats at t={mid_timestep} - mean: {x.mean():.4f}, std: {x.std():.4f}")

        # Custom reverse diffusion process starting from mid_timestep
        print(
            f"Starting reverse diffusion from t={mid_timestep} to t={model.forward_diffusion.tmin} with step {denoise_step}"
        )

        # Perform reverse diffusion from mid_timestep to 0
        for t in reversed(
            range(model.forward_diffusion.tmin, mid_timestep + 1, denoise_step)
        ):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            x = model.reverse_diffusion.reverse_diffusion_step(x, t_tensor)

            # Print statistics every 100 steps
            if t % 100 == 0 or t == model.forward_diffusion.tmin:
                # print(f"Step {t} stats - mean: {x.mean():.4f}, std: {x.std():.4f}")
                pass

        # Clamp to valid range [0, 0.5] for scaled data
        x = torch.clamp(x, 0, 0.5)

        # Discretize if requested
        if discretize:
            x = torch.round(x * 2) / 2

        # print(f"Final sample stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

    return x


# === Visualize Reverse Denoising ===
def visualize_reverse_diffusion(
    model,
    output_dir,
    start_timestep=None,  # None means full noise
    step_size=100,
    num_samples=1,
    save_prefix="viz_",
    discretize=False,
    seed=42,
):
    """
    Visualize the reverse denoising process from any starting timestep.
    Args:
        model: DiffusionModel instance
        output_dir: Directory to save visualization
        start_timestep: Timestep to start from (None for full noise)
        step_size: Step size for visualization (higher = fewer steps shown)
        num_samples: Number of samples to visualize (default: 1)
        save_prefix: Prefix for saved plot filename
        discretize: Whether to discretize final output
    """

    if start_timestep is None:
        start_timestep = model.forward_diffusion.tmax

    batch_shape = (num_samples,) + model._data_shape
    # Set the seed for reproducibility
    set_seed(seed)

    # Start from pure noise at the given timestep
    x = torch.randn(batch_shape, device=model.device)
    t = torch.full(
        (num_samples,), start_timestep, device=model.device, dtype=torch.long
    )
    x = model.forward_diffusion.sample(torch.zeros_like(x), t, x)

    reverse_samples = [x.clone()]
    timesteps = [start_timestep]

    for t_val in reversed(
        range(model.forward_diffusion.tmin, start_timestep + 1, step_size)
    ):
        if t_val == model.forward_diffusion.tmin:
            continue
        t_tensor = torch.full((x.size(0),), t_val, device=x.device, dtype=torch.long)
        x = model.reverse_diffusion.reverse_diffusion_step(x, t_tensor)
        x = torch.clamp(x, -5.0, 5.0)
        reverse_samples.append(x.clone())
        timesteps.append(t_val)

    # Final step (fully denoised)
    t_tensor = torch.full(
        (x.size(0),), model.forward_diffusion.tmin, device=x.device, dtype=torch.long
    )
    x = model.reverse_diffusion.reverse_diffusion_step(x, t_tensor)
    x = torch.clamp(x, 0, 0.5)  # Clamp to [0, 0.5] for scaled data
    reverse_samples.append(x.clone())
    timesteps.append(model.forward_diffusion.tmin)

    if discretize:
        x = torch.round(x * 2) / 2

    reverse_samples_tensor = torch.stack(reverse_samples)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{save_prefix}reverse_diffusion_t{start_timestep}.png"
    visualize_diffusion(
        reverse_samples_tensor.cpu(),
        plot_path,
        f"Reverse Diffusion Process (t={start_timestep} to t={model.forward_diffusion.tmin})",
        timesteps=timesteps,
    )
    print(f"Visualization saved to: {plot_path}")
