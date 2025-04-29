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

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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


def count_genotype_values(arr):
    """
    Count occurrences of 0.0, 0.5, and 1.0 in arr (with float tolerance).
    Returns tuple (count_0, count_05, count_1)
    """
    arr = np.asarray(arr)
    count_0 = np.sum(np.isclose(arr, 0.0, atol=1e-3))
    count_05 = np.sum(np.isclose(arr, 0.5, atol=1e-3))
    count_1 = np.sum(np.isclose(arr, 1.0, atol=1e-3))
    return int(count_0), int(count_05), int(count_1)


# === Sample Analysis ===


def compare_samples(real_samples, generated_samples, save_path):
    """
    Compare and plot the genotype distributions (counts of 0, 0.5, 1.0) for
    real and generated samples. Prints stats and saves a bar plot to save_path.

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        save_path (str): Path to save the plot

    Returns:
        None
    """

    # Validate and Flatten samples
    real_flat, gen_flat = validate_samples(
        real_samples, generated_samples, flatten=True
    )

    # Count genotype values
    real_0, real_05, real_1 = count_genotype_values(real_flat)
    gen_0, gen_05, gen_1 = count_genotype_values(gen_flat)

    # Print stats
    print("Real samples:")
    print(f"  0.0:   {real_0}")
    print(f"  0.5:   {real_05}")
    print(f"  1.0:   {real_1}")
    print("Generated samples:")
    print(f"  0.0:   {gen_0}")
    print(f"  0.5:   {gen_05}")
    print(f"  1.0:   {gen_1}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot value distributions as density
    hist_params = {"bins": 50, "alpha": 0.5, "density": True}
    axes[0].hist(real_flat, **hist_params, label="Real", color="tab:blue")
    axes[0].hist(gen_flat, **hist_params, label="Generated", color="tab:orange")
    axes[0].set_title("Value Distribution (Density)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Bar plot
    labels = ["0.0", "0.5", "1.0"]
    real_counts = [real_0, real_05, real_1]
    gen_counts = [gen_0, gen_05, gen_1]

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


def visualize_samples(real_samples, generated_samples, save_path, max_seq_len=10000):
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

    # Bottom left: Real data heatmap (first channel, first 100 positions)
    im_real = axes[1, 0].imshow(
        real[0].reshape(1, -1)[:, :first_100],
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
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
        vmin=0.0,
        vmax=1.0,
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


# === MAF Analysis Functions ===
# These functions handle the calculation and analysis of Minor Allele Frequency (MAF),
# which is a key metric in genomic data analysis. MAF represents the frequency of the
# less common allele at a specific position in the genome.


def calculate_maf(samples):
    """Calculate Minor Allele Frequency (MAF) for a batch of samples.

    This is the core function for MAF calculation. It processes the input data to extract
    both raw allele frequencies and their corresponding MAF values. For SNP data, MAF is
    calculated as min(freq, 1-freq) where freq is the mean value across samples at each position.

    Args:
        samples: Input samples tensor of shape [batch_size, channels, seq_len]

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
    maf = torch.minimum(freq, 1 - freq)

    return maf, freq


def calculate_maf_stats(maf):
    """Calculate basic statistical measures for MAF values.

    This function computes summary statistics (mean, median, standard deviation)
    for a set of MAF values, which is useful for comparing distributions between
    real and generated data.

    Args:
        maf: Array/Tensor of MAF values

    Returns:
        dict: Dictionary containing:
            - mean: Mean MAF value
            - median: Median MAF value
            - std: Standard deviation of MAF values
            - min_maf: Minimum MAF value
            - max_maf: Maximum MAF value
            - num_half_maf: Number of MAF values equal to 0.5
            - min_freq: Minimum frequency value (same as min_maf)
            - max_freq: Maximum frequency value (same as max_maf)
            - num_half_freq: Number of frequency values equal to 0.5 (same as num_half_maf)
    """
    # Convert to tensor if numpy
    if not torch.is_tensor(maf):
        maf = torch.from_numpy(maf)

    # Calculate min/max values
    min_val = float(torch.min(maf).item())
    max_val = float(torch.max(maf).item())

    # Count 0.5 values with tolerance
    num_half = int(torch.sum((maf - 0.5).abs() < 0.001).item())

    return {
        "mean": float(torch.mean(maf).item()),
        "median": float(torch.median(maf).item()),
        "std": float(torch.std(maf).item()),
        "min_maf": min_val,
        "max_maf": max_val,
        "num_half_maf": num_half,
        "min_freq": min_val,  # For backward compatibility
        "max_freq": max_val,  # For backward compatibility
        "num_half_freq": num_half,  # For backward compatibility
    }


def analyze_maf_distribution(samples, save_path, bin_width=0.001):
    """Analyze and visualize the Minor Allele Frequency (MAF) distribution for a dataset.

    This function calculates MAF values, prints summary statistics, and creates a
    histogram visualization of the MAF distribution. It also analyzes the expected
    peak spacing based on the number of samples and presence of heterozygous (0.5) values.

    The visualization shows the theoretical peaks that should appear in the MAF distribution
    based on population genetics principles, which is useful for assessing the quality of
    both real and generated genomic data.

    Args:
        samples: Tensor of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]
        save_path: Path to save the MAF histogram plot
        bin_width: Width of histogram bins (default: 0.001)

    Returns:
        tuple: (maf, freq) where:
            - maf: numpy.ndarray of Minor Allele Frequency values
            - freq: numpy.ndarray of raw allele frequencies
    """
    # Calculate MAF and raw frequencies using the dedicated function
    maf, freq = calculate_maf(samples)

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

    return maf, freq


def compare_maf_distributions(real_maf, gen_maf, output_dir, bin_width=0.001):
    """Plot and compare MAF distributions between real and generated genomic data.

    This function creates two key visualizations:
    1. A histogram comparing the MAF distributions of real and generated data
    2. A scatter plot showing the correlation between real and generated MAF values

    These visualizations help assess how well the generated data captures the
    allele frequency patterns present in the real data, which is a critical metric
    for evaluating synthetic genomic data quality.

    Args:
        real_maf: numpy array of MAF values from real data
        gen_maf: numpy array of MAF values from generated data
        output_dir: Directory to save the plots
        bin_width: Width of histogram bins (default: 0.001 for fine-grained analysis)

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
        axes[i].imshow(samples[i, 0, :].reshape(1, -1), aspect="auto", cmap="viridis")
        if timesteps is not None:
            axes[i].set_title(f"t = {timesteps[i]}")
        else:
            axes[i].set_title(f"Step {i}")
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_samples(
    model,
    num_samples=1,
    start_timestep=None,
    output_dir=None,
    step_size=None,
    discretize=False,
    save_prefix="",
    collect_interval=100,
    verbose=True,
):
    """Generate samples from the diffusion model with flexible starting timestep.

    This unified function handles both full-noise (t=1000) and mid-noise generation,
    as well as visualization of the reverse diffusion process. It leverages the
    DiffusionModel's built-in generate_samples method for basic sample generation.

    Args:
        model: DiffusionModel instance
        num_samples: Number of samples to generate (default: 1)
        start_timestep: Timestep to start reverse diffusion from (default: model.ddpm.tmax)
        output_dir: Directory to save outputs (default: None, no saving)
        step_size: Step size for reverse diffusion (default: 10 for mid-noise, 100 for full-noise)
        discretize: Whether to discretize final output to 0, 0.5, 1.0 values (default: False)
        save_prefix: Prefix for saved files (default: "", empty string)
        collect_interval: Interval for collecting samples for visualization (default: 100)
        verbose: Whether to print progress information (default: True)

    Returns:
        tuple: (samples, plot_path) where:
            - samples: Generated samples tensor
            - plot_path: Path to visualization plot (None if output_dir is None)

    Raises:
        ValueError: If num_samples < 1 or if output_dir is provided but not a directory
        ValueError: If start_timestep is provided but >= model.ddpm.tmax
    """
    # Input validation
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory, got {output_dir}")

    if start_timestep is not None and start_timestep >= model.ddpm.tmax:
        raise ValueError(
            f"start_timestep ({start_timestep}) must be less than tmax ({model.ddpm.tmax})"
        )

    # Set defaults based on whether we're doing mid-noise or full-noise generation
    is_mid_noise = start_timestep is not None and start_timestep < model.ddpm.tmax

    if start_timestep is None:
        # For full-noise generation, use DiffusionModel's generate_samples
        samples = model.generate_samples(num_samples=num_samples, discretize=discretize)
        if verbose:
            print(f"Generated {num_samples} samples from full noise")
            print(
                f"Sample stats - mean: {samples.mean():.4f}, std: {samples.std():.4f}"
            )

        # Save samples if output directory is provided
        if output_dir:
            sample_path = output_dir / f"{save_prefix}generated_samples.pt"
            torch.save(samples, sample_path)
            if verbose:
                print(f"Samples saved to: {sample_path}")

        # No visualization for basic sample generation
        return samples, None

    # For mid-noise generation, we need to do the process step by step
    if step_size is None:
        step_size = 10 if is_mid_noise else 100

    if verbose:
        print(
            f"\nGenerating {num_samples} samples from mid-noise (t={start_timestep})..."
        )

    with torch.no_grad():
        # Create initial noisy samples
        batch_shape = (num_samples,) + model._data_shape
        x0 = torch.rand(batch_shape, device=model.device)
        eps = torch.randn_like(x0)
        t = torch.full(
            (num_samples,), start_timestep, device=model.device, dtype=torch.long
        )
        x = model.ddpm.sample(x0, t, eps)

        if verbose:
            print(f"Mid-noise stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

        # Save mid-noise samples if requested
        if output_dir:
            torch.save(x, output_dir / f"{save_prefix}mid_noise_samples.pt")

        # Store samples for visualization
        reverse_samples = [x.clone()]
        timesteps = [start_timestep]

        if verbose:
            print(f"Starting reverse diffusion from t={start_timestep} to t=0")

        # Reverse diffusion process
        for t in reversed(range(0, start_timestep + 1, step_size)):
            if t == 0:
                continue

            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            x = model._reverse_process_step(x, t_tensor)
            x = torch.clamp(x, -5.0, 5.0)

            if t % collect_interval == 0:
                reverse_samples.append(x.clone())
                timesteps.append(t)

                if verbose:
                    print(f"Step {t} stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

        # Final step
        t_tensor = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
        x = model._reverse_process_step(x, t_tensor)
        x = torch.clamp(x, 0, 1)
        reverse_samples.append(x.clone())
        timesteps.append(0)

        # Discretize if requested
        if discretize:
            x = torch.round(x * 2) / 2  # Round to nearest 0, 0.5, or 1.0

        # Save final samples
        if output_dir:
            sample_path = output_dir / f"{save_prefix}mid_noise_generated_samples.pt"
            torch.save(x, sample_path)
            if verbose:
                print(f"Samples saved to: {sample_path}")

        # Create visualization
        plot_path = None
        if output_dir:
            reverse_samples_tensor = torch.stack(reverse_samples)
            filename = f"{save_prefix}{'mid_noise_' if is_mid_noise else ''}reverse_diffusion.png"
            plot_path = output_dir / filename
            visualize_diffusion(
                reverse_samples_tensor.cpu(),
                plot_path,
                f"Reverse Diffusion Process (t={start_timestep} to t=0)",
                timesteps=timesteps,
            )
            if verbose:
                print(f"Visualization saved to: {plot_path}")

        return x, plot_path


def visualize_reverse_diffusion(model, output_dir, step_size=100):
    """Visualize the reverse diffusion process from full noise.

    Args:
        model: DiffusionModel instance
        output_dir: Directory to save visualization
        step_size: Step size for visualization (higher = fewer steps shown)

    Returns:
        Path to saved visualization
    """
    # Use the unified generate_samples function with full-noise settings
    _, plot_path = generate_samples(
        model=model,
        num_samples=1,  # Just one sample for visualization
        start_timestep=None,  # Use default (full noise)
        output_dir=output_dir,
        step_size=step_size,
        discretize=False,
        save_prefix="viz_full_noise_",  # Add prefix to avoid overwriting other files
        collect_interval=step_size,  # Match the step_size for visualization
        verbose=False,  # Reduce output verbosity for visualization
    )
    return plot_path


# === Mid-Noise Diffusion Functions ===
# These functions handle the generation and visualization of samples starting from
# mid-noise levels (t < tmax). This allows testing model behavior with less noisy
# starting points and can be useful for analyzing the denoising process.


def generate_mid_noise_samples(
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
    if mid_timestep >= model.ddpm.tmax:
        raise ValueError(
            f"mid_timestep ({mid_timestep}) must be less than tmax ({model.ddpm.tmax})"
        )

    # Create noise for the specified number of samples
    dummy_batch = torch.zeros((num_samples,) + model._data_shape, device=model.device)

    # Start from mid-noise level
    with torch.no_grad():
        # Create noise tensor for the specified timestep
        # Set random seed for reproducible noise initialization
        if seed is not None:
            torch.manual_seed(seed)

        # Initialize noise tensor
        x = torch.randn_like(dummy_batch)

        # Reset random seed to avoid affecting other random operations
        if seed is not None:
            torch.manual_seed(torch.seed())
        t = torch.full(
            (num_samples,), mid_timestep, device=model.device, dtype=torch.long
        )

        # Sample noise at mid_timestep
        x = model.ddpm.sample(dummy_batch, t, x)

        # Print initial statistics
        # print(f"Initial noise stats at t={mid_timestep} - mean: {x.mean():.4f}, std: {x.std():.4f}")

        # Custom reverse diffusion process starting from mid_timestep
        print(
            f"Starting reverse diffusion from t={mid_timestep} to t={model.ddpm.tmin} with step {denoise_step}"
        )

        # Perform reverse diffusion from mid_timestep to 0
        for t in reversed(range(model.ddpm.tmin, mid_timestep + 1, denoise_step)):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            x = model._reverse_process_step(x, t_tensor)

            # Print statistics every 100 steps
            if t % 100 == 0 or t == model.ddpm.tmin:
                # print(f"Step {t} stats - mean: {x.mean():.4f}, std: {x.std():.4f}")
                pass

        # Clamp to valid range [0, 1]
        x = torch.clamp(x, 0, 1)

        # Discretize if requested
        if discretize:
            x = torch.round(x * 2) / 2

        # print(f"Final sample stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

    return x


def visualize_mid_noise_diffusion(
    samples, output_path, num_examples=4, title="Mid-Noise Generated Samples"
):
    """Visualize samples generated from mid-noise level.

    This function creates a grid visualization of samples generated from a mid-noise level,
    helping to understand the quality and characteristics of mid-noise generation.

    Args:
        samples: Tensor of generated samples from mid-noise [batch_size, channels, seq_len]
        output_path: Path to save the visualization
        num_examples: Number of examples to show in visualization (default: 4)
        title: Title for the visualization plot (default: "Mid-Noise Generated Samples")

    Returns:
        Path: Path to the saved visualization file
    """
    output_path = Path(output_path)

    # Ensure we have enough samples
    if samples.size(0) < num_examples:
        raise ValueError(
            f"Not enough samples ({samples.size(0)}) for visualization (need {num_examples})"
        )

    # Create figure with subplots in a grid
    fig, axes = plt.subplots(2, num_examples // 2, figsize=(15, 8), squeeze=False)
    fig.suptitle(title, fontsize=14)

    # Plot each example
    for idx in range(num_examples):
        row, col = idx // (num_examples // 2), idx % (num_examples // 2)
        ax = axes[row, col]

        # Get sample and convert to numpy
        sample = samples[idx].squeeze().cpu().numpy()

        # Create heatmap
        sns.heatmap(
            sample.reshape(1, -1),
            ax=ax,
            cmap="viridis",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"Sample {idx+1}")

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
