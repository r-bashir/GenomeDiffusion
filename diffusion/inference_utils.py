#!/usr/bin/env python
# coding: utf-8

"""Utility functions for inference and evaluation of diffusion models.

This module contains reusable functions for:
1. MAF (Minor Allele Frequency) analysis
2. Visualization of samples and diffusion process
3. Genomic metrics calculation
4. PCA analysis
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.decomposition import PCA


def calculate_maf(samples):
    """Calculate Minor Allele Frequency (MAF) for a batch of samples.

    Args:
        samples: Tensor of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]

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

    # Convert to minor allele frequency
    maf = np.minimum(freq, 1 - freq)
    
    return maf


def calculate_maf_stats(maf):
    """Calculate MAF statistics.

    Args:
        maf: numpy array of MAF values

    Returns:
        dict: Dictionary of MAF statistics
    """
    # Explicitly convert numpy values to Python floats for JSON serialization
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


def plot_sample_grid(samples, save_path, title, timesteps=None):
    """Plot a grid of samples showing the diffusion process.

    Args:
        samples: Tensor of samples to plot
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


def plot_comparison(real_samples, generated_samples, save_path):
    """Plot comparison between real and generated samples.

    Args:
        real_samples: Tensor of real SNP data
        generated_samples: Tensor of generated SNP data
        save_path: Path to save the plot
    """
    # Convert to numpy for plotting
    real = real_samples.cpu().numpy()
    gen = generated_samples.cpu().numpy()

    # Print statistics for debugging
    print("\nData Statistics:")
    print(
        f"Real - shape: {real.shape}, range: [{real.min():.3f}, {real.max():.3f}], mean: {real.mean():.3f}, std: {real.std():.3f}"
    )
    print(
        f"Generated - shape: {gen.shape}, range: [{gen.min():.3f}, {gen.max():.3f}], mean: {gen.mean():.3f}, std: {gen.std():.3f}"
    )

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
        plt.savefig(save_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"Warning: Error in plot_comparison: {e}")
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


def plot_maf_distribution(samples, save_path, bin_width=0.001):
    """Alias for analyze_maf_distribution for backward compatibility.

    Args:
        samples: Tensor of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]
        save_path: Path to save the MAF histogram plot
        bin_width: Width of histogram bins (default: 0.001 for fine-grained analysis)

    Returns:
        numpy.ndarray: Array of MAF values
    """
    return analyze_maf_distribution(samples, save_path, bin_width)


def visualize_reverse_diffusion(model, output_dir, step_size=100):
    """Visualize the reverse diffusion process.

    Args:
        model: DiffusionModel instance
        output_dir: Directory to save visualization
        step_size: Step size for visualization (higher = fewer steps shown)

    Returns:
        Path to saved visualization
    """
    reverse_samples = []
    timesteps = []

    # Start with noise (t=T)
    x = torch.randn((1,) + model._data_shape, device=model.device)
    reverse_samples.append(x)
    timesteps.append(model.ddpm.tmax)

    # Reverse diffusion process
    for t in range(model.ddpm.tmax, 0, -step_size):
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

    return plot_path
