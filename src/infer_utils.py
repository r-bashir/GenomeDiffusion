#!/usr/bin/env python
# coding: utf-8

"""
Clean utility functions for inference and evaluation of diffusion models.

This module contains essential functions for:
1. Sample generation (wrapper functions)
2. Basic sample analysis and visualization
3. Reverse diffusion visualization

Removed redundant/outdated functions and kept only the core utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from .utils import set_seed

# Default genotype values for SNP data
DEFAULT_GENOTYPE_VALUES = [0.0, 0.25, 0.5]


def _discretize_to_genotype_values(x, genotype_values=None):
    """
    Discretize tensor values to the closest valid genotype values.

    Args:
        x (torch.Tensor): Input tensor to discretize
        genotype_values (list): Valid genotype values to discretize to

    Returns:
        torch.Tensor: Discretized tensor with values mapped to closest genotype values
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

    # Convert to tensor for computation
    genotype_tensor = torch.tensor(genotype_values, device=x.device, dtype=x.dtype)

    # Find closest genotype value for each element
    # Expand dimensions for broadcasting: x[..., None] and genotype_tensor[None, ...]
    distances = torch.abs(x.unsqueeze(-1) - genotype_tensor)
    closest_indices = torch.argmin(distances, dim=-1)

    # Map to closest genotype values
    discretized = genotype_tensor[closest_indices]

    return discretized


# === Core Wrapper Functions ===
def generate_samples(
    model, num_samples=1, start_timestep=None, denoise_step=1, discretize=False, seed=42
):
    """
    Generate samples using reverse diffusion from any starting timestep.

    Args:
        model: The diffusion model to use for generation
        num_samples (int): Number of samples to generate
        start_timestep (int or None): Timestep to start from (None for full noise)
        denoise_step (int): Number of timesteps to skip in reverse diffusion
        discretize (bool): Whether to discretize the final output
        seed (int): Random seed for reproducibility

    Returns:
        torch.Tensor: Generated samples [B, C, L]
    """
    model.eval()
    set_seed(seed)

    # Default to full noise if not specified
    if start_timestep is None:
        start_timestep = model.forward_diffusion.tmax

    # Validate start_timestep
    tmax = model.forward_diffusion.tmax
    tmin = model.forward_diffusion.tmin
    if start_timestep > tmax:
        raise ValueError(
            f"start_timestep ({start_timestep}) cannot be greater than tmax ({tmax})"
        )
    if start_timestep < tmin:
        raise ValueError(
            f"start_timestep ({start_timestep}) cannot be less than tmin ({tmin})"
        )

    # Get data shape from model
    batch_shape = (num_samples,) + model._data_shape  # [B, C, L]
    device = model.device

    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(batch_shape, device=device)

        # Run reverse diffusion process
        for t in reversed(range(tmin, start_timestep + 1, denoise_step)):
            x = model.reverse_diffusion.reverse_diffusion_step(x, t)

        # Post-processing for SNP data
        if discretize:
            x = _discretize_to_genotype_values(x)

    return x


def denoise_samples(
    model,
    batch,
    start_timestep=None,
    denoise_step=1,
    discretize=False,
    seed=42,
):
    """
    Denoise an input batch using reverse diffusion starting from an arbitrary timestep.

    Args:
        model: The diffusion model to use
        batch (Tensor): Input batch to denoise [B, C, L]
        start_timestep (int or None): Timestep to start denoising from
        denoise_step (int): Number of timesteps to skip in reverse diffusion
        discretize (bool): Whether to discretize the final output
        seed (int): Random seed for reproducibility

    Returns:
        torch.Tensor: Denoised samples [B, C, L]
    """
    model.eval()
    set_seed(seed)

    # Default to full noise if not specified
    if start_timestep is None:
        start_timestep = model.forward_diffusion.tmax

    # Validate start_timestep
    tmax = model.forward_diffusion.tmax
    tmin = model.forward_diffusion.tmin
    if start_timestep > tmax:
        raise ValueError(
            f"start_timestep ({start_timestep}) cannot be greater than tmax ({tmax})"
        )
    if start_timestep < tmin:
        raise ValueError(
            f"start_timestep ({start_timestep}) cannot be less than tmin ({tmin})"
        )

    device = model.device
    x = batch.to(device)

    with torch.no_grad():
        # Run reverse diffusion process
        for t in reversed(range(tmin, start_timestep + 1, denoise_step)):
            x = model.reverse_diffusion.reverse_diffusion_step(x, t)

        # Post-processing for SNP data
        if discretize:
            x = _discretize_to_genotype_values(x)

    return x


# === Sample Analysis ===
def sample_statistics(samples, label, unique_values=False, genotype_counts=True):
    """
    Print comprehensive statistics for a sample tensor.

    Args:
        samples (Tensor): Sample tensor to analyze
        label (str): Label for the output
        unique_values (bool): Whether to show unique values
        show_genotype_counts (bool): Whether to show genotype frequency counts
    """
    print(f"\n{label} Statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {torch.mean(samples):.4f}")
    print(f"  Std: {torch.std(samples):.4f}")
    print(f"  Min: {torch.min(samples):.4f}")
    print(f"  Max: {torch.max(samples):.4f}")

    if unique_values:
        unique_vals = torch.unique(samples).tolist()
        print(f"  Unique values: {unique_vals}")

    if genotype_counts:
        # Show genotype distribution for genomic data
        flat_samples = samples.cpu().numpy().flatten()
        counts = count_genotype_values(flat_samples)
        total = len(flat_samples)

        print(f"  Genotype Distribution:")
        for i, (val, count) in enumerate(zip(DEFAULT_GENOTYPE_VALUES, counts)):
            percentage = (count / total) * 100
            print(f"    {val:.2f}: {count:,} ({percentage:.1f}%)")


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


def count_genotype_values(arr, genotype_values=None):
    """
    Count occurrences of each genotype value in array.

    Args:
        arr: Array-like object containing genotype values
        genotype_values: List of expected genotype values

    Returns:
        tuple: Counts for each genotype value
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

    arr = np.asarray(arr)
    counts = []

    for value in genotype_values:
        count = np.sum(np.isclose(arr, value, atol=1e-3))
        counts.append(int(count))

    return tuple(counts)


def sample_distribution(
    real_samples, generated_samples, save_path, genotype_values=None
):
    """
    Compare and plot the genotype distributions for real and generated samples.
    Prints stats and saves a bar plot to save_path.

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        save_path (str): Path to save the plot
        genotype_values (list): Expected genotype values, defaults to DEFAULT_GENOTYPE_VALUES

    Returns:
        None
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

    # Validate and Flatten samples
    real_flat, gen_flat = validate_samples(
        real_samples, generated_samples, flatten=True
    )

    # Count genotype values
    real_counts = count_genotype_values(real_flat, genotype_values)
    gen_counts = count_genotype_values(gen_flat, genotype_values)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: Histogram (consistent with visualize_quality_metrics histogram approach)
    # Use histogram for continuous distribution view (same as visualize_quality_metrics)
    axes[0].hist(
        real_flat, bins=50, alpha=0.7, label="Real", density=True, color="blue"
    )
    axes[0].hist(
        gen_flat,
        bins=50,
        alpha=0.7,
        label="Generated",
        density=True,
        color="red",
    )

    axes[0].set_title("Genotype Value Distribution")
    axes[0].set_xlabel("Genotype Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar plot (consistent styling with visualize_quality_metrics)
    labels = [f"{value:.2f}" for value in genotype_values]

    x = np.arange(len(labels))
    width = 0.35

    axes[1].bar(
        x - width / 2,
        real_counts,
        width,
        label="Real",
        alpha=0.8,
        color="blue",
        edgecolor="darkblue",
    )
    axes[1].bar(
        x + width / 2,
        gen_counts,
        width,
        label="Generated",
        alpha=0.8,
        color="red",
        edgecolor="darkred",
    )

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
    axes[1].set_title("Genotype Value Counts")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def sample_visualization(
    real_samples,
    generated_samples,
    save_path,
    genotype_values=None,
    max_seq_len=10000,
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
        genotype_values (list): Expected genotype values, defaults to DEFAULT_GENOTYPE_VALUES

    Returns:
        None
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

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


# === Quick Quality Metrics ===
def compute_quality_metrics(real_samples, generated_samples, max_value=0.5):
    """
    Compute essential quality metrics for immediate feedback during inference.

    Args:
        real_samples: Real data [B, C, L] or [B, L]
        generated_samples: Generated data [B, C, L] or [B, L]
        max_value: Maximum value for MAF calculations

    Returns:
        float: Quick quality score (0-1, higher is better)
    """
    # === CENTRALIZED DATA PREPARATION ===
    # Convert to numpy
    real = (
        real_samples.cpu().numpy()
        if torch.is_tensor(real_samples)
        else np.array(real_samples)
    )
    gen = (
        generated_samples.cpu().numpy()
        if torch.is_tensor(generated_samples)
        else np.array(generated_samples)
    )

    # Ensure consistent shape: [batch_size, sequence_length]
    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    # Prepare different data views for various analyses
    real_flat = real.flatten()  # For distribution analysis
    gen_flat = gen.flatten()  # For distribution analysis

    # Calculate allele frequencies per SNP position (axis=0 means across samples)
    real_af = np.mean(real, axis=0)  # Shape: [sequence_length]
    gen_af = np.mean(gen, axis=0)  # Shape: [sequence_length]

    # Calculate MAF (Minor Allele Frequency)
    real_maf = np.minimum(real_af, max_value - real_af)
    gen_maf = np.minimum(gen_af, max_value - gen_af)

    # === COMPUTE ALL METRICS ONCE ===
    # Correlations
    af_corr = np.corrcoef(real_af, gen_af)[0, 1]
    maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]

    # Statistical tests
    ks_stat, ks_pvalue = stats.ks_2samp(real_flat, gen_flat)

    # Basic statistics comparison
    mean_diff = abs(np.mean(real_flat) - np.mean(gen_flat))
    std_diff = abs(np.std(real_flat) - np.std(gen_flat))

    # Range coverage (using flattened arrays for consistency)
    real_min, real_max = np.min(real_flat), np.max(real_flat)
    gen_min, gen_max = np.min(gen_flat), np.max(gen_flat)
    range_coverage = (min(gen_max, real_max) - max(gen_min, real_min)) / (
        real_max - real_min
    )
    range_coverage = max(0.0, range_coverage)

    # === PRINT RESULTS ===
    print("=" * 40)
    print(f"🧬 AF Correlation: {af_corr:.3f} (should be 1.0)")
    print(f"🔬 MAF Correlation: {maf_corr:.3f} (should be 1.0)")
    print(f"📈 KS Test p-value: {ks_pvalue:.6f} (should be >0.05)")
    print(f"📉 KS Statistic: {ks_stat:.6f} (should be ~0.0)")
    print(f"📊 Mean Difference: {mean_diff:.6f} (should be ~0.0)")
    print(f"📊 Std Difference: {std_diff:.6f} (should be ~0.0)")
    print(f"📏 Range Coverage: {range_coverage:.3f} (should be 1.0)")

    # Compute overall quick score
    scores = []

    # AF correlation (higher is better)
    if not np.isnan(af_corr):
        scores.append(max(0, af_corr))

    # MAF correlation (higher is better)
    if not np.isnan(maf_corr):
        scores.append(max(0, maf_corr))

    # KS test (higher p-value is better, but cap at reasonable threshold)
    ks_score = min(1.0, ks_pvalue * 10)  # Scale p-value
    scores.append(ks_score)

    # KS statistic (lower is better)
    ks_stat_score = 1.0 - min(1.0, ks_stat)
    scores.append(ks_stat_score)

    # Range coverage (higher is better)
    scores.append(range_coverage)

    # Mean and std differences (lower is better, normalize)
    mean_score = 1.0 - min(1.0, mean_diff * 10)  # Scale difference
    std_score = 1.0 - min(1.0, std_diff * 10)  # Scale difference
    scores.append(mean_score)
    scores.append(std_score)

    # Overall score
    overall_score = np.mean(scores) if scores else 0.0
    print(f"🎯 Quality Score: {overall_score:.3f}/1.000")

    return float(overall_score)


def visualize_quality_metrics(
    real_samples, generated_samples, output_path, max_value=0.5
):
    """
    Create a visual summary of key metrics for immediate feedback.

    Args:
        real_samples: Real data [B, C, L] or [B, L]
        generated_samples: Generated data [B, C, L] or [B, L]
        output_path: Path to save the plot
        max_value: Maximum value for MAF calculations
    """
    # === CENTRALIZED DATA PREPARATION ===
    # Convert to numpy
    real = (
        real_samples.cpu().numpy()
        if torch.is_tensor(real_samples)
        else np.array(real_samples)
    )
    gen = (
        generated_samples.cpu().numpy()
        if torch.is_tensor(generated_samples)
        else np.array(generated_samples)
    )

    # Ensure consistent shape: [batch_size, sequence_length]
    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    # Prepare different data views for various analyses
    real_flat = real.flatten()  # For distribution analysis
    gen_flat = gen.flatten()  # For distribution analysis

    # Calculate allele frequencies per SNP position (axis=0 means across samples)
    real_af = np.mean(real, axis=0)  # Shape: [sequence_length]
    gen_af = np.mean(gen, axis=0)  # Shape: [sequence_length]

    # Calculate MAF (Minor Allele Frequency)
    real_maf = np.minimum(real_af, max_value - real_af)
    gen_maf = np.minimum(gen_af, max_value - gen_af)

    # === COMPUTE ALL METRICS ONCE ===
    # Correlations
    af_corr = np.corrcoef(real_af, gen_af)[0, 1]
    maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]

    # Statistical tests
    ks_stat, ks_pvalue = stats.ks_2samp(real_flat, gen_flat)
    mean_diff = abs(np.mean(real_flat) - np.mean(gen_flat))
    std_diff = abs(np.std(real_flat) - np.std(gen_flat))

    # Range coverage
    real_min, real_max = np.min(real_flat), np.max(real_flat)
    gen_min, gen_max = np.min(gen_flat), np.max(gen_flat)
    range_coverage = (min(gen_max, real_max) - max(gen_min, real_min)) / (
        real_max - real_min
    )
    range_coverage = max(0.0, range_coverage)

    # Basic statistics for comparison
    real_stats = [
        np.mean(real_flat),
        np.std(real_flat),
        np.min(real_flat),
        np.max(real_flat),
    ]
    gen_stats = [
        np.mean(gen_flat),
        np.std(gen_flat),
        np.min(gen_flat),
        np.max(gen_flat),
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Quick Quality Assessment - Key Metrics", fontsize=16, fontweight="bold"
    )

    # === SUBPLOT 1: Genotype Value Distribution ===
    ax = axes[0, 0]

    # Get unique values to determine if data is discrete
    real_flat = real.flatten()
    gen_flat = gen.flatten()
    all_unique = np.unique(np.concatenate([real_flat, gen_flat]))

    # Use bar chart if we have few unique values (discrete data)
    if len(all_unique) <= 10:
        # Count occurrences of each unique value
        real_counts = [
            np.sum(np.isclose(real_flat, val, atol=1e-6)) for val in all_unique
        ]
        gen_counts = [
            np.sum(np.isclose(gen_flat, val, atol=1e-6)) for val in all_unique
        ]

        # Create bar chart
        x_pos = np.arange(len(all_unique))
        width = 0.35

        ax.bar(
            x_pos - width / 2,
            real_counts,
            width,
            label="Real",
            alpha=0.8,
            color="blue",
            edgecolor="darkblue",
        )
        ax.bar(
            x_pos + width / 2,
            gen_counts,
            width,
            label="Generated",
            alpha=0.8,
            color="red",
            edgecolor="darkred",
        )

        ax.set_xlabel("Genotype Value")
        ax.set_ylabel("Count")
        ax.set_title("Genotype Value Distribution")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{val:.2f}" for val in all_unique])
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        # Use regular binning for continuous data
        ax.hist(
            real.flatten(), bins=50, alpha=0.7, label="Real", density=True, color="blue"
        )
        ax.hist(
            gen.flatten(),
            bins=50,
            alpha=0.7,
            label="Generated",
            density=True,
            color="red",
        )

    ax.set_xlabel("Genotype Value")
    ax.set_ylabel("Density")
    ax.set_title("Genotype Value Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 2: Basic Statistics Comparison ===
    ax = axes[0, 1]

    stat_names = ["Mean", "Std", "Min", "Max"]
    x = np.arange(len(stat_names))
    width = 0.35

    ax.bar(x - width / 2, real_stats, width, label="Real", alpha=0.8, color="blue")
    ax.bar(x + width / 2, gen_stats, width, label="Generated", alpha=0.8, color="red")
    ax.set_xlabel("Statistics")
    ax.set_ylabel("Value")
    ax.set_title("Basic Statistics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(stat_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 3: Quality Metrics Summary ===
    ax = axes[0, 2]

    # Normalize metrics for visualization (0-1 scale)
    metrics = {
        "AF Corr": max(0, af_corr),
        "MAF Corr": max(0, maf_corr),
        "KS p-val": min(1.0, ks_pvalue * 10),  # Scale p-value
        "KS stat": 1.0 - min(1.0, ks_stat),  # Invert (lower is better)
        "Mean Sim": 1.0 - min(1.0, mean_diff * 10),  # Invert and scale
        "Std Sim": 1.0 - min(1.0, std_diff * 10),  # Invert and scale
        "Range Cov": range_coverage,  # Range coverage (higher is better)
    }

    bars = ax.bar(
        metrics.keys(),
        metrics.values(),
        color=["blue", "green", "orange", "red", "purple", "brown", "gray"],
    )
    ax.set_ylabel("Quality Score")
    ax.set_title("Quality Metrics Summary")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # === SUBPLOT 4: Allele Frequency Correlation ===
    ax = axes[1, 0]

    ax.scatter(real_af, gen_af, alpha=0.6, s=10, color="purple")
    ax.plot([0, max_value], [0, max_value], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Real Allele Frequency")
    ax.set_ylabel("Generated Allele Frequency")
    ax.set_title(f"Allele Frequency Correlation\n(r = {af_corr:.3f})")
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 5: MAF Correlation ===
    ax = axes[1, 1]

    ax.scatter(real_maf, gen_maf, alpha=0.6, s=10, color="green")
    ax.plot([0, max_value / 2], [0, max_value / 2], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Real MAF")
    ax.set_ylabel("Generated MAF")
    ax.set_title(f"MAF Correlation\n(r = {maf_corr:.3f})")
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 6: Allele Frequency Residuals Plot ===
    ax = axes[1, 2]

    # Calculate residuals (errors) for each SNP position
    af_residuals = gen_af - real_af
    positions = np.arange(len(af_residuals))

    # Create residuals scatter plot
    ax.scatter(positions, af_residuals, alpha=0.6, s=8, color="red", edgecolors="none")

    # Add zero line (perfect prediction)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=1)

    # Add mean residual line
    mean_residual = np.mean(af_residuals)
    ax.axhline(
        y=mean_residual,
        color="blue",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label=f"Mean: {mean_residual:.4f}",
    )

    # Formatting
    ax.set_xlabel("SNP Position")
    ax.set_ylabel("AF Residual (Generated - Real)")
    ax.set_title(
        f"Allele Frequency Residuals\n(RMSE: {np.sqrt(np.mean(af_residuals**2)):.4f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Add statistics text
    rmse = np.sqrt(np.mean(af_residuals**2))
    mae = np.mean(np.abs(af_residuals))
    ax.text(
        0.02,
        0.98,
        f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# === Diffusion Visualization ===
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
