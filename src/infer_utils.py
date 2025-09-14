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


import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from .analysis_utils import calculate_ld
from .utils import set_seed

# Default genotype values for SNP data
DEFAULT_GENOTYPE_VALUES = [0.0, 0.25, 0.5]


# === Helper Functions for Metrics ===
def compute_wasserstein_distance(real_flat, gen_flat):
    """Compute Wasserstein distance between two flattened distributions."""
    try:
        # Validate inputs
        real_flat = np.asarray(real_flat).flatten()
        gen_flat = np.asarray(gen_flat).flatten()

        if len(real_flat) == 0 or len(gen_flat) == 0:
            return np.nan

        return float(stats.wasserstein_distance(real_flat, gen_flat))
    except Exception as e:
        print(f"Warning: Wasserstein distance computation failed: {e}")
        return np.nan


def compute_ld_decay_metrics(real, gen, max_distance=50, n_pairs=2000):
    """Compute LD decay metrics and return correlation of binned curves."""
    try:
        # Check if real data has sufficient variance for LD analysis
        real_variances = np.var(real, axis=0)
        low_variance_snps = np.sum(real_variances < 1e-6)
        total_snps = real.shape[1]

        if low_variance_snps > total_snps * 0.8:  # More than 80% constant SNPs
            print(
                f"Warning: Real data has {low_variance_snps}/{total_snps} constant SNPs. Skipping LD analysis."
            )
            print(
                "This suggests population-level data rather than individual genotypes."
            )
            return 0.0, [], [], [], []

        real_distances, real_r2 = calculate_ld(
            real, max_distance=max_distance, n_pairs=n_pairs
        )
        gen_distances, gen_r2 = calculate_ld(
            gen, max_distance=max_distance, n_pairs=n_pairs
        )

        if len(real_distances) == 0 or len(gen_distances) == 0:
            print(
                f"LD computation failed - real_distances={len(real_distances)}, gen_distances={len(gen_distances)}"
            )
            return 0.0, [], [], [], []

        # Binning
        max_dist = int(
            min(max_distance, max(np.max(real_distances), np.max(gen_distances)))
        )
        if max_dist < 2:
            max_dist = 2
        bins = np.arange(1, max_dist + 1)

        real_binned = [
            (
                np.mean(real_r2[real_distances == d])
                if np.any(real_distances == d)
                else np.nan
            )
            for d in bins
        ]
        gen_binned = [
            (
                np.mean(gen_r2[gen_distances == d])
                if np.any(gen_distances == d)
                else np.nan
            )
            for d in bins
        ]

        # Remove NaN values for correlation computation
        valid_mask = ~(np.isnan(real_binned) | np.isnan(gen_binned))
        if np.sum(valid_mask) < 2:
            correlation = 0.0
        else:
            real_binned_clean = np.array(real_binned)[valid_mask]
            gen_binned_clean = np.array(gen_binned)[valid_mask]

            # Compute correlation of binned curves
            if len(real_binned_clean) > 1 and len(gen_binned_clean) > 1:
                correlation = np.corrcoef(real_binned_clean, gen_binned_clean)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

        return (
            float(correlation),
            real_distances.tolist(),
            real_r2.tolist(),
            gen_distances.tolist(),
            gen_r2.tolist(),
        )
    except Exception as e:
        print(f"Exception in LD computation: {e}")
        return 0.0, [], [], [], []


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


def get_encoding_params(scaling: bool):
    """Return genotype values and max_value based on scaling flag.

    Args:
        scaling (bool): True if data was scaled (e.g., original [0, 0.5, 1.0] divided by 2)

    Returns:
        dict: {"genotype_values": [...], "max_value": float}
    """
    if scaling:
        return {"genotype_values": [0.0, 0.25, 0.5], "max_value": 0.5}
    else:
        return {"genotype_values": [0.0, 0.5, 1.0], "max_value": 1.0}


# === Core Wrapper Functions ===
def generate_samples(
    model,
    num_samples=1,
    start_timestep=None,
    denoise_step=1,
    discretize=False,
    seed=42,
    true_x0=None,
    mask=None,
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
        true_x0 (torch.Tensor, optional): Ground-truth clean sample [B, C, L] for imputation
        mask (torch.Tensor, optional): Tensor [B, C, L] with 1 = known SNP, 0 = unknown SNP

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
            x = model.reverse_diffusion.reverse_diffusion_step(
                x, t, true_x0=true_x0, mask=mask
            )

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
    true_x0=None,
    mask=None,
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
        true_x0 (torch.Tensor, optional): Ground-truth clean sample [B, C, L] for imputation
        mask (torch.Tensor, optional): Tensor [B, C, L] with 1 = known SNP, 0 = unknown SNP

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
            x = model.reverse_diffusion.reverse_diffusion_step(
                x, t, true_x0=true_x0, mask=mask
            )

        # Post-processing for SNP data
        if discretize:
            x = _discretize_to_genotype_values(x)

    return x


# === Sample Analysis ===
def sample_statistics(
    samples, label, unique_values=False, genotype_counts=True, genotype_values=None
):
    """
    Print comprehensive statistics for a sample tensor.

    Args:
        samples (Tensor): Sample tensor to analyze
        label (str): Label for the output
        unique_values (bool): Whether to show unique values
        show_genotype_counts (bool): Whether to show genotype frequency counts
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

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
        counts = count_genotype_values(flat_samples, genotype_values)
        total = len(flat_samples)

        print("  Genotype Distribution:")
        for i, (val, count) in enumerate(zip(genotype_values, counts)):
            percentage = (count / total) * 100
            print(f"    {val:.2f}: {count:,} ({percentage:.1f}%)")


def validate_samples(real_samples, generated_samples, flatten=False):
    """
    Validate shape, convert tensors to numpy, and (optionally) flatten
    both real and generated samples.
    Returns (real_out, gen_out)
    """
    try:
        # Handle None inputs
        if real_samples is None or generated_samples is None:
            raise ValueError("Input samples cannot be None")

        # Convert to tensors if needed
        if not isinstance(real_samples, torch.Tensor):
            real_samples = torch.tensor(real_samples)
        if not isinstance(generated_samples, torch.Tensor):
            generated_samples = torch.tensor(generated_samples)

        if real_samples.shape != generated_samples.shape:
            raise ValueError(
                f"Input tensors must have same shape. Got {real_samples.shape} and {generated_samples.shape}"
            )
        if len(real_samples.shape) != 3:
            raise ValueError(
                f"Samples must be 3D tensors with shape [B, C, L]. "
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


def count_genotype_values(arr, genotype_values=None, atol=None):
    """
    Count occurrences of each genotype value in array.

    Args:
        arr: Array-like object containing genotype values
        genotype_values: List of expected genotype values
        atol: Absolute tolerance for matching (if None, uses 1e-3)

    Returns:
        tuple: Counts for each genotype value
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES
    if atol is None:
        atol = 1e-3

    arr = np.asarray(arr).flatten()  # Ensure flattened
    if len(arr) == 0:
        return tuple([0] * len(genotype_values))

    counts = []
    for value in genotype_values:
        # Handle potential NaN/inf values in array
        valid_mask = np.isfinite(arr)
        valid_arr = arr[valid_mask]
        count = np.sum(np.isclose(valid_arr, value, atol=atol, rtol=0))
        counts.append(int(count))

    return tuple(counts)


def sample_distribution(
    real_samples, generated_samples, output_dir, genotype_values=None
):
    """
    Compare and plot the genotype distributions for real and generated samples.
    Prints stats and saves a bar plot to output_dir.

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        output_dir (str): Path to save the plot
        genotype_values (list): Expected genotype values, defaults to DEFAULT_GENOTYPE_VALUES

    Returns:
        None
    """
    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

    # Validate and get flattened samples
    real_flat, gen_flat = validate_samples(
        real_samples, generated_samples, flatten=True
    )

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # SUBPLOT 1: Histogram for continuous distribution view
    # Compute shared bins for fair comparison of densities
    data_for_bins = np.concatenate([real_flat, gen_flat])
    bins = np.histogram_bin_edges(data_for_bins, bins=50)
    hist_args = {
        "bins": bins,
        "alpha": 0.7,
        "density": True,
    }
    axes[0].hist(real_flat, **hist_args, label="Real", color="blue")
    axes[0].hist(gen_flat, **hist_args, label="Generated", color="red")

    # Bin size annotation
    bin_widths = np.diff(bins)
    n_bins = len(bins) - 1
    if np.allclose(bin_widths, bin_widths[0]):
        bin_info = f"bins: {n_bins}\nwidth: {bin_widths[0]:.4g}"
    else:
        # Non-uniform binning (depending on strategy)
        bin_info = f"bins: {n_bins}\nmedian width: {np.median(bin_widths):.4g}"

    axes[0].text(
        0.98,
        0.98,
        bin_info,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.8),
    )

    # Axis Params
    axes[0].set_title("Genotype Value Distribution", fontsize=10)
    axes[0].set_xlabel("Genotype Value", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SUBPLOT 2: Bar plot to show genotype counts.
    # Use bin width as tolerance for consistent counting
    bin_width = (
        bin_widths[0]
        if np.allclose(bin_widths, bin_widths[0])
        else np.median(bin_widths)
    )
    tolerance = bin_width / 2  # Half bin width as reasonable tolerance

    # Count genotype values
    real_counts = count_genotype_values(real_flat, genotype_values, atol=tolerance)
    gen_counts = count_genotype_values(gen_flat, genotype_values, atol=tolerance)

    # Axis Params
    labels = [f"{value:.2f}" for value in genotype_values]
    x = np.arange(len(labels))
    width = 0.35

    # Bar plot
    axes[1].bar(
        x - width / 2,
        real_counts,
        width,
        label="Real",
        alpha=0.8,
        color="blue",
    )
    axes[1].bar(
        x + width / 2,
        gen_counts,
        width,
        label="Generated",
        alpha=0.8,
        color="red",
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

    # Axis Params
    axes[1].set_xlabel("Genotype Value", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title(f"Genotype Value Counts (±{tolerance:.4f})", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Figure Params
    fig.tight_layout()
    fig.savefig(
        output_dir / "sample_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "sample_distribution.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def sample_visualization(
    real_samples,
    generated_samples,
    output_dir,
    genotype_values=None,
    max_seq_len=100,
):
    """Plot comparison between real and generated samples.

    Creates a 2x2 grid showing:
    1. Real data pattern (first 100 positions)
    2. Generated data pattern (first 100 positions)
    3. Value distributions for both datasets

    Args:
        real_samples (Tensor): Real SNP data [batch_size, channels, seq_len]
        generated_samples (Tensor): Generated SNP data [batch_size, channels, seq_len]
        output_dir (str): Path to save the plot
        max_seq_len (int, optional): Max sequence length to plot in all plots. Default is 100.
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

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Real and Generated Samples Comparison")

    # Limit sequence length for upper plots
    real_selected = real_flat[:max_seq_len]
    gen_selected = gen_flat[:max_seq_len]

    # Plot real sample sequence (limited length)
    axes[0, 0].plot(real_selected, color="tab:blue")
    axes[0, 0].set_title(f"Real Sample Sequence (first {max_seq_len})")
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Value")

    # Plot generated sample sequence (limited length)
    axes[0, 1].plot(gen_selected, color="tab:orange")
    axes[0, 1].set_title(f"Generated Sample Sequence (first {max_seq_len})")
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_ylabel("Value")

    # Custom colormap: 0 → blue, 0.5 → green, 1 → red
    # cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
    cmap = plt.cm.viridis

    # Set min/max values based on genotype values
    vmin = min(genotype_values)
    vmax = max(genotype_values)

    # Bottom left: Real data heatmap (first channel, first 100 positions)
    im_real = axes[1, 0].imshow(
        real_selected.reshape(1, -1)[:, :max_seq_len],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title(f"Real Data Pattern (first {max_seq_len}) positions")
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
        gen_selected.reshape(1, -1)[:, :max_seq_len],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title(f"Generated Data Pattern (first {max_seq_len}) positions")
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
    fig.savefig(
        output_dir / "sample_visualization.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "sample_visualization.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# === Centralized Quality Metrics ===
def compute_quality_metrics(
    real_samples,
    generated_samples,
    max_value: float,
    genotype_values: list[float] | None = None,
    print_results: bool = True,
):
    """
    Compute all quality metrics in one place to ensure consistency.

    Args:
        real_samples: Real data [B, C, L] or [B, L]
        generated_samples: Generated data [B, C, L] or [B, L]
        max_value: Maximum value for MAF calculations (required)
        genotype_values: Expected genotype values for discrete analysis
        print_results: Whether to print results to console

    Returns:
        dict: Comprehensive metrics dictionary containing all computed values
    """
    if max_value is None:
        raise ValueError("max_value is required and cannot be None")

    if genotype_values is None:
        genotype_values = DEFAULT_GENOTYPE_VALUES

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
    real_flat = real.flatten()
    gen_flat = gen.flatten()

    # Calculate allele frequencies per SNP position (axis=0 means across samples)
    real_af = np.mean(real, axis=0)
    gen_af = np.mean(gen, axis=0)

    # Calculate MAF (Minor Allele Frequency)
    real_maf = np.minimum(real_af, max_value - real_af)
    gen_maf = np.minimum(gen_af, max_value - gen_af)

    # === COMPUTE ALL METRICS ONCE ===
    # Correlations with variance guards
    # AF Correlation: Should be 1.0 (perfect correlation between real and generated allele frequencies)
    if np.std(real_af) > 0 and np.std(gen_af) > 0:
        af_corr = float(np.corrcoef(real_af, gen_af)[0, 1])
        if np.isnan(af_corr):
            af_corr = 0.0
    else:
        af_corr = 0.0

    # MAF Correlation: Should be 1.0 (perfect correlation between real and generated minor allele frequencies)
    if np.std(real_maf) > 0 and np.std(gen_maf) > 0:
        maf_corr = float(np.corrcoef(real_maf, gen_maf)[0, 1])
        if np.isnan(maf_corr):
            maf_corr = 0.0
    else:
        maf_corr = 0.0

    # Statistical tests
    # KS Test: p-value should be >0.05 (distributions are similar), statistic should be ~0.0
    ks_stat, ks_pvalue = stats.ks_2samp(real_flat, gen_flat)

    # Basic statistics
    real_stats = {
        "mean": float(np.mean(real_flat)),
        "std": float(np.std(real_flat)),
        "min": float(np.min(real_flat)),
        "max": float(np.max(real_flat)),
    }
    gen_stats = {
        "mean": float(np.mean(gen_flat)),
        "std": float(np.std(gen_flat)),
        "min": float(np.min(gen_flat)),
        "max": float(np.max(gen_flat)),
    }

    # Mean/Std Differences: Should be ~0.0 (similar distributions)
    mean_diff = abs(real_stats["mean"] - gen_stats["mean"])
    std_diff = abs(real_stats["std"] - gen_stats["std"])

    # Range coverage with safe division
    # Range Coverage: Should be 1.0 (generated data covers the same range as real data)
    real_min, real_max = real_stats["min"], real_stats["max"]
    gen_min, gen_max = gen_stats["min"], gen_stats["max"]
    denom = real_max - real_min
    if denom <= 0:
        range_coverage = 1.0 if (gen_min == real_min and gen_max == real_max) else 0.0
    else:
        range_coverage = (min(gen_max, real_max) - max(gen_min, real_min)) / denom
    range_coverage = max(0.0, range_coverage)

    # Wasserstein Distance: Should be ~0.0 (distributions are identical)
    wasserstein_dist = compute_wasserstein_distance(real_flat, gen_flat)

    # LD Decay Correlation: Should be 1.0 (perfect correlation between LD decay patterns)
    ld_corr, real_distances, real_r2, gen_distances, gen_r2 = compute_ld_decay_metrics(
        real, gen
    )

    # AF residuals
    # RMSE/MAE: Should be ~0.0 (minimal prediction errors)
    af_residuals = gen_af - real_af
    rmse_af = float(np.sqrt(np.mean(af_residuals**2)))
    mae_af = float(np.mean(np.abs(af_residuals)))

    # Compute overall quality score
    scores = []
    if not np.isnan(af_corr):
        scores.append(max(0, af_corr))
    if not np.isnan(maf_corr):
        scores.append(max(0, maf_corr))

    ks_score = min(1.0, ks_pvalue * 10)
    ks_stat_score = 1.0 - min(1.0, ks_stat)
    mean_score = 1.0 - min(1.0, mean_diff * 10)
    std_score = 1.0 - min(1.0, std_diff * 10)

    scores.extend([ks_score, ks_stat_score, range_coverage, mean_score, std_score])
    overall_score = float(np.mean(scores)) if scores else 0.0

    # === COMPREHENSIVE METRICS DICTIONARY ===
    metrics = {
        # Data arrays for visualization
        "real": real,
        "gen": gen,
        "real_flat": real_flat,
        "gen_flat": gen_flat,
        "real_af": real_af,
        "gen_af": gen_af,
        "real_maf": real_maf,
        "gen_maf": gen_maf,
        "af_residuals": af_residuals,
        # Core metrics
        "af_corr": af_corr,
        "maf_corr": maf_corr,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "range_coverage": range_coverage,
        "wasserstein_dist": wasserstein_dist,
        "ld_corr": ld_corr,
        "rmse_af": rmse_af,
        "mae_af": mae_af,
        # Statistics
        "real_stats": real_stats,
        "gen_stats": gen_stats,
        # LD data for plotting
        "real_distances": real_distances,
        "real_r2": real_r2,
        "gen_distances": gen_distances,
        "gen_r2": gen_r2,
        # Scores
        "overall_score": overall_score,
        # Parameters
        "max_value": max_value,
        "genotype_values": genotype_values,
    }

    # === PRINT RESULTS ===
    if print_results:
        print("=" * 40)
        print(f"🧬 AF Correlation: {af_corr:.3f} (should be 1.0)")
        print(f"🔬 MAF Correlation: {maf_corr:.3f} (should be 1.0)")
        print(f"📈 KS Test p-value: {ks_pvalue:.6f} (should be >0.05)")
        print(f"📉 KS Statistic: {ks_stat:.6f} (should be ~0.0)")
        print(f"📊 Mean Difference: {mean_diff:.6f} (should be ~0.0)")
        print(f"📊 Std Difference: {std_diff:.6f} (should be ~0.0)")
        print(f"📏 Range Coverage: {range_coverage:.3f} (should be 1.0)")
        print(f"🌊 Wasserstein Distance: {wasserstein_dist:.4f} (should be ~0.0)")
        print(f"🧬 LD Decay Correlation: {ld_corr:.3f} (should be 1.0)")
        print(f"🧬 Genotype Values: {genotype_values}")
        print(f"🎯 Quality Score: {overall_score:.3f}/1.000")

    return metrics


def visualize_quality_metrics(
    real_samples,
    generated_samples,
    output_dir,
    max_value: float,
    genotype_values: list[float] | None = None,
):
    """
    Create a visual summary using centralized metrics computation.

    Args:
        real_samples: Real data [B, C, L] or [B, L]
        generated_samples: Generated data [B, C, L] or [B, L]
        output_dir: Path to save the plot
        max_value: Maximum value for MAF calculations (required)
        genotype_values: Expected genotype values for discrete analysis
    """
    # Use centralized computation - no duplication!
    metrics = compute_quality_metrics(
        real_samples, generated_samples, max_value, genotype_values, print_results=False
    )

    # Extract all needed values from centralized metrics
    real_flat = metrics["real_flat"]
    gen_flat = metrics["gen_flat"]
    real_af = metrics["real_af"]
    gen_af = metrics["gen_af"]
    real_maf = metrics["real_maf"]
    gen_maf = metrics["gen_maf"]
    af_residuals = metrics["af_residuals"]

    af_corr = metrics["af_corr"]
    maf_corr = metrics["maf_corr"]
    ks_stat = metrics["ks_stat"]
    ks_pvalue = metrics["ks_pvalue"]
    mean_diff = metrics["mean_diff"]
    std_diff = metrics["std_diff"]
    range_coverage = metrics["range_coverage"]
    wasserstein_dist = metrics["wasserstein_dist"]
    ld_corr = metrics["ld_corr"]
    rmse_af = metrics["rmse_af"]
    mae_af = metrics["mae_af"]

    real_stats = [
        metrics["real_stats"]["mean"],
        metrics["real_stats"]["std"],
        metrics["real_stats"]["min"],
        metrics["real_stats"]["max"],
    ]
    gen_stats = [
        metrics["gen_stats"]["mean"],
        metrics["gen_stats"]["std"],
        metrics["gen_stats"]["min"],
        metrics["gen_stats"]["max"],
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle("Quality Assessment - Key Metrics", fontsize=16, fontweight="bold")

    # === SUBPLOT 1: Wasserstein Distance (Empirical CDFs) ===
    ax = axes[0, 0]
    r_sorted = np.sort(real_flat)
    g_sorted = np.sort(gen_flat)
    r_ecdf = np.arange(1, r_sorted.size + 1) / r_sorted.size
    g_ecdf = np.arange(1, g_sorted.size + 1) / g_sorted.size

    ax.step(r_sorted, r_ecdf, where="post", label="Real eCDF", color="blue", alpha=0.8)
    ax.step(
        g_sorted, g_ecdf, where="post", label="Generated eCDF", color="red", alpha=0.8
    )
    ax.set_xlabel("Genotype Value")
    ax.set_ylabel("eCDF")
    ax.set_title(
        f"Wasserstein Distance (should be ~0.0)\n($W_1 = {wasserstein_dist:.4f}$)",
        fontsize=10,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 2: LD Decay (mini) ===
    ax = axes[0, 1]

    # Use centralized LD computation
    real_distances = metrics["real_distances"]
    real_r2 = metrics["real_r2"]
    gen_distances = metrics["gen_distances"]
    gen_r2 = metrics["gen_r2"]

    # Ensure arrays for boolean masking and indexing
    real_distances = np.asarray(real_distances)
    real_r2 = np.asarray(real_r2)
    gen_distances = np.asarray(gen_distances)
    gen_r2 = np.asarray(gen_r2)

    if len(real_distances) > 0 and len(gen_distances) > 0:
        # Binning (recreate for plotting)
        max_dist = int(min(50, max(np.max(real_distances), np.max(gen_distances))))
        if max_dist < 2:
            max_dist = 2
        bins = np.arange(1, max_dist + 1)
        bin_centers = bins
        real_binned = [
            np.mean(real_r2[real_distances == d]) if np.any(real_distances == d) else 0
            for d in bins
        ]
        gen_binned = [
            np.mean(gen_r2[gen_distances == d]) if np.any(gen_distances == d) else 0
            for d in bins
        ]

        # Plot light scatter of raw points (subsample for speed)
        if len(real_distances) > 0:
            idx = np.random.choice(
                len(real_distances), size=min(1000, len(real_distances)), replace=False
            )
            ax.scatter(
                np.array(real_distances)[idx],
                np.array(real_r2)[idx],
                s=6,
                alpha=0.15,
                color="blue",
            )
        if len(gen_distances) > 0:
            idx = np.random.choice(
                len(gen_distances), size=min(1000, len(gen_distances)), replace=False
            )
            ax.scatter(
                np.array(gen_distances)[idx],
                np.array(gen_r2)[idx],
                s=6,
                alpha=0.15,
                color="red",
            )

        # Plot binned means
        ax.plot(
            bin_centers,
            real_binned,
            "b-o",
            linewidth=2,
            markersize=4,
            label="Real (binned)",
        )
        ax.plot(
            bin_centers,
            gen_binned,
            "r-s",
            linewidth=2,
            markersize=4,
            label="Generated (binned)",
        )

        ax.set_title(
            f"LD Decay Correlation (should be 1.0)\n($r^2 = {ld_corr:.3f}$)",
            fontsize=10,
        )
        ax.set_xlabel("Distance (SNPs)")
        ax.set_ylabel("$r^2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient data for LD", ha="center", va="center")
        ax.axis("off")

    # === SUBPLOT 3: Discretized CM or Q-Q plot ===
    subplot = "qq"
    if subplot == "dcm":
        ax = axes[0, 2]
        gvals = np.array(metrics["genotype_values"], dtype=float)  # [0.0, 0.25, 0.5]

        # Map to nearest genotype
        def nearest_idx(x, gvals):
            return np.abs(x[..., None] - gvals[None, ...]).argmin(axis=-1)

        real_idx = nearest_idx(real_flat, gvals)
        gen_idx = nearest_idx(gen_flat, gvals)

        # Build confusion matrix counts
        K = len(gvals)
        cm = np.zeros((K, K), dtype=int)
        for i in range(K):
            sel = real_idx == i
            if sel.any():
                gi = gen_idx[sel]
                for j in range(K):
                    cm[i, j] = int(np.sum(gi == j))

        # Normalize rows to percentages (handle empty rows)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_pct = np.where(row_sums > 0, 100.0 * cm / row_sums, 0.0)

        # Plot heatmap
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(K):
            for j in range(K):
                ax.text(
                    j, i, f"{cm_pct[i, j]:.1f}%", va="center", ha="center", fontsize=9
                )

        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels([f"{v:.2f}" for v in gvals])
        ax.set_yticklabels([f"{v:.2f}" for v in gvals])
        ax.set_xlabel("Generated (nearest genotype)")
        ax.set_ylabel("Real (nearest genotype)")
        ax.set_title("Genotype Confusion Matrix (% per real row)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if subplot == "qq":
        ax = axes[0, 2]
        q = np.linspace(0, 1, 200)
        rq = np.quantile(real_flat, q)
        gq = np.quantile(gen_flat, q)
        ax.scatter(rq, gq, s=10, alpha=0.6, color="tab:blue")
        lims = [min(rq.min(), gq.min()), max(rq.max(), gq.max())]
        ax.plot(lims, lims, "r--", linewidth=2, alpha=0.8)
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Generated quantiles")
        ax.set_title("Q–Q Plot (distributional alignment)")
        ax.grid(True, alpha=0.3)

    # === SUBPLOT 4: Basic Statistics Comparison ===
    ax = axes[0, 3]

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

    # === SUBPLOT 5: MAF Correlation ===
    ax = axes[1, 0]

    ax.scatter(real_maf, gen_maf, alpha=0.6, s=10, color="green")
    ax.plot([0, max_value / 2], [0, max_value / 2], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Real MAF")
    ax.set_ylabel("Generated MAF")
    ax.set_title(f"MAF Correlation (should be 1.0)\n($r = {maf_corr:.3f}$)")
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 6: AF Correlation ===
    ax = axes[1, 1]

    ax.scatter(real_af, gen_af, alpha=0.6, s=10, color="purple")
    ax.plot([0, max_value], [0, max_value], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Real AF")
    ax.set_ylabel("Generated AF")
    ax.set_title(f"AF Correlation (should be 1.0)\n($r = {af_corr:.3f}$)")
    ax.grid(True, alpha=0.3)

    # === SUBPLOT 7: AF Residuals ===
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
    ax.set_title(f"AF Residuals (RMSE should be ~0.0)\n($RMSE = {rmse_af:.4f}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Add statistics text using centralized values
    ax.text(
        0.02,
        0.98,
        f"RMSE: {rmse_af:.4f}\nMAE: {mae_af:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(fontsize=8)

    # === SUBPLOT 8: Quality Metrics Summary ===
    ax = axes[1, 3]

    # Normalize metrics for visualization (0-1 scale, higher is better)
    viz_metrics = {
        "AF Corr": max(0, af_corr),  # Should be 1.0
        "MAF Corr": max(0, maf_corr),  # Should be 1.0
        "KS p-val": min(1.0, ks_pvalue * 10),  # Should be >0.05, scaled for viz
        "KS stat": 1.0 - min(1.0, ks_stat),  # Should be ~0.0, inverted for viz
        "Mean Sim": 1.0 - min(1.0, mean_diff * 10),  # Should be ~0.0, inverted for viz
        "Std Sim": 1.0 - min(1.0, std_diff * 10),  # Should be ~0.0, inverted for viz
        "Range Cov": range_coverage,  # Should be 1.0
    }

    bars = ax.bar(
        viz_metrics.keys(),
        viz_metrics.values(),
        color=["blue", "green", "orange", "red", "purple", "brown", "gray"],
    )
    ax.set_ylabel("Quality Score (0-1, higher better)")
    ax.set_title("Quality Metrics Summary")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, viz_metrics.values()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(
        output_dir / "quality_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "quality_metrics.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# === Diffusion Visualization ===
def visualize_diffusion(samples, output_dir, title, timesteps=None):
    """Plot a grid of samples showing the diffusion process.

    Creates a grid visualization showing how samples evolve during the diffusion process.
    Each row represents a different timestep, allowing us to see how the noise level
    changes throughout the process.

    Args:
        samples: Tensor of samples to plot [num_steps, batch_size, channels, seq_len]
        output_dir: Path to save the plot
        title: Title for the plot
        timesteps: List of timesteps corresponding to each sample
    """
    n_samples = min(samples.shape[0], 20)  # Show at most 20 timesteps
    # seq_length = samples.shape[-1]

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
    fig.savefig(
        output_dir / "diffusion.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "diffusion.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
