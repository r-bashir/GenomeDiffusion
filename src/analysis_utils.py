#!/usr/bin/env python
# coding: utf-8

"""
Modular genomic sample analysis utilities organized as calculate → plot pairs.

Functions are organized in logical groups:
1. Linkage Disequilibrium: calculate_ld() → plot_ld_decay()
2. Principal Component Analysis: run_pca_analysis() (includes plotting)
3. Genetic Diversity: calculate_genetic_diversity()
4. Dimensionality Metrics: compute_dimensionality_metrics()
5. Reporting: print_evaluation_summary(), create_evaluation_report()

Used by analyze_samples.py for transparent, modular genomic sample analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# Standard plotting configuration for consistency
PLOT_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 300,
    "font_size_title": 14,
    "font_size_label": 12,
    "colors": {"real": "blue", "generated": "red"},
    "grid_alpha": 0.3,
}

# Try to import wasserstein_distance, use alternative if not available
try:
    from scipy.spatial.distance import wasserstein_distance

    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False
    print(
        "Warning: wasserstein_distance not available in this scipy version. Using alternative implementation."
    )

    def wasserstein_distance(u_values, v_values):
        """
        Simple alternative implementation of 1D Wasserstein distance.
        This is a basic approximation using sorted values.
        """
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)

        # Make arrays same length by interpolation
        n = max(len(u_sorted), len(v_sorted))
        u_interp = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted
        )
        v_interp = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted
        )

        return np.mean(np.abs(u_interp - v_interp))


# =============================================================================
# SAMPLE DIVERSITY ANALYSIS (Critical for detecting mode collapse)
# =============================================================================


def calculate_sample_diversity(samples):
    """
    Calculate diversity metrics between generated samples to detect mode collapse.

    Args:
        samples: Generated samples [batch_size, seq_len] or [batch_size, channels, seq_len]

    Returns:
        dict: Sample diversity metrics
    """
    # Convert to numpy
    data = samples.cpu().numpy() if torch.is_tensor(samples) else np.array(samples)

    if data.ndim == 3:
        data = data.squeeze(1)

    n_samples, seq_len = data.shape

    # Calculate pairwise sample correlations
    sample_correlations = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Check for zero variance before correlation calculation
            if np.std(data[i]) == 0 or np.std(data[j]) == 0:
                corr = 1.0 if np.array_equal(data[i], data[j]) else 0.0
            else:
                corr = np.corrcoef(data[i], data[j])[0, 1]
            if not np.isnan(corr):
                sample_correlations.append(corr)

    avg_sample_correlation = (
        np.mean(sample_correlations) if sample_correlations else 1.0
    )

    # Calculate position-wise standard deviation across samples
    position_std = np.std(data, axis=0)
    avg_position_std = np.mean(position_std)
    min_position_std = np.min(position_std)
    max_position_std = np.max(position_std)

    # Calculate sample-wise standard deviation (diversity within each sample)
    sample_std = np.std(data, axis=1)
    avg_sample_std = np.mean(sample_std)

    # Effective number of unique patterns (rough estimate)
    # Count samples that are significantly different from each other
    unique_patterns = 0
    threshold = 0.95  # Correlation threshold for "same" pattern

    for i in range(n_samples):
        is_unique = True
        for j in range(i):
            # Check for zero variance before correlation calculation
            if np.std(data[i]) == 0 or np.std(data[j]) == 0:
                corr = 1.0 if np.array_equal(data[i], data[j]) else 0.0
            else:
                corr = np.corrcoef(data[i], data[j])[0, 1]
            if not np.isnan(corr) and corr > threshold:
                is_unique = False
                break
        if is_unique:
            unique_patterns += 1

    effective_diversity = unique_patterns / n_samples if n_samples > 0 else 0

    return {
        "avg_sample_correlation": float(avg_sample_correlation),
        "avg_position_std": float(avg_position_std),
        "min_position_std": float(min_position_std),
        "max_position_std": float(max_position_std),
        "avg_sample_std": float(avg_sample_std),
        "effective_diversity": float(effective_diversity),
        "n_unique_patterns": int(unique_patterns),
        "total_samples": int(n_samples),
    }


def detect_mode_collapse(real_samples, generated_samples):
    """
    Detect mode collapse by comparing sample diversity between real and generated data.

    Args:
        real_samples: Real data
        generated_samples: Generated data

    Returns:
        dict: Mode collapse detection results with warnings
    """
    real_diversity = calculate_sample_diversity(real_samples)
    gen_diversity = calculate_sample_diversity(generated_samples)

    warnings = []

    # Check for mode collapse indicators
    if gen_diversity["avg_sample_correlation"] > 0.95:
        warnings.append(
            "🚨 CRITICAL: Generated samples are nearly identical (avg correlation > 0.95)"
        )

    if gen_diversity["avg_position_std"] < 0.01:
        warnings.append(
            "🚨 CRITICAL: No variation across samples (position std < 0.01)"
        )

    if gen_diversity["effective_diversity"] < 0.1:
        warnings.append("🚨 CRITICAL: Less than 10% of samples are unique patterns")

    # Compare to real data
    diversity_ratio = gen_diversity["effective_diversity"] / max(
        real_diversity["effective_diversity"], 0.01
    )
    if diversity_ratio < 0.5:
        warnings.append(
            f"⚠️  Generated diversity is {diversity_ratio:.1%} of real data diversity"
        )

    correlation_ratio = gen_diversity["avg_sample_correlation"] / max(
        real_diversity["avg_sample_correlation"], 0.01
    )
    if correlation_ratio > 2.0:
        warnings.append(
            f"⚠️  Generated samples are {correlation_ratio:.1f}x more correlated than real samples"
        )

    return {
        "real_diversity": real_diversity,
        "generated_diversity": gen_diversity,
        "diversity_ratio": float(diversity_ratio),
        "correlation_ratio": float(correlation_ratio),
        "warnings": warnings,
        "mode_collapse_detected": len([w for w in warnings if "CRITICAL" in w]) > 0,
    }


# =============================================================================
# DIMENSIONALITY ANALYSIS
# =============================================================================


def compute_dimensionality_metrics(real_samples, generated_samples):
    """
    Compute high-dimensional similarity metrics using PCA.

    Args:
        real_samples: Real data [B, C, L] or [B, L]
        generated_samples: Generated data [B, C, L] or [B, L]

    Returns:
        dict: Dimensionality metrics
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    metrics = {}

    try:
        # Combined PCA
        combined = np.vstack([real, gen])
        pca = PCA(n_components=min(10, combined.shape[1]))  # Up to 10 components
        pca_result = pca.fit_transform(combined)

        real_pca = pca_result[: real.shape[0]]
        gen_pca = pca_result[real.shape[0] :]

        # Centroid distance in PCA space
        real_centroid = np.mean(real_pca, axis=0)
        gen_centroid = np.mean(gen_pca, axis=0)
        metrics["pca_centroid_distance"] = float(
            np.linalg.norm(real_centroid - gen_centroid)
        )

        # Variance explained
        metrics["pca_variance_explained"] = pca.explained_variance_ratio_.tolist()
        metrics["pca_cumulative_variance"] = np.cumsum(
            pca.explained_variance_ratio_
        ).tolist()

        # Distribution overlap in PCA space (first 2 components)
        if pca.n_components_ >= 2:
            # Calculate overlap using 2D histograms
            real_pc12 = real_pca[:, :2]
            gen_pc12 = gen_pca[:, :2]

            # Create 2D histograms
            x_range = [
                min(real_pc12[:, 0].min(), gen_pc12[:, 0].min()),
                max(real_pc12[:, 0].max(), gen_pc12[:, 0].max()),
            ]
            y_range = [
                min(real_pc12[:, 1].min(), gen_pc12[:, 1].min()),
                max(real_pc12[:, 1].max(), gen_pc12[:, 1].max()),
            ]

            bins = 20
            real_hist, _, _ = np.histogram2d(
                real_pc12[:, 0], real_pc12[:, 1], bins=bins, range=[x_range, y_range]
            )
            gen_hist, _, _ = np.histogram2d(
                gen_pc12[:, 0], gen_pc12[:, 1], bins=bins, range=[x_range, y_range]
            )

            # Normalize
            real_hist = real_hist / np.sum(real_hist)
            gen_hist = gen_hist / np.sum(gen_hist)

            # Calculate overlap (intersection over union)
            intersection = np.sum(np.minimum(real_hist, gen_hist))
            union = np.sum(np.maximum(real_hist, gen_hist))
            metrics["pca_2d_overlap"] = (
                float(intersection / union) if union > 0 else 0.0
            )

    except Exception as e:
        print(f"Warning: PCA analysis failed: {e}")
        metrics["pca_centroid_distance"] = None
        metrics["pca_variance_explained"] = None
        metrics["pca_cumulative_variance"] = None
        metrics["pca_2d_overlap"] = None

    return metrics


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================


def create_evaluation_visualizations(
    real_samples, generated_samples, metrics, output_dir
):
    """
    Create comprehensive visualizations of the evaluation results.

    Args:
        real_samples: Real data
        generated_samples: Generated data
        metrics: Computed metrics dictionary
        output_dir: Directory to save plots
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    # 1. Distribution comparison
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(real.flatten(), bins=50, alpha=0.7, label="Real", density=True)
    plt.hist(gen.flatten(), bins=50, alpha=0.7, label="Generated", density=True)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Value Distribution Comparison")
    plt.legend()

    plt.subplot(1, 3, 2)
    real_af = np.mean(real, axis=0)
    gen_af = np.mean(gen, axis=0)
    plt.scatter(real_af, gen_af, alpha=0.6, s=10)
    plt.plot([0, 0.5], [0, 0.5], "r--", alpha=0.8)
    plt.xlabel("Real Allele Frequency")
    plt.ylabel("Generated Allele Frequency")
    plt.title(
        f'Allele Frequency Correlation\n(r = {metrics["genomic"]["allele_freq_correlation"]:.3f})'
    )

    plt.subplot(1, 3, 3)
    # Quality score radar chart would go here, simplified as bar chart
    quality_components = {
        "AF Corr": max(0, metrics["genomic"]["allele_freq_correlation"] or 0),
        "MAF Corr": max(0, metrics["genomic"]["maf_correlation"] or 0),
        "Het Corr": max(0, metrics["genomic"]["heterozygosity_correlation"] or 0),
        "KS Test": 1.0 - min(1.0, metrics["statistical"]["ks_statistic"] or 0),
        "Overall": metrics["overall_quality_score"],
    }

    bars = plt.bar(quality_components.keys(), quality_components.values())
    bars[-1].set_color("red")  # Highlight overall score
    plt.ylabel("Quality Score")
    plt.title("Quality Metrics Summary")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        output_dir / "evaluation_summary.png",
        dpi=PLOT_CONFIG["dpi"],
        bbox_inches="tight",
    )
    plt.close()

    print(f"Evaluation visualizations saved to: {output_dir}")


# =============================================================================
# LINKAGE DISEQUILIBRIUM IMPLEMENTATION
# =============================================================================


def calculate_ld(samples, max_distance=50, n_pairs=2000):
    """Calculate Linkage Disequilibrium (LD) patterns with improved sampling.

    Args:
        samples: Tensor or array of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]
        max_distance: Maximum distance between SNPs to consider
        n_pairs: Number of random SNP pairs to sample

    Returns:
        tuple: (distances, r2_values) for LD decay plotting
    """
    # Convert tensor to numpy array on CPU if needed
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    # Ensure 2D shape [batch_size, seq_len]
    if len(samples.shape) == 3:
        samples = samples.squeeze(1)

    # Get dimensions
    n_samples, n_snps = samples.shape

    # Ensure we have enough SNPs for meaningful LD analysis
    if n_snps < 10:
        return np.array([]), np.array([])

    # Initialize arrays to store results
    distances = []
    r2_values = []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Adjust max_distance and n_pairs based on sequence length
    max_distance = min(max_distance, n_snps // 2)
    if max_distance < 1:
        return np.array([]), np.array([])

    # Pre-generate all valid SNP pairs to avoid sampling issues
    valid_pairs = []
    for i in range(n_snps):
        for j in range(i + 1, min(i + max_distance + 1, n_snps)):
            distance = j - i
            if 1 <= distance <= max_distance:
                valid_pairs.append((i, j, distance))

    if len(valid_pairs) == 0:
        return np.array([]), np.array([])

    # Sample from valid pairs
    n_pairs = min(n_pairs, len(valid_pairs))
    selected_pairs = np.random.choice(len(valid_pairs), size=n_pairs, replace=False)

    for pair_idx in selected_pairs:
        snp1_idx, snp2_idx, distance = valid_pairs[pair_idx]

        # Extract genotype/dosage columns across samples
        a = samples[:, snp1_idx]
        b = samples[:, snp2_idx]

        # Compute Pearson correlation across samples and square to get r^2
        # Guard against zero variance columns which yield NaN correlations
        if np.std(a) == 0 or np.std(b) == 0:
            continue

        try:
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr):
                continue
            r2 = float(corr * corr)

            # Store results
            distances.append(distance)
            r2_values.append(r2)
        except Exception:
            continue

    return np.array(distances), np.array(r2_values)


def plot_ld_decay(
    real_samples, gen_samples, output_dir, max_distance=100, n_pairs=1000
):
    """Plot LD decay for real and generated data.

    Args:
        real_samples: Real SNP data
        gen_samples: Generated SNP data
        output_dir: Directory to save plots
        max_distance: Maximum distance between SNPs to consider
        n_pairs: Number of random SNP pairs to sample

    Returns:
        float: LD pattern correlation
    """
    # Calculate LD for both datasets
    real_distances, real_r2 = calculate_ld(real_samples, max_distance, n_pairs)
    gen_distances, gen_r2 = calculate_ld(gen_samples, max_distance, n_pairs)

    # Check if we have valid LD data
    if len(real_distances) == 0 or len(gen_distances) == 0:
        print("⚠️  WARNING: No valid LD pairs found. Skipping LD analysis.")
        return 0.0

    # Create distance bins for averaging
    max_dist = min(max_distance, max(np.max(real_distances), np.max(gen_distances)))

    # Ensure we have at least 2 bins for meaningful analysis
    if max_dist < 2:
        print("⚠️  WARNING: Maximum distance < 2. Insufficient data for LD analysis.")
        return 0.0

    bins = np.arange(1, max_dist + 1, max(1, max_dist // 20))

    # Ensure we have at least 2 bins
    if len(bins) < 2:
        bins = np.array([1, max_dist + 1])

    # Bin the data
    real_binned_r2 = []
    gen_binned_r2 = []
    bin_centers = []

    for i in range(len(bins) - 1):
        bin_start, bin_end = bins[i], bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Real data
        mask_real = (real_distances >= bin_start) & (real_distances < bin_end)
        if np.sum(mask_real) > 0:
            real_binned_r2.append(np.mean(real_r2[mask_real]))
        else:
            real_binned_r2.append(0)

        # Generated data
        mask_gen = (gen_distances >= bin_start) & (gen_distances < bin_end)
        if np.sum(mask_gen) > 0:
            gen_binned_r2.append(np.mean(gen_r2[mask_gen]))
        else:
            gen_binned_r2.append(0)

        bin_centers.append(bin_center)

    # Calculate correlation between LD patterns
    real_binned_r2 = np.array(real_binned_r2)
    gen_binned_r2 = np.array(gen_binned_r2)

    # Remove bins with no data
    valid_bins = (real_binned_r2 > 0) | (gen_binned_r2 > 0)
    if np.sum(valid_bins) > 1:
        ld_correlation = np.corrcoef(
            real_binned_r2[valid_bins], gen_binned_r2[valid_bins]
        )[0, 1]
        if np.isnan(ld_correlation):
            ld_correlation = 0.0
    else:
        ld_correlation = 0.0

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot raw data as scatter
    plt.scatter(
        real_distances, real_r2, alpha=0.3, s=10, label="Real (raw)", color="blue"
    )
    plt.scatter(
        gen_distances, gen_r2, alpha=0.3, s=10, label="Generated (raw)", color="red"
    )

    # Plot binned averages as lines
    plt.plot(
        bin_centers,
        real_binned_r2,
        "b-",
        linewidth=2,
        label="Real (binned avg)",
        marker="o",
    )
    plt.plot(
        bin_centers,
        gen_binned_r2,
        "r-",
        linewidth=2,
        label="Generated (binned avg)",
        marker="s",
    )

    plt.xlabel("Distance (SNPs)")
    plt.ylabel("LD (r²)")
    plt.title(f"Linkage Disequilibrium Decay\nCorrelation: {ld_correlation:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_dist)
    plt.ylim(0, 0.15)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ld_decay.png", dpi=PLOT_CONFIG["dpi"], bbox_inches="tight"
    )
    plt.close()

    # Add diagnostic warnings for LD analysis
    ld_warnings = []

    if ld_correlation < 0.1:
        ld_warnings.append(
            "🚨 CRITICAL: LD correlation < 0.1 - spatial correlations destroyed"
        )
    elif ld_correlation < 0.3:
        ld_warnings.append(
            "⚠️  WARNING: LD correlation < 0.3 - poor spatial correlation preservation"
        )

    # Check for flat LD patterns (indicator of staircase/uniform generation)
    if len(gen_binned_r2) > 0:
        gen_ld_std = np.std(gen_binned_r2)
        if gen_ld_std < 0.01:
            ld_warnings.append(
                "🚨 CRITICAL: Generated LD pattern is flat (std < 0.01) - possible uniform generation"
            )

    # Print warnings
    for warning in ld_warnings:
        print(warning)

    return float(ld_correlation)


# =============================================================================
# PRINCIPAL COMPONENT ANALYSIS IMPLEMENTATION
# =============================================================================


def run_pca_analysis(real_samples, gen_samples, output_dir, n_components=2):
    """Run PCA analysis on real and generated samples.

    Args:
        real_samples: Real SNP data
        gen_samples: Generated SNP data
        output_dir: Directory to save plots
        n_components: Number of PCA components to compute

    Returns:
        tuple: (avg_wasserstein_distance, component_distances)
    """
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(real_samples):
        real_samples = real_samples.cpu().numpy()
    if torch.is_tensor(gen_samples):
        gen_samples = gen_samples.cpu().numpy()

    # Ensure 2D shape [batch_size, seq_len]
    if len(real_samples.shape) == 3:
        real_samples = real_samples.squeeze(1)
    if len(gen_samples.shape) == 3:
        gen_samples = gen_samples.squeeze(1)

    # Combine data for PCA fitting
    combined = np.vstack([real_samples, gen_samples])

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(combined)

    # Transform data
    real_pca = pca.transform(real_samples)
    gen_pca = pca.transform(gen_samples)

    # Calculate Wasserstein distance between distributions
    w_distances = []
    for i in range(n_components):
        w_distances.append(wasserstein_distance(real_pca[:, i], gen_pca[:, i]))

    avg_w_distance = np.mean(w_distances)

    # Plot PCA results
    plt.figure(figsize=PLOT_CONFIG["figure_size"])

    # Plot first two components
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label="Real", s=30)
    plt.scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label="Generated", s=30)

    # Add explained variance
    explained_var = pca.explained_variance_ratio_

    # Plot formatting
    plt.title("PCA of Real and Generated SNP Data", fontsize=14)
    plt.xlabel(f"PC1 ({explained_var[0]:.2%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({explained_var[1]:.2%} variance)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add Wasserstein distance text
    plt.text(
        0.95,
        0.95,
        f"Avg. Wasserstein Distance: {avg_w_distance:.4f}\n"
        f"PC1 W-Distance: {w_distances[0]:.4f}\n"
        f"PC2 W-Distance: {w_distances[1]:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "pca_analysis.png", dpi=300)
    plt.close()

    # If we have more than 2 components, create a scree plot
    if n_components > 2:
        plt.figure(figsize=PLOT_CONFIG["figure_size"])
        plt.bar(range(1, n_components + 1), explained_var)
        plt.plot(range(1, n_components + 1), np.cumsum(explained_var), "r-o")

        plt.title("PCA Scree Plot", fontsize=PLOT_CONFIG["font_size_title"])
        plt.xlabel("Principal Component", fontsize=PLOT_CONFIG["font_size_label"])
        plt.ylabel("Explained Variance Ratio", fontsize=PLOT_CONFIG["font_size_label"])
        plt.xticks(range(1, n_components + 1))
        plt.grid(True, alpha=PLOT_CONFIG["grid_alpha"])

        plt.tight_layout()
        plt.savefig(output_dir / "pca_scree_plot.png", dpi=PLOT_CONFIG["dpi"])
        plt.close()

    return avg_w_distance, w_distances


# =============================================================================
# GENETIC DIVERSITY IMPLEMENTATION
# =============================================================================


def calculate_genetic_diversity(samples):
    """Calculate genetic diversity metrics.

    Args:
        samples: SNP data

    Returns:
        dict: Dictionary of genetic diversity metrics
    """
    # Convert tensor to numpy array if needed
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()

    # Ensure 2D shape [batch_size, seq_len]
    if len(samples.shape) == 3:
        samples = samples.squeeze(1)

    # Calculate allele frequencies
    freq = np.mean(samples, axis=0)

    # Calculate heterozygosity (expected)
    het_exp = 2 * freq * (1 - freq)
    mean_het_exp = float(np.mean(het_exp))

    # Calculate heterozygosity (observed)
    # For discrete SNP data with values [0.0, 0.5, 1.0], heterozygotes are exactly 0.5
    # Use a small tolerance for floating point comparison
    het_obs = np.mean(np.abs(samples - 0.5) < 1e-6, axis=0)
    mean_het_obs = float(np.mean(het_obs))

    # Calculate polymorphic sites ratio
    polymorphic = np.logical_and(freq > 0, freq < 1)
    polymorphic_ratio = float(np.mean(polymorphic))

    # Calculate nucleotide diversity (pi)
    # For SNP data, this is approximately the average heterozygosity
    pi = mean_het_exp

    return {
        "expected_heterozygosity": mean_het_exp,
        "observed_heterozygosity": mean_het_obs,
        "polymorphic_ratio": polymorphic_ratio,
        "nucleotide_diversity": pi,
    }


# =============================================================================
# ADVANCED GENOMIC ANALYSIS PLOTS
# =============================================================================


def plot_haplotype_blocks(real_samples, generated_samples, output_dir, window_size=50):
    """
    Create heatmap showing linkage patterns across genomic regions.

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save plots
        window_size: Size of sliding window for LD calculation
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    def calculate_ld_matrix(data, max_snps=100):
        """Calculate LD matrix for visualization (limited for performance)"""
        n_snps = min(data.shape[1], max_snps)
        data_subset = data[:, :n_snps]
        ld_matrix = np.zeros((n_snps, n_snps))

        for i in range(n_snps):
            for j in range(i, n_snps):
                if i == j:
                    ld_matrix[i, j] = 1.0
                else:
                    # Calculate r-squared
                    corr = np.corrcoef(data_subset[:, i], data_subset[:, j])[0, 1]
                    ld_matrix[i, j] = ld_matrix[j, i] = (
                        corr**2 if not np.isnan(corr) else 0
                    )

        return ld_matrix

    # Calculate LD matrices
    real_ld = calculate_ld_matrix(real)
    gen_ld = calculate_ld_matrix(gen)

    # Create heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Real data LD
    im1 = axes[0].imshow(real_ld, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Real Data LD Structure")
    axes[0].set_xlabel("SNP Position")
    axes[0].set_ylabel("SNP Position")
    plt.colorbar(im1, ax=axes[0], label="r²")

    # Generated data LD
    im2 = axes[1].imshow(gen_ld, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Generated Data LD Structure")
    axes[1].set_xlabel("SNP Position")
    axes[1].set_ylabel("SNP Position")
    plt.colorbar(im2, ax=axes[1], label="r²")

    # Difference
    ld_diff = np.abs(real_ld - gen_ld)
    im3 = axes[2].imshow(ld_diff, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("LD Structure Difference")
    axes[2].set_xlabel("SNP Position")
    axes[2].set_ylabel("SNP Position")
    plt.colorbar(im3, ax=axes[2], label="|Δr²|")

    plt.tight_layout()
    plt.savefig(
        output_dir / "haplotype_blocks.png", dpi=PLOT_CONFIG["dpi"], bbox_inches="tight"
    )
    plt.close()

    # Calculate summary metric
    ld_similarity = 1.0 - np.mean(ld_diff)
    print(f"🧬 Haplotype Block Similarity: {ld_similarity:.3f}")
    return float(ld_similarity)


def plot_maf_spectrum(real_samples, generated_samples, output_dir, max_value=0.5):
    """
    Create histogram comparing Minor Allele Frequency distributions.

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save plots
        max_value: Maximum allele frequency value
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    # Calculate allele frequencies
    real_af = np.mean(real, axis=0)
    gen_af = np.mean(gen, axis=0)

    # Calculate MAF (Minor Allele Frequency)
    real_maf = np.minimum(real_af, max_value - real_af)
    gen_maf = np.minimum(gen_af, max_value - gen_af)

    # Create MAF spectrum plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MAF distributions
    axes[0, 0].hist(
        real_maf, bins=50, alpha=0.7, label="Real", density=True, color="blue"
    )
    axes[0, 0].hist(
        gen_maf, bins=50, alpha=0.7, label="Generated", density=True, color="red"
    )
    axes[0, 0].set_xlabel("Minor Allele Frequency")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("MAF Distribution Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAF correlation
    if np.std(real_maf) == 0 or np.std(gen_maf) == 0:
        maf_corr = 0.0
    else:
        maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]
        if np.isnan(maf_corr):
            maf_corr = 0.0
    axes[0, 1].scatter(real_maf, gen_maf, alpha=0.6, s=10)
    axes[0, 1].plot([0, max_value / 2], [0, max_value / 2], "r--", alpha=0.8)
    axes[0, 1].set_xlabel("Real MAF")
    axes[0, 1].set_ylabel("Generated MAF")
    axes[0, 1].set_title(f"MAF Correlation (r = {maf_corr:.3f})")
    axes[0, 1].grid(True, alpha=0.3)

    # Rare variants (MAF < 0.05)
    rare_threshold = 0.05
    real_rare = np.sum(real_maf < rare_threshold)
    gen_rare = np.sum(gen_maf < rare_threshold)

    axes[1, 0].bar(
        ["Real", "Generated"], [real_rare, gen_rare], color=["blue", "red"], alpha=0.7
    )
    axes[1, 0].set_ylabel("Number of Rare Variants")
    axes[1, 0].set_title(f"Rare Variants (MAF < {rare_threshold})")
    axes[1, 0].grid(True, alpha=0.3)

    # MAF residuals
    maf_residuals = gen_maf - real_maf
    axes[1, 1].scatter(real_maf, maf_residuals, alpha=0.6, s=10)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.8)
    axes[1, 1].set_xlabel("Real MAF")
    axes[1, 1].set_ylabel("MAF Residuals (Gen - Real)")
    axes[1, 1].set_title("MAF Prediction Errors")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "maf_spectrum.png", dpi=PLOT_CONFIG["dpi"], bbox_inches="tight"
    )
    plt.close()

    print(f"🔬 MAF Correlation: {maf_corr:.3f}")
    print(f"📊 Rare Variants - Real: {real_rare}, Generated: {gen_rare}")
    return float(maf_corr)


def plot_hardy_weinberg_deviation(real_samples, generated_samples, output_dir):
    """
    Create scatter plot of Hardy-Weinberg Equilibrium deviations.

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save plots
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    def calculate_hwe_stats(data):
        """Calculate Hardy-Weinberg statistics for each SNP"""
        n_samples, n_snps = data.shape
        hwe_stats = []

        for i in range(n_snps):
            snp_data = data[:, i]

            # Count genotypes (assuming 0, 0.25, 0.5 for diploid)
            # For simplicity, treat as allele frequencies
            p = np.mean(snp_data)  # Frequency of reference allele
            q = 0.5 - p  # Frequency of alternative allele (assuming max_value=0.5)

            if p <= 0 or q <= 0:
                hwe_stats.append(0)
                continue

            # Expected heterozygosity under HWE
            expected_het = 2 * p * q

            # Observed heterozygosity (approximate for continuous data)
            # Count values that are not at extremes
            het_mask = (snp_data > 0.1) & (snp_data < 0.4)  # Approximate heterozygotes
            observed_het = np.mean(het_mask)

            # HWE deviation (Inbreeding coefficient approximation)
            if expected_het > 0:
                f_is = 1 - (observed_het / expected_het)
            else:
                f_is = 0

            hwe_stats.append(f_is)

        return np.array(hwe_stats)

    # Calculate HWE statistics
    real_hwe = calculate_hwe_stats(real)
    gen_hwe = calculate_hwe_stats(gen)

    # Create HWE deviation plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # HWE deviation distributions
    axes[0, 0].hist(
        real_hwe, bins=30, alpha=0.7, label="Real", density=True, color="blue"
    )
    axes[0, 0].hist(
        gen_hwe, bins=30, alpha=0.7, label="Generated", density=True, color="red"
    )
    axes[0, 0].set_xlabel("Inbreeding Coefficient (F_IS)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("HWE Deviation Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0, color="black", linestyle="--", alpha=0.5, label="HWE")

    # HWE correlation
    if np.std(real_hwe) == 0 or np.std(gen_hwe) == 0:
        hwe_corr = 0.0
    else:
        hwe_corr = np.corrcoef(real_hwe, gen_hwe)[0, 1]
        if np.isnan(hwe_corr):
            hwe_corr = 0.0
    axes[0, 1].scatter(real_hwe, gen_hwe, alpha=0.6, s=10)
    axes[0, 1].plot([-1, 1], [-1, 1], "r--", alpha=0.8)
    axes[0, 1].set_xlabel("Real F_IS")
    axes[0, 1].set_ylabel("Generated F_IS")
    axes[0, 1].set_title(f"HWE Deviation Correlation (r = {hwe_corr:.3f})")
    axes[0, 1].grid(True, alpha=0.3)

    # SNP position vs HWE deviation
    positions = np.arange(len(real_hwe))
    axes[1, 0].scatter(positions, real_hwe, alpha=0.6, s=5, label="Real", color="blue")
    axes[1, 0].scatter(
        positions, gen_hwe, alpha=0.6, s=5, label="Generated", color="red"
    )
    axes[1, 0].set_xlabel("SNP Position")
    axes[1, 0].set_ylabel("F_IS")
    axes[1, 0].set_title("HWE Deviation by Position")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # HWE residuals
    hwe_residuals = gen_hwe - real_hwe
    axes[1, 1].scatter(real_hwe, hwe_residuals, alpha=0.6, s=10)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.8)
    axes[1, 1].set_xlabel("Real F_IS")
    axes[1, 1].set_ylabel("F_IS Residuals (Gen - Real)")
    axes[1, 1].set_title("HWE Prediction Errors")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "hardy_weinberg_deviation.png",
        dpi=PLOT_CONFIG["dpi"],
        bbox_inches="tight",
    )
    plt.close()

    print(f"🧪 HWE Deviation Correlation: {hwe_corr:.3f}")
    return float(hwe_corr)


def plot_genomic_position_effects(
    real_samples, generated_samples, output_dir, window_size=50
):
    """
    Create line plot showing quality metrics across genomic positions.

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save plots
        window_size: Size of sliding window
    """
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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    n_snps = real.shape[1]
    n_windows = max(1, n_snps // window_size)

    # Calculate metrics in sliding windows
    window_positions = []
    af_correlations = []
    maf_correlations = []
    mean_differences = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, n_snps)

        if end_idx - start_idx < 10:  # Skip small windows
            continue

        # Extract window data
        real_window = real[:, start_idx:end_idx]
        gen_window = gen[:, start_idx:end_idx]

        # Calculate window metrics
        real_af = np.mean(real_window, axis=0)
        gen_af = np.mean(gen_window, axis=0)

        if len(real_af) > 1 and np.std(real_af) > 0 and np.std(gen_af) > 0:
            af_corr = np.corrcoef(real_af, gen_af)[0, 1]
            if np.isnan(af_corr):
                af_corr = 0.0
        else:
            af_corr = 0.0

        real_maf = np.minimum(real_af, 0.5 - real_af)
        gen_maf = np.minimum(gen_af, 0.5 - gen_af)
        if len(real_maf) > 1 and np.std(real_maf) > 0 and np.std(gen_maf) > 0:
            maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]
            if np.isnan(maf_corr):
                maf_corr = 0.0
        else:
            maf_corr = 0.0

        mean_diff = np.abs(np.mean(real_window) - np.mean(gen_window))

        window_positions.append((start_idx + end_idx) / 2)
        af_correlations.append(af_corr if not np.isnan(af_corr) else 0)
        maf_correlations.append(maf_corr if not np.isnan(maf_corr) else 0)
        mean_differences.append(mean_diff)

    # Create position effects plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # AF correlation by position
    axes[0, 0].plot(window_positions, af_correlations, "b-", linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel("Genomic Position")
    axes[0, 0].set_ylabel("AF Correlation")
    axes[0, 0].set_title("Allele Frequency Correlation by Position")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect")
    axes[0, 0].legend()

    # MAF correlation by position
    axes[0, 1].plot(window_positions, maf_correlations, "g-", linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel("Genomic Position")
    axes[0, 1].set_ylabel("MAF Correlation")
    axes[0, 1].set_title("Minor Allele Frequency Correlation by Position")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Perfect")
    axes[0, 1].legend()

    # Mean difference by position
    axes[1, 0].plot(window_positions, mean_differences, "r-", linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel("Genomic Position")
    axes[1, 0].set_ylabel("Mean Difference")
    axes[1, 0].set_title("Mean Value Difference by Position")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.0, color="black", linestyle="--", alpha=0.5, label="Perfect")
    axes[1, 0].legend()

    # Summary statistics
    avg_af_corr = np.mean(af_correlations)
    avg_maf_corr = np.mean(maf_correlations)
    avg_mean_diff = np.mean(mean_differences)

    summary_metrics = ["AF Correlation", "MAF Correlation", "Mean Difference"]
    summary_values = [
        avg_af_corr,
        avg_maf_corr,
        1.0 - avg_mean_diff,
    ]  # Invert mean diff for consistency

    bars = axes[1, 1].bar(
        summary_metrics, summary_values, color=["blue", "green", "red"], alpha=0.7
    )
    axes[1, 1].set_ylabel("Average Quality Score")
    axes[1, 1].set_title("Position-wise Quality Summary")
    axes[1, 1].set_ylim(0, 1.07)
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, summary_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "genomic_position_effects.png",
        dpi=PLOT_CONFIG["dpi"],
        bbox_inches="tight",
    )
    plt.close()

    print(f"📍 Average AF Correlation by Position: {avg_af_corr:.3f}")
    print(f"📍 Average MAF Correlation by Position: {avg_maf_corr:.3f}")
    return float(avg_af_corr)


def plot_sample_clustering(
    real_samples, generated_samples, output_dir, max_samples=100
):
    """
    Create hierarchical clustering heatmap of sample similarities.

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save plots
        max_samples: Maximum number of samples to include (for performance)
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

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

    if real.ndim == 3:
        real = real.squeeze(1)
    if gen.ndim == 3:
        gen = gen.squeeze(1)

    # Subsample for performance
    n_real = min(max_samples // 2, real.shape[0])
    n_gen = min(max_samples // 2, gen.shape[0])

    real_subset = real[:n_real]
    gen_subset = gen[:n_gen]

    # Combine samples
    all_samples = np.vstack([real_subset, gen_subset])
    sample_labels = ["Real"] * n_real + ["Generated"] * n_gen

    # Calculate pairwise distances
    distances = pdist(all_samples, metric="euclidean")

    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method="ward")

    # Create clustering plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Dendrogram
    dendro = dendrogram(
        linkage_matrix,
        ax=axes[0],
        labels=sample_labels,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    axes[0].set_title("Sample Clustering Dendrogram")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("Distance")

    # Distance matrix heatmap
    from scipy.spatial.distance import squareform

    distance_matrix = squareform(distances)

    im = axes[1].imshow(distance_matrix, cmap="viridis", aspect="auto")
    axes[1].set_title("Sample Distance Matrix")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Sample Index")

    # Add colorbar
    plt.colorbar(im, ax=axes[1], label="Euclidean Distance")

    # Add dividing lines to show real vs generated
    axes[1].axhline(y=n_real - 0.5, color="red", linestyle="--", alpha=0.8, linewidth=2)
    axes[1].axvline(x=n_real - 0.5, color="red", linestyle="--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig(
        output_dir / "sample_clustering.png",
        dpi=PLOT_CONFIG["dpi"],
        bbox_inches="tight",
    )
    plt.close()

    # Calculate clustering quality metric
    # Average within-group distance vs between-group distance
    real_indices = np.arange(n_real)
    gen_indices = np.arange(n_real, n_real + n_gen)

    # Within-group distances
    real_within = distance_matrix[np.ix_(real_indices, real_indices)]
    gen_within = distance_matrix[np.ix_(gen_indices, gen_indices)]
    avg_within = (np.mean(real_within) + np.mean(gen_within)) / 2

    # Between-group distances
    between = distance_matrix[np.ix_(real_indices, gen_indices)]
    avg_between = np.mean(between)

    # Clustering quality (lower is better - samples should be similar)
    clustering_score = avg_within / avg_between if avg_between > 0 else 1.0

    print(f"🔗 Sample Clustering Score: {clustering_score:.3f} (lower is better)")
    print(f"📊 Average within-group distance: {avg_within:.3f}")
    print(f"📊 Average between-group distance: {avg_between:.3f}")

    return float(clustering_score)


# =============================================================================
# REPORTING AND SUMMARY FUNCTIONS
# =============================================================================


def print_evaluation_summary(metrics):
    """
    Print a human-readable summary of evaluation results.

    Args:
        metrics: Dictionary of computed metrics (new structure)
    """
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE SAMPLE ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n🎯 Overall Quality Score: {metrics['overall_quality_score']:.3f}/1.000")

    # Basic quality metrics (from infer_utils.py)
    print(f"\n📊 Basic Quality Assessment: {metrics['basic_quality_score']:.3f}/1.000")

    # Dimensionality metrics
    if metrics.get("dimensionality"):
        dim_metrics = metrics["dimensionality"]
        print("\n🔍 High-Dimensional Analysis:")
        if dim_metrics.get("pca_centroid_distance") is not None:
            print(
                f"  • PCA Centroid Distance: {dim_metrics['pca_centroid_distance']:.6f}"
            )
        if dim_metrics.get("pca_2d_overlap") is not None:
            print(f"  • PCA 2D Overlap: {dim_metrics['pca_2d_overlap']:.3f}")

    # Advanced genomic analysis
    if metrics.get("advanced_genomics"):
        adv_metrics = metrics["advanced_genomics"]
        print("\n🧬 Advanced Genomic Analysis:")

        if adv_metrics.get("ld_pattern_correlation") is not None:
            print(
                f"  • LD Pattern Correlation: {adv_metrics['ld_pattern_correlation']:.3f}"
            )

        if adv_metrics.get("pca_wasserstein_distance") is not None:
            print(
                f"  • PCA Wasserstein Distance: {adv_metrics['pca_wasserstein_distance']:.6f}"
            )

        if adv_metrics.get("genetic_diversity_real") and adv_metrics.get(
            "genetic_diversity_generated"
        ):
            real_div = adv_metrics["genetic_diversity_real"]
            gen_div = adv_metrics["genetic_diversity_generated"]
            print(
                f"  • Expected Heterozygosity (Real): {real_div['expected_heterozygosity']:.3f}"
            )
            print(
                f"  • Expected Heterozygosity (Generated): {gen_div['expected_heterozygosity']:.3f}"
            )
            print(f"  • Polymorphic Ratio (Real): {real_div['polymorphic_ratio']:.3f}")
            print(
                f"  • Polymorphic Ratio (Generated): {gen_div['polymorphic_ratio']:.3f}"
            )

    # Interpretation
    quality = metrics["overall_quality_score"]
    if quality >= 0.9:
        interpretation = "🟢 EXCELLENT - Generated samples are very close to real data"
    elif quality >= 0.8:
        interpretation = (
            "🟡 GOOD - Generated samples are reasonably similar to real data"
        )
    elif quality >= 0.7:
        interpretation = "🟠 FAIR - Generated samples show some similarity to real data"
    else:
        interpretation = (
            "🔴 POOR - Generated samples differ significantly from real data"
        )

    print(f"\n💡 Interpretation: {interpretation}")
    print("=" * 60)


def create_evaluation_report(metrics, real_samples, generated_samples, output_dir):
    """Create a comprehensive evaluation report in markdown format."""

    report_file = output_dir / "evaluation_report.md"

    # Convert samples to numpy for statistics
    real_np = (
        real_samples.cpu().numpy() if torch.is_tensor(real_samples) else real_samples
    )
    gen_np = (
        generated_samples.cpu().numpy()
        if torch.is_tensor(generated_samples)
        else generated_samples
    )

    if real_np.ndim == 3:
        real_np = real_np.squeeze(1)
    if gen_np.ndim == 3:
        gen_np = gen_np.squeeze(1)

    with open(report_file, "w") as f:
        f.write("# Genomic Sample Evaluation Report\n\n")

        # Basic info
        f.write("## Dataset Information\n\n")
        f.write(f"- **Real Data Shape**: {real_samples.shape}\n")
        f.write(f"- **Generated Data Shape**: {generated_samples.shape}\n")
        f.write(f"- **Sequence Length**: {real_samples.shape[-1]}\n")
        f.write(
            f"- **Number of Samples**: Real: {real_samples.shape[0]}, Generated: {generated_samples.shape[0]}\n"
        )
        f.write(f"- **Evaluation Date**: {Path().cwd()}\n\n")

        # Overall score
        quality = metrics["overall_quality_score"]
        f.write("## Overall Quality Assessment\n\n")
        f.write(f"**Quality Score: {quality:.3f}/1.000**\n\n")

        if quality >= 0.9:
            interpretation = (
                "🟢 **EXCELLENT** - Generated samples are very close to real data"
            )
        elif quality >= 0.8:
            interpretation = (
                "🟡 **GOOD** - Generated samples are reasonably similar to real data"
            )
        elif quality >= 0.7:
            interpretation = (
                "🟠 **FAIR** - Generated samples show some similarity to real data"
            )
        else:
            interpretation = (
                "🔴 **POOR** - Generated samples differ significantly from real data"
            )

        f.write(f"{interpretation}\n\n")

        # Detailed metrics
        f.write("## Detailed Metrics\n\n")

        f.write("### Critical Diagnostic Metrics\n")
        f.write(
            f"- **Mode Collapse Detected**: {metrics.get('mode_collapse_detected', 'Unknown')}\n"
        )
        f.write(
            f"- **Sample Diversity Ratio**: {metrics.get('sample_diversity_ratio', 0):.3f}\n"
        )
        f.write(f"- **LD Correlation**: {metrics.get('ld_correlation', 0):.6f}\n")
        f.write(f"- **Status**: {metrics.get('status', 'Unknown')}\n\n")

        f.write("### Genomic Similarity\n")
        f.write(f"- **MAF Correlation**: {metrics.get('maf_correlation', 0):.6f}\n")
        f.write(
            f"- **Genetic Diversity (Real)**: {metrics.get('genetic_diversity_real', 0):.6f}\n"
        )
        f.write(
            f"- **Genetic Diversity (Generated)**: {metrics.get('genetic_diversity_generated', 0):.6f}\n"
        )
        f.write(f"- **PCA Distance**: {metrics.get('pca_distance', 0):.4f}\n\n")

        f.write("### Additional Metrics\n")
        f.write(
            f"- **Haplotype Block Similarity**: {metrics.get('haplotype_similarity', 0):.3f}\n"
        )
        f.write(
            f"- **HWE Deviation Correlation**: {metrics.get('hwe_correlation', 0):.3f}\n"
        )
        f.write(
            f"- **Position Effects Correlation**: {metrics.get('position_correlation', 0):.3f}\n"
        )
        f.write(
            f"- **Sample Clustering Score**: {metrics.get('clustering_score', 0):.3f}\n\n"
        )

        f.write("### High-Dimensional Similarity\n")
        f.write(
            f"- PCA Centroid Distance: {metrics['dimensionality']['pca_centroid_distance']:.6f}\n"
        )
        f.write(
            f"- PCA 2D Overlap: {metrics['dimensionality']['pca_2d_overlap']:.3f}\n\n"
        )

        # Basic statistics comparison
        f.write("## Sample Statistics Comparison\n\n")
        f.write("| Metric | Real Samples | Generated Samples | Difference |\n")
        f.write("|--------|-------------|------------------|------------|\n")
        f.write(
            f"| Mean | {np.mean(real_np):.4f} | {np.mean(gen_np):.4f} | {abs(np.mean(real_np) - np.mean(gen_np)):.4f} |\n"
        )
        f.write(
            f"| Std Dev | {np.std(real_np):.4f} | {np.std(gen_np):.4f} | {abs(np.std(real_np) - np.std(gen_np)):.4f} |\n"
        )
        f.write(
            f"| Min | {np.min(real_np):.4f} | {np.min(gen_np):.4f} | {abs(np.min(real_np) - np.min(gen_np)):.4f} |\n"
        )
        f.write(
            f"| Max | {np.max(real_np):.4f} | {np.max(gen_np):.4f} | {abs(np.max(real_np) - np.max(gen_np)):.4f} |\n\n"
        )

        # Files generated
        f.write("## Generated Files\n\n")
        f.write("- `comprehensive_evaluation.json` - Complete metrics in JSON format\n")
        f.write("- `sample_comparison.png` - Side-by-side sample comparison\n")
        f.write("- `sample_visualization.png` - Detailed sample visualization\n")
        f.write("- `evaluation_summary.png` - Visual summary of key metrics\n")
        f.write("- `evaluation_report.md` - This report\n\n")

        f.write("---\n")
        f.write("*Report generated by GenomeDiffusion evaluation pipeline*\n")

    print(f"📄 Evaluation report saved to: {report_file}")


# =============================================================================
# COMPREHENSIVE DIAGNOSTIC EVALUATION
# =============================================================================


def run_comprehensive_evaluation(real_samples, generated_samples, output_dir):
    """
    Run comprehensive diagnostic evaluation with improved metrics and warnings.

    This function integrates all analysis metrics in a meaningful order:
    1. Sample diversity analysis (detects mode collapse)
    2. Linkage disequilibrium analysis (detects spatial correlation preservation)
    3. Traditional genomic metrics (MAF, diversity, etc.)
    4. Overall assessment with diagnostic warnings

    Args:
        real_samples: Real genomic data
        generated_samples: Generated genomic data
        output_dir: Directory to save results

    Returns:
        dict: Comprehensive evaluation results with diagnostics
    """
    print("\n" + "=" * 80)
    print("🧬 COMPREHENSIVE GENOMIC MODEL EVALUATION")
    print("=" * 80)

    results = {}

    # 1. CRITICAL: Sample Diversity Analysis (Mode Collapse Detection)
    print("\n🔍 1. SAMPLE DIVERSITY ANALYSIS (Mode Collapse Detection)")
    print("-" * 60)

    mode_collapse_results = detect_mode_collapse(real_samples, generated_samples)
    results["mode_collapse"] = mode_collapse_results

    print(
        f"Real sample diversity: {mode_collapse_results['real_diversity']['effective_diversity']:.3f}"
    )
    print(
        f"Generated sample diversity: {mode_collapse_results['generated_diversity']['effective_diversity']:.3f}"
    )
    print(f"Diversity ratio: {mode_collapse_results['diversity_ratio']:.3f}")
    print(f"Sample correlation ratio: {mode_collapse_results['correlation_ratio']:.3f}")

    if mode_collapse_results["warnings"]:
        print("\n⚠️  DIAGNOSTIC WARNINGS:")
        for warning in mode_collapse_results["warnings"]:
            print(f"   {warning}")
    else:
        print("✅ No mode collapse detected")

    # 2. CRITICAL: Linkage Disequilibrium Analysis (Spatial Correlation Preservation)
    print("\n🧬 2. LINKAGE DISEQUILIBRIUM ANALYSIS (Spatial Correlation Preservation)")
    print("-" * 70)

    ld_correlation = plot_ld_decay(real_samples, generated_samples, output_dir)
    results["ld_correlation"] = ld_correlation
    print(f"LD correlation: {ld_correlation:.6f}")

    # 3. Traditional Genomic Metrics
    print("\n📊 3. TRADITIONAL GENOMIC METRICS")
    print("-" * 40)

    # MAF analysis
    maf_correlation = plot_maf_spectrum(real_samples, generated_samples, output_dir)
    results["maf_correlation"] = maf_correlation
    print(f"MAF correlation: {maf_correlation:.6f}")

    # Genetic diversity
    real_diversity = calculate_genetic_diversity(real_samples)
    gen_diversity = calculate_genetic_diversity(generated_samples)
    results["genetic_diversity"] = {
        "real": real_diversity["nucleotide_diversity"],
        "generated": gen_diversity["nucleotide_diversity"],
        "difference": abs(
            real_diversity["nucleotide_diversity"]
            - gen_diversity["nucleotide_diversity"]
        ),
    }
    print(
        f"Genetic diversity - Real: {real_diversity['nucleotide_diversity']:.6f}, Generated: {gen_diversity['nucleotide_diversity']:.6f}"
    )

    # PCA analysis
    pca_distance, _ = run_pca_analysis(real_samples, generated_samples, output_dir)
    results["pca_distance"] = pca_distance
    print(f"PCA distance: {pca_distance:.6f}")

    # 4. Overall Assessment with Diagnostic Interpretation
    print("\n🎯 4. OVERALL ASSESSMENT")
    print("-" * 30)

    # Calculate weighted score with emphasis on critical metrics
    weights = {
        "ld_correlation": 0.4,  # Most important - spatial structure
        "diversity_ratio": 0.3,  # Second most important - sample diversity
        "maf_correlation": 0.15,  # Traditional metric
        "genetic_diversity_similarity": 0.1,  # Traditional metric
        "pca_distance_inv": 0.05,  # Traditional metric
    }

    # Normalize metrics to 0-1 scale
    ld_score = max(0, min(1, ld_correlation))
    diversity_score = max(0, min(1, mode_collapse_results["diversity_ratio"]))
    maf_score = max(0, min(1, maf_correlation))
    genetic_div_score = max(0, min(1, 1 - results["genetic_diversity"]["difference"]))
    pca_score = max(0, min(1, 1 - min(pca_distance, 1)))

    overall_score = (
        weights["ld_correlation"] * ld_score
        + weights["diversity_ratio"] * diversity_score
        + weights["maf_correlation"] * maf_score
        + weights["genetic_diversity_similarity"] * genetic_div_score
        + weights["pca_distance_inv"] * pca_score
    )

    results["overall_score"] = overall_score
    results["component_scores"] = {
        "ld_score": ld_score,
        "diversity_score": diversity_score,
        "maf_score": maf_score,
        "genetic_diversity_score": genetic_div_score,
        "pca_score": pca_score,
    }

    # Diagnostic interpretation
    print(f"Overall Score: {overall_score:.3f}/1.000")

    # Critical failure detection
    critical_failures = []
    if ld_correlation < 0.1:
        critical_failures.append(
            "Spatial correlations destroyed (LD correlation < 0.1)"
        )
    if mode_collapse_results["mode_collapse_detected"]:
        critical_failures.append("Mode collapse detected")
    if mode_collapse_results["diversity_ratio"] < 0.1:
        critical_failures.append("Severe lack of sample diversity")

    if critical_failures:
        print("\n🚨 CRITICAL FAILURES DETECTED:")
        for failure in critical_failures:
            print(f"   • {failure}")
        print(
            "\n💡 DIAGNOSIS: Model likely learned uniform/staircase pattern instead of real genomic variation"
        )
        print(
            "   Recommendation: Retrain with real genomic data, check loss function and architecture"
        )
        results["status"] = "CRITICAL_FAILURE"
    elif overall_score < 0.5:
        print(f"\n⚠️  POOR PERFORMANCE (Score: {overall_score:.3f})")
        print("   Model shows significant issues with genomic data generation")
        results["status"] = "POOR"
    elif overall_score < 0.8:
        print(f"\n🟡 MODERATE PERFORMANCE (Score: {overall_score:.3f})")
        print("   Model shows some issues but captures basic genomic properties")
        results["status"] = "MODERATE"
    else:
        print(f"\n🟢 GOOD PERFORMANCE (Score: {overall_score:.3f})")
        print("   Model successfully captures genomic properties and structure")
        results["status"] = "GOOD"

    # Save comprehensive results
    results_file = output_dir / "comprehensive_evaluation.json"
    import json

    with open(results_file, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)

    print(f"\n📄 Comprehensive results saved to: {results_file}")
    print("=" * 80)

    return results
