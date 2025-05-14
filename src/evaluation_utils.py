#!/usr/bin/env python
# coding: utf-8

"""Utility functions for comprehensive evaluation of diffusion models.

This module contains specialized evaluation functions for:
1. Linkage Disequilibrium (LD) analysis
2. Principal Component Analysis (PCA) visualization
3. Genetic diversity metrics calculation
4. Advanced statistical comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


def calculate_ld(samples, max_distance=100, n_pairs=1000):
    """Calculate Linkage Disequilibrium (LD) patterns.

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

    # Initialize arrays to store results
    distances = []
    r2_values = []

    # Sample random pairs of SNPs within max_distance
    np.random.seed(42)  # For reproducibility

    # Try to sample n_pairs, but don't exceed available pairs
    max_possible_pairs = n_snps * max_distance
    n_pairs = min(n_pairs, max_possible_pairs)

    sampled_pairs = 0
    max_attempts = n_pairs * 10  # Limit attempts to avoid infinite loops
    attempts = 0

    while sampled_pairs < n_pairs and attempts < max_attempts:
        attempts += 1

        # Sample first SNP
        snp1_idx = np.random.randint(0, n_snps - 1)

        # Sample second SNP within max_distance
        max_idx = min(snp1_idx + max_distance, n_snps - 1)
        min_idx = max(snp1_idx - max_distance, 0)

        # Ensure we don't sample the same SNP
        if max_idx == snp1_idx:
            continue
        if min_idx == snp1_idx:
            continue

        # Randomly choose direction (up or down)
        if np.random.random() < 0.5 and snp1_idx > min_idx:
            # Sample below
            snp2_idx = np.random.randint(min_idx, snp1_idx)
        elif snp1_idx < max_idx:
            # Sample above
            snp2_idx = np.random.randint(snp1_idx + 1, max_idx + 1)
        else:
            continue

        # Calculate distance
        distance = abs(snp2_idx - snp1_idx)

        # Extract alleles
        a = samples[:, snp1_idx]
        b = samples[:, snp2_idx]

        # Calculate allele frequencies
        p_a = np.mean(a)
        p_b = np.mean(b)

        # Skip if either SNP is monomorphic (no variation)
        if p_a == 0 or p_a == 1 or p_b == 0 or p_b == 1:
            continue

        # Calculate observed haplotype frequency
        p_ab = np.mean(a * b)

        # Calculate D
        d = p_ab - (p_a * p_b)

        # Calculate D'
        if d > 0:
            d_max = min(p_a * (1 - p_b), (1 - p_a) * p_b)
        else:
            d_max = min(p_a * p_b, (1 - p_a) * (1 - p_b))

        # Avoid division by zero
        if d_max == 0:
            continue

        d_prime = d / d_max

        # Calculate r^2
        denominator = p_a * (1 - p_a) * p_b * (1 - p_b)
        if denominator == 0:
            continue

        r2 = (d * d) / denominator

        # Store results
        distances.append(distance)
        r2_values.append(r2)
        sampled_pairs += 1

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
    # Calculate LD for real and generated data
    real_distances, real_r2 = calculate_ld(real_samples, max_distance, n_pairs)
    gen_distances, gen_r2 = calculate_ld(gen_samples, max_distance, n_pairs)

    # Create binned averages for smoother curves
    bin_edges = np.linspace(1, max_distance, 20)
    real_binned = np.zeros(len(bin_edges) - 1)
    gen_binned = np.zeros(len(bin_edges) - 1)

    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find r2 values in this bin
        real_mask = (real_distances >= bin_start) & (real_distances < bin_end)
        gen_mask = (gen_distances >= bin_start) & (gen_distances < bin_end)

        # Calculate average r2 in bin
        if np.sum(real_mask) > 0:
            real_binned[i] = np.mean(real_r2[real_mask])
        if np.sum(gen_mask) > 0:
            gen_binned[i] = np.mean(gen_r2[gen_mask])

    # Calculate correlation between binned values
    valid_bins = (real_binned > 0) & (gen_binned > 0)
    if np.sum(valid_bins) >= 2:
        ld_corr = np.corrcoef(real_binned[valid_bins], gen_binned[valid_bins])[0, 1]
    else:
        ld_corr = np.nan

    # Plot LD decay
    plt.figure(figsize=(12, 8))

    # Plot scatter and binned averages
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.scatter(
        real_distances, real_r2, alpha=0.3, s=10, label="Real (raw)", color="blue"
    )
    plt.scatter(
        gen_distances, gen_r2, alpha=0.3, s=10, label="Generated (raw)", color="orange"
    )

    plt.plot(bin_centers, real_binned, "b-", linewidth=2, label="Real (binned)")
    plt.plot(bin_centers, gen_binned, "r-", linewidth=2, label="Generated (binned)")

    # Add correlation text
    if not np.isnan(ld_corr):
        plt.text(
            0.95,
            0.95,
            f"LD Pattern Correlation: {ld_corr:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Plot formatting
    plt.title("Linkage Disequilibrium (LD) Decay", fontsize=14)
    plt.xlabel("Distance (SNPs)", fontsize=12)
    plt.ylabel("r²", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "ld_decay.png", dpi=300)
    plt.close()

    return ld_corr


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
    plt.figure(figsize=(12, 10))

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
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_components + 1), explained_var)
        plt.plot(range(1, n_components + 1), np.cumsum(explained_var), "r-o")

        plt.title("PCA Scree Plot", fontsize=14)
        plt.xlabel("Principal Component", fontsize=12)
        plt.ylabel("Explained Variance Ratio", fontsize=12)
        plt.xticks(range(1, n_components + 1))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pca_scree_plot.png", dpi=300)
        plt.close()

    return avg_w_distance, w_distances


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

    # Calculate observed heterozygosity (assuming 0.5 represents heterozygous)
    het_obs = np.mean(np.abs(samples - 0.5) < 0.1, axis=0)
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
