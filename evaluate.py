#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive evaluation script for diffusion models on genomic data.

This script evaluates diffusion models using metrics appropriate for generative models
in genomic data, including:
1. Minor Allele Frequency (MAF) correlation
2. Linkage Disequilibrium (LD) patterns
3. Principal Component Analysis (PCA) visualization
4. Wasserstein distance between distributions
5. Genetic diversity metrics

Example:
    python evaluate_diffusion.py --checkpoint path/to/checkpoint.ckpt

Outputs:
- Evaluation metrics in JSON format
- Visualization plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from diffusion.diffusion_model import DiffusionModel


def calculate_maf(samples):
    """Calculate Minor Allele Frequency (MAF) for each position.

    Args:
        samples: Tensor or array of SNP data [batch_size, seq_len] or [batch_size, channels, seq_len]

    Returns:
        numpy.ndarray: Array of MAF values
    """
    # Convert tensor to numpy array on CPU if needed
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


def plot_maf_distribution(real_maf, gen_maf, output_dir, bin_width=0.01):
    """Plot MAF distributions for real and generated data.

    Args:
        real_maf: MAF values for real data
        gen_maf: MAF values for generated data
        output_dir: Directory to save plots
        bin_width: Width of histogram bins
    """
    plt.figure(figsize=(12, 8))

    # Plot histograms
    bins = np.arange(0, 0.5 + bin_width, bin_width)
    plt.hist(real_maf, bins=bins, alpha=0.5, label="Real", density=True)
    plt.hist(gen_maf, bins=bins, alpha=0.5, label="Generated", density=True)

    # Calculate correlation
    maf_corr = np.corrcoef(real_maf, gen_maf)[0, 1]

    # Add statistics
    real_stats = calculate_maf_stats(real_maf)
    gen_stats = calculate_maf_stats(gen_maf)

    stats_text = (
        f"Statistics:\n"
        f"MAF Correlation: {maf_corr:.4f}\n\n"
        f"Real Data:\n"
        f'  Mean: {real_stats["mean"]:.4f}\n'
        f'  Median: {real_stats["median"]:.4f}\n'
        f'  Std: {real_stats["std"]:.4f}\n\n'
        f"Generated Data:\n"
        f'  Mean: {gen_stats["mean"]:.4f}\n'
        f'  Median: {gen_stats["median"]:.4f}\n'
        f'  Std: {gen_stats["std"]:.4f}'
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

    # Plot formatting
    plt.title("Minor Allele Frequency Distribution Comparison", fontsize=14)
    plt.xlabel("MAF", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "maf_distribution_comparison.png", dpi=300)
    plt.close()

    # Also create a scatter plot of MAF values
    plt.figure(figsize=(10, 10))
    plt.scatter(real_maf, gen_maf, alpha=0.5, s=10)
    plt.plot([0, 0.5], [0, 0.5], "r--")  # Diagonal line for perfect correlation

    plt.title(f"MAF Correlation (r = {maf_corr:.4f})", fontsize=14)
    plt.xlabel("Real MAF", fontsize=12)
    plt.ylabel("Generated MAF", fontsize=12)
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "maf_correlation.png", dpi=300)
    plt.close()

    return maf_corr


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
        float: Wasserstein distance between real and generated PCA distributions
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
    from scipy.stats import wasserstein_distance

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion model on genomic data"
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
        help="Number of samples to generate (default: match test set size)",
    )
    parser.add_argument(
        "--discretize",
        action="store_true",
        help="Discretize generated samples to 0, 0.5, and 1.0",
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

    output_dir = base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluation results will be saved to: {output_dir}")

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
        # Match to real sample size if not specified
        num_samples = args.num_samples or real_samples.shape[0]
        print(f"\nGenerating {num_samples} synthetic sequences...")
        gen_samples = model.generate_samples(
            num_samples=num_samples, discretize=args.discretize
        )

        # Check for NaN values in generated samples
        if torch.isnan(gen_samples).any():
            print(
                "Warning: Generated samples contain NaN values. Replacing with zeros..."
            )
            gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

        # Save samples
        torch.save(gen_samples, output_dir / "synthetic_sequences.pt")

        # Print statistics
        print(f"\nGenerated samples shape: {gen_samples.shape}")
        print(f"Generated samples unique values: {torch.unique(gen_samples)}")

        # Initialize results dictionary
        results = {}

        # 1. MAF Analysis
        print("\n1. Analyzing Minor Allele Frequency (MAF) distribution...")
        real_maf = calculate_maf(real_samples)
        gen_maf = calculate_maf(gen_samples)

        # Plot MAF distributions and get correlation
        maf_corr = plot_maf_distribution(real_maf, gen_maf, output_dir)

        # Store MAF statistics
        results["maf"] = {
            "real": calculate_maf_stats(real_maf),
            "generated": calculate_maf_stats(gen_maf),
            "correlation": float(maf_corr),
        }

        # 2. Linkage Disequilibrium Analysis
        print("\n2. Analyzing Linkage Disequilibrium (LD) patterns...")
        ld_corr = plot_ld_decay(real_samples, gen_samples, output_dir)

        results["ld"] = {
            "correlation": float(ld_corr) if not np.isnan(ld_corr) else None,
        }

        # 3. PCA Analysis
        print("\n3. Running Principal Component Analysis (PCA)...")
        w_distance, component_distances = run_pca_analysis(
            real_samples, gen_samples, output_dir, n_components=5
        )

        results["pca"] = {
            "wasserstein_distance": float(w_distance),
            "component_distances": [float(d) for d in component_distances],
        }

        # 4. Genetic Diversity Metrics
        print("\n4. Calculating genetic diversity metrics...")
        real_diversity = calculate_genetic_diversity(real_samples)
        gen_diversity = calculate_genetic_diversity(gen_samples)

        # Calculate relative differences
        diversity_diffs = {}
        for key in real_diversity:
            if real_diversity[key] != 0:
                rel_diff = (gen_diversity[key] - real_diversity[key]) / real_diversity[
                    key
                ]
                diversity_diffs[key] = float(rel_diff)
            else:
                diversity_diffs[key] = None

        results["genetic_diversity"] = {
            "real": real_diversity,
            "generated": gen_diversity,
            "relative_difference": diversity_diffs,
        }

        # Save all results to JSON
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        print(
            f"\nAll evaluation metrics saved to: {output_dir / 'evaluation_metrics.json'}"
        )

        # Print summary of key metrics
        print("\n=== EVALUATION SUMMARY ===")
        print(f"MAF Correlation: {maf_corr:.4f}")
        print(
            f"LD Pattern Correlation: {ld_corr:.4f}"
            if not np.isnan(ld_corr)
            else "LD Pattern Correlation: N/A"
        )
        print(f"PCA Distribution Distance: {w_distance:.4f}")
        print(
            f"Heterozygosity Preservation: {diversity_diffs['expected_heterozygosity']:.2%}"
        )
        print("==========================")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Evaluation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
