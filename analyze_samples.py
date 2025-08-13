#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive sample analysis script for analyzing inference results from GenomeDiffusion.

This script performs comprehensive analysis of generated samples using the output
directory from inference.py, providing detailed quantitative metrics and visualizations.

WORKFLOW:
    1. First, run inference.py to generate samples and save to output directory:
       python inference.py --checkpoint path/to/model.ckpt --num_samples 64

    2. Then, run this sample analysis script on the inference directory:
       python analyze_samples.py --input_dir path/to/inference/results

Usage Examples:
    # Analyze inference results (typical workflow)
    python analyze_samples.py --input_dir outputs/inference
    # → Analysis results saved to outputs/analysis/

    # Analyze with custom settings
    python analyze_samples.py --input_dir outputs/inference --max_value 1.0
    # → Analysis results saved to outputs/analysis/

    # Generate comprehensive report
    python analyze_samples.py --input_dir outputs/inference --create_report
    # → Analysis results and report saved to outputs/analysis/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analyze_sample_utils import (
    calculate_genetic_diversity,
    compute_dimensionality_metrics,
    create_evaluation_report,
    plot_genomic_position_effects,
    plot_haplotype_blocks,
    plot_hardy_weinberg_deviation,
    plot_ld_decay,
    plot_maf_spectrum,
    plot_sample_clustering,
    run_pca_analysis,
)
from src.infer_utils import compute_quality_metrics
from src.utils import set_seed, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of GenomeDiffusion inference results"
    )

    # Input directory (from inference.py output)
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory from inference.py containing real_samples.pt and generated_samples.pt",
    )

    # Evaluation settings
    parser.add_argument(
        "--max_value",
        type=float,
        default=0.5,
        help="Maximum value for MAF calculations (should match inference settings)",
    )

    parser.add_argument(
        "--create_report",
        action="store_true",
        help="Create comprehensive markdown report",
    )

    return parser.parse_args()


def load_samples(input_dir):
    """Load real and generated samples from input directory."""
    logger = setup_logging(name="analyze_samples")

    # Check for required sample files
    real_samples_file = input_dir / "real_samples.pt"
    generated_samples_file = input_dir / "generated_samples.pt"

    if not real_samples_file.exists():
        raise FileNotFoundError(f"Real samples not found: {real_samples_file}")
    if not generated_samples_file.exists():
        raise FileNotFoundError(
            f"Generated samples not found: {generated_samples_file}"
        )

    # Load samples with error handling
    logger.info(f"Loading samples from: {input_dir}")
    try:
        real_samples = torch.load(real_samples_file, weights_only=False)
        generated_samples = torch.load(generated_samples_file, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load samples: {e}")

    # Validate shapes
    logger.info(f"Real samples shape: {real_samples.shape}")
    logger.info(f"Generated samples shape: {generated_samples.shape}")

    if real_samples.shape != generated_samples.shape:
        raise ValueError(
            f"Sample shape mismatch! Real: {real_samples.shape}, Generated: {generated_samples.shape}"
        )

    logger.info("✅ Samples loaded and validated")
    return real_samples, generated_samples


def create_output_dir(input_dir):
    """Create analysis output directory at same level as input directory."""
    output_dir = input_dir.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_results(metrics, output_dir):
    """Save comprehensive metrics to JSON file."""
    logger = setup_logging(name="analyze_samples")

    results_file = output_dir / "comprehensive_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"📊 Results saved to: {results_file}")
    return results_file


def print_final_summary(metrics, input_dir, output_dir):
    """Print final evaluation summary with interpretation."""
    logger = setup_logging(name="analyze_samples")

    overall_score = metrics["overall_quality_score"]

    # Quality interpretation
    if overall_score >= 0.9:
        interpretation = "🟢 EXCELLENT"
    elif overall_score >= 0.8:
        interpretation = "🟡 GOOD"
    elif overall_score >= 0.7:
        interpretation = "🟠 FAIR"
    else:
        interpretation = "🔴 POOR"

    # Final summary
    logger.info(f"🎯 EVALUATION COMPLETE!")
    logger.info(f"Overall Quality Score: {overall_score:.3f}/1.000")
    logger.info(f"Interpretation: {interpretation}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")


def main():
    """Main analysis workflow."""
    # Setup
    args = parse_args()
    logger = setup_logging(name="analyze_samples")
    set_seed(seed=42)

    logger.info("Starting comprehensive genomic sample analysis...")

    # 1. Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # 2. Create output directory
    output_dir = create_output_dir(input_dir)
    logger.info(f"Analysis results will be saved to: {output_dir}")

    # 3. Load samples
    real_samples, generated_samples = load_samples(input_dir)

    # 4. Run individual analysis functions
    logger.info("\n📊 Computing evaluation metrics...")

    # Basic quality metrics (from infer_utils.py)
    basic_score = compute_quality_metrics(
        real_samples, generated_samples, args.max_value
    )

    # Linkage Disequilibrium analysis
    logger.info("🧬 Analyzing Linkage Disequilibrium...")
    ld_correlation = plot_ld_decay(real_samples, generated_samples, output_dir)

    # PCA analysis
    logger.info("📈 Running PCA analysis...")
    pca_distance, _ = run_pca_analysis(real_samples, generated_samples, output_dir)

    # Genetic diversity
    logger.info("🧪 Computing genetic diversity...")
    diversity_real = calculate_genetic_diversity(real_samples)
    diversity_gen = calculate_genetic_diversity(generated_samples)

    # Dimensionality metrics
    logger.info("📐 Computing dimensionality metrics...")
    dim_metrics = compute_dimensionality_metrics(real_samples, generated_samples)

    # Advanced genomic analysis plots
    logger.info("🧬 Creating haplotype block analysis...")
    haplotype_similarity = plot_haplotype_blocks(
        real_samples, generated_samples, output_dir
    )

    logger.info("🔬 Analyzing MAF spectrum...")
    maf_correlation = plot_maf_spectrum(
        real_samples, generated_samples, output_dir, args.max_value
    )

    logger.info("🧪 Computing Hardy-Weinberg deviations...")
    hwe_correlation = plot_hardy_weinberg_deviation(
        real_samples, generated_samples, output_dir
    )

    logger.info("📍 Analyzing genomic position effects...")
    position_correlation = plot_genomic_position_effects(
        real_samples, generated_samples, output_dir
    )

    logger.info("🔗 Creating sample clustering analysis...")
    clustering_score = plot_sample_clustering(
        real_samples, generated_samples, output_dir
    )

    # Combine all metrics
    metrics = {
        "basic_quality_score": basic_score,
        "ld_correlation": ld_correlation,
        "pca_distance": pca_distance,
        "genetic_diversity_real": diversity_real,
        "genetic_diversity_generated": diversity_gen,
        "dimensionality": dim_metrics,
        "haplotype_similarity": haplotype_similarity,
        "maf_correlation": maf_correlation,
        "hwe_correlation": hwe_correlation,
        "position_correlation": position_correlation,
        "clustering_score": clustering_score,
        "overall_quality_score": basic_score,
    }

    # 5. Print analysis summary
    logger.info("✅ All analyses complete")
    print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print(f"🎯 Basic Quality Score: {basic_score:.3f}")
    print(f"🧬 LD Correlation: {ld_correlation:.3f}")
    print(f"📈 PCA Distance: {pca_distance:.4f}")
    print(f"🧪 Genetic Diversity (Real): {diversity_real['nucleotide_diversity']:.4f}")
    print(
        f"🧪 Genetic Diversity (Generated): {diversity_gen['nucleotide_diversity']:.4f}"
    )
    print(f"🧬 Haplotype Block Similarity: {haplotype_similarity:.3f}")
    print(f"🔬 MAF Correlation: {maf_correlation:.3f}")
    print(f"🧪 HWE Deviation Correlation: {hwe_correlation:.3f}")
    print(f"📍 Position Effects Correlation: {position_correlation:.3f}")
    print(f"🔗 Sample Clustering Score: {clustering_score:.3f} (lower is better)")

    # 6. Save results to JSON
    save_results(metrics, output_dir)

    # 7. Create markdown report (optional)
    if args.create_report:
        logger.info("📄 Creating evaluation report...")
        create_evaluation_report(metrics, real_samples, generated_samples, output_dir)
        logger.info("✅ Report created")

    # 8. Print final summary
    print_final_summary(metrics, input_dir, output_dir)


if __name__ == "__main__":
    main()
