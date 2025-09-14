#!/usr/bin/env python
# coding: utf-8

"""Comprehensive sample analysis script for analyzing inference results from GenomeDiffusion.

This script performs comprehensive analysis of generated samples using checkpoint paths
to automatically derive inference and analysis directories, providing detailed quantitative
metrics and visualizations with advanced diagnostic capabilities.

WORKFLOW:
    1. First, run inference.py to generate samples:
       python inference.py --checkpoint path/to/model.ckpt --num_samples 64

    2. Then, run this analysis script using the same checkpoint:
       python sample_analysis.py --checkpoint path/to/model.ckpt

Usage Examples:
    # Analyze inference results (typical workflow)
    python sample_analysis.py --checkpoint outputs/lightning_logs/StairUnetDiff/uu8onswd/checkpoints/epoch=99-step=1000.ckpt
    # → Analysis results saved to outputs/lightning_logs/StairUnetDiff/uu8onswd/analysis/

    # Generate comprehensive report with diagnostic insights
    python sample_analysis.py --checkpoint path/to/model.ckpt --create_report
    # → Comprehensive diagnostic evaluation with mode collapse and LD analysis

    # Analyze with custom LD settings
    python sample_analysis.py --checkpoint path/to/model.ckpt --max_distance 50 --n_pairs 500
"""

import argparse
import json
from pathlib import Path

import torch

from src.analysis_utils import (
    compute_dimensionality_metrics,
    create_evaluation_report,
    plot_genomic_position_effects,
    plot_haplotype_blocks,
    plot_hardy_weinberg_deviation,
    plot_sample_clustering,
    run_comprehensive_evaluation,
)
from src.utils import set_seed, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of genomic samples from diffusion model checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (will auto-derive inference and analysis directories)",
    )
    parser.add_argument(
        "--max_distance",
        type=int,
        default=100,
        help="Maximum distance for LD analysis",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=1000,
        help="Number of SNP pairs to sample for LD analysis",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of PCA components",
    )
    parser.add_argument(
        "--max_value",
        type=float,
        default=0.5,
        help="Maximum value for MAF spectrum analysis",
    )
    parser.add_argument(
        "--create_report",
        action="store_true",
        help="Create markdown evaluation report",
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


def derive_directories(checkpoint_path):
    """Derive inference and analysis directories from checkpoint path.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        tuple: (inference_dir, analysis_dir)
    """
    checkpoint_path = Path(checkpoint_path)

    # Get the run directory by going up from checkpoints/ to the parent
    # e.g., outputs/lightning_logs/StairUnetDiff/uu8onswd/checkpoints/model.ckpt
    # -> outputs/lightning_logs/StairUnetDiff/uu8onswd
    if checkpoint_path.parent.name == "checkpoints":
        run_dir = checkpoint_path.parent.parent
    else:
        # If checkpoint is not in checkpoints/ directory, use parent directly
        run_dir = checkpoint_path.parent

    # Derive inference directory
    inference_dir = run_dir / "inference"

    # Derive analysis directory
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    return inference_dir, analysis_dir


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
    logger.info("🎯 EVALUATION COMPLETE!")
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
    logger.info(f"Using checkpoint: {args.checkpoint}")

    # 1. Derive directories from checkpoint path
    inference_dir, analysis_dir = derive_directories(args.checkpoint)
    logger.info(f"Inference directory: {inference_dir}")
    logger.info(f"Analysis results will be saved to: {analysis_dir}")

    # 2. Validate inference directory exists
    if not inference_dir.exists():
        raise FileNotFoundError(
            f"Inference directory not found: {inference_dir}\n"
            f"Please run inference.py first with the same checkpoint: {args.checkpoint}"
        )

    # 3. Load samples
    real_samples, generated_samples = load_samples(inference_dir)

    # 4. Run comprehensive diagnostic evaluation
    logger.info("📊 Running comprehensive diagnostic evaluation...")

    comprehensive_results = run_comprehensive_evaluation(
        real_samples, generated_samples, analysis_dir
    )

    # Extract key metrics for backward compatibility
    metrics = {
        "comprehensive_evaluation": comprehensive_results,
        "overall_quality_score": comprehensive_results["overall_score"],
        "ld_correlation": comprehensive_results["ld_correlation"],
        "pca_distance": comprehensive_results["pca_distance"],
        "genetic_diversity_real": comprehensive_results["genetic_diversity"]["real"],
        "genetic_diversity_generated": comprehensive_results["genetic_diversity"][
            "generated"
        ],
        "maf_correlation": comprehensive_results["maf_correlation"],
        "mode_collapse_detected": comprehensive_results["mode_collapse"][
            "mode_collapse_detected"
        ],
        "sample_diversity_ratio": comprehensive_results["mode_collapse"][
            "diversity_ratio"
        ],
        "status": comprehensive_results["status"],
    }

    # Run additional detailed plots for completeness
    logger.info("🧬 Creating additional detailed plots...")

    # Haplotype blocks
    haplotype_similarity = plot_haplotype_blocks(
        real_samples, generated_samples, analysis_dir
    )
    metrics["haplotype_similarity"] = haplotype_similarity

    # Hardy-Weinberg deviations
    hwe_correlation = plot_hardy_weinberg_deviation(
        real_samples, generated_samples, analysis_dir
    )
    metrics["hwe_correlation"] = hwe_correlation

    # Genomic position effects
    position_correlation = plot_genomic_position_effects(
        real_samples, generated_samples, analysis_dir
    )
    metrics["position_correlation"] = position_correlation

    # Sample clustering
    clustering_score = plot_sample_clustering(
        real_samples, generated_samples, analysis_dir
    )
    metrics["clustering_score"] = clustering_score

    # Dimensionality metrics
    dim_metrics = compute_dimensionality_metrics(real_samples, generated_samples)
    metrics["dimensionality"] = dim_metrics

    # 5. Print analysis summary
    logger.info("✅ All analyses complete")
    print("\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print(f"🎯 Overall Quality Score: {metrics['overall_quality_score']:.3f}")
    print(f"🧬 LD Correlation: {metrics['ld_correlation']:.6f}")
    print(f"📈 PCA Distance: {metrics['pca_distance']:.4f}")
    print(f"🧪 Genetic Diversity (Real): {metrics['genetic_diversity_real']:.6f}")
    print(
        f"🧪 Genetic Diversity (Generated): {metrics['genetic_diversity_generated']:.6f}"
    )
    print(f"🔬 MAF Correlation: {metrics['maf_correlation']:.6f}")
    print(f"🧬 Haplotype Block Similarity: {metrics['haplotype_similarity']:.3f}")
    print(f"🧪 HWE Deviation Correlation: {metrics['hwe_correlation']:.3f}")
    print(f"📍 Position Effects Correlation: {metrics['position_correlation']:.3f}")
    print(
        f"🔗 Sample Clustering Score: {metrics['clustering_score']:.3f} (lower is better)"
    )
    print(f"🚨 Mode Collapse Detected: {metrics['mode_collapse_detected']}")
    print(f"📊 Sample Diversity Ratio: {metrics['sample_diversity_ratio']:.3f}")
    print(f"📋 Status: {metrics['status']}")

    # 6. Save results to JSON
    save_results(metrics, analysis_dir)

    # 7. Create markdown report (optional)
    if args.create_report:
        logger.info("📄 Creating evaluation report...")
        create_evaluation_report(metrics, real_samples, generated_samples, analysis_dir)
        logger.info("✅ Report created")

    # 8. Print final summary
    print_final_summary(metrics, args.checkpoint, analysis_dir)


if __name__ == "__main__":
    main()
