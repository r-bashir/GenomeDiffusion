#!/usr/bin/env python
# coding: utf-8

"""Script for generating and analyzing samples from trained SNP diffusion models.

This script uses the utility functions from inference_utils.py to:
1. Generate samples from diffusion models
2. Analyze MAF distributions
3. Create visualizations of real vs. generated data
4. Compute genomic metrics

Examples:
    # Generate samples and analyze
    python inference_new.py --checkpoint path/to/checkpoint.ckpt

    # Generate specific number of samples
    python inference_new.py --checkpoint path/to/checkpoint.ckpt --num_samples 100

    # Generate discretized samples (0, 0.5, 1.0)
    python inference_new.py --checkpoint path/to/checkpoint.ckpt --discretize
"""

import argparse
from pathlib import Path

import torch

from src.infer_utils import (
    compute_quality_metrics,
    generate_samples,
    get_encoding_params,
    sample_distribution,
    sample_statistics,
    sample_visualization,
    visualize_quality_metrics,
)
from src.utils import load_model_from_checkpoint, set_seed, setup_logging

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and analyze samples from SNP diffusion models"
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
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="infer")
    logger.info("Starting `inference.py` script.")

    # Set global seed
    set_seed(seed=42)

    # Load Model
    try:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully from checkpoint on %s", device)
        logger.info("Model config loaded from checkpoint:")
        print(f"\n{config}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    logger.info("Setting up output directory...")
    output_dir = Path(args.checkpoint).parent.parent / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory set to: %s", output_dir)

    # Load Dataset, gives shape of [B, L]
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Collect all test batches
    real_samples = []
    logger.info("Loading all test batches...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = batch.to(device)
            real_samples.append(batch)
    real_samples = torch.cat(real_samples, dim=0)

    # Select samples to use
    if args.num_samples is None:
        num_samples_to_use = len(real_samples)
        logger.info(f"Using all {num_samples_to_use} real samples from test dataset...")
    else:
        num_samples_to_use = args.num_samples
        if num_samples_to_use > len(real_samples):
            num_samples_to_use = len(real_samples)
        logger.info(f"Selecting {num_samples_to_use} real samples from test dataset...")

    # Select samples and ensure shape [num_samples, 1, L]
    real_samples = real_samples[:num_samples_to_use].unsqueeze(1)
    logger.info(f"Real sample shape: {real_samples.shape}")

    # Generate samples from model
    logger.info(f"Getting {num_samples_to_use} generated samples from model...")

    timestep = config["diffusion"]["timesteps"]
    logger.info(f"Starting at denoising from T={timestep}")

    with torch.no_grad():
        generated_samples = generate_samples(
            model,
            num_samples=num_samples_to_use,
            start_timestep=timestep,
            discretize=args.discretize,
        )
    logger.info(f"Generated sample shape: {generated_samples.shape}")

    # Verify shapes match
    if real_samples.shape != generated_samples.shape:
        logger.error(
            f"Shape mismatch! Real: {real_samples.shape}, Generated: {generated_samples.shape}"
        )
        raise ValueError(
            "Real and generated samples must have the same shape for comparison"
        )

    logger.info(
        f"✅ Sample shapes match: {real_samples.shape} (Real) == {generated_samples.shape} (Generated)"
    )

    # Determine encoding from data (scaled vs unscaled)
    # If max value <= 0.5, we assume scaled encoding [0.0, 0.25, 0.5]
    try:
        max_val = float(torch.max(real_samples).item())
    except Exception:
        max_val = 0.5
    scaled = max_val <= 0.5 + 1e-6
    enc = get_encoding_params(scaled)
    genotype_values = enc["genotype_values"]
    max_value = enc["max_value"]

    # === Basic Sample Analysis ===
    print(f"\n📊 BASIC SAMPLE ANALYSIS")
    print("=" * 40)

    sample_statistics(
        real_samples, "Real Samples", unique_values=False, genotype_counts=True
    )
    sample_statistics(
        generated_samples,
        "Generated Samples",
        unique_values=False,
        genotype_counts=True,
    )

    # === Sample Distribution Analysis ===
    print(f"\n📊 SAMPLE DISTRIBUTION ANALYSIS")
    print("=" * 40)

    sample_distribution(
        real_samples,
        generated_samples,
        output_dir / "sample_distribution.png",
        genotype_values=genotype_values,
    )
    logger.info(
        f"✅ Sample distributions (sample_distribution.png) saved with genotype_values={genotype_values}"
    )

    sample_visualization(
        real_samples,
        generated_samples,
        output_dir / "sample_visualization.png",
        genotype_values=genotype_values,
        max_seq_len=100,
    )
    logger.info("✅ Sample visualizations (sample_visualization.png) are saved")

    # Save real and generated samples for further evaluation
    torch.save(real_samples, output_dir / "real_samples.pt")
    logger.info("✅ Real samples (real_samples.pt) are saved")
    torch.save(generated_samples, output_dir / "generated_samples.pt")
    logger.info("✅ Generated samples (generated_samples.pt) are saved")

    # === Basic Quality Assessment ===
    print(f"\n📊 QUICK QUALITY METRICS")
    logger.info("Computing quality metrics...")
    metrics = compute_quality_metrics(
        real_samples,
        generated_samples,
        max_value=max_value,
        genotype_values=genotype_values,
    )
    quality_score = metrics["overall_score"]

    # Create visual metrics plot
    logger.info("Visualizing quality metrics...")
    visualize_quality_metrics(
        real_samples,
        generated_samples,
        output_dir / "quick_metrics.png",
        max_value=max_value,
        genotype_values=genotype_values,
    )

    print(f"\n🎯 QUICK QUALITY ASSESSMENT SUMMARY")
    print("=" * 40)
    print(f"Overall Quality Score: {quality_score:.3f}/1.000")

    logger.info(
        f"Detected encoding: {'scaled' if scaled else 'unscaled'} (max_value={max_value})"
    )
    logger.info(f"Genotype values: {genotype_values}")
    logger.info(
        f"Key metrics - AF corr: {metrics['af_corr']:.3f}, MAF corr: {metrics['maf_corr']:.3f}, Wasserstein: {metrics['wasserstein_dist']:.4f}"
    )

    if quality_score >= 0.8:
        status = "🟢 EXCELLENT/GOOD"
        recommendation = "Samples look very promising!"
    elif quality_score >= 0.6:
        status = "🟡 FAIR"
        recommendation = "Samples show reasonable quality."
    else:
        status = "🔴 POOR"
        recommendation = "Samples may need improvement."

    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    print(f"📊 Visual metrics summary (quick_metrics.png) saved.\n")
    print(f"To run comprehensive analysis, run:\n")
    print(f"python sample_analysis.py --checkpoint {args.checkpoint}\n")
    logger.info("Inference completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
