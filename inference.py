#!/usr/bin/env python
# coding: utf-8

"""Script to perform inference on SNP diffusion models.

Examples:
    # Run inference on test dataset
    python inference.py --checkpoint path/to/checkpoint.ckpt

    # Run inference on test dataset with specific number of samples
    python inference.py --checkpoint path/to/checkpoint.ckpt --num_samples 100
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
    parser = argparse.ArgumentParser(description="Inference")
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
    parser.add_argument(
        "--test_imputation",
        action="store_true",
        help="Test imputation functionality using real samples as ground truth",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.3,
        help="Ratio of SNPs to keep as known for imputation testing (0.0-1.0)",
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

    # Collect single/all test batches
    logger.info("Collecting batches...")
    with torch.no_grad():
        # real_samples = torch.cat([batch.to(device) for batch in test_loader], dim=0)
        real_samples = next(iter(test_loader)).to(device)

    logger.info(f"Sample shape: {real_samples.shape}, and dim: {real_samples.dim()}")

    # Select number of samples
    num_samples = (
        len(real_samples)
        if args.num_samples is None
        else min(args.num_samples, len(real_samples))
    )

    logger.info(f"Selecting {num_samples} for inference...")

    # Add channel dimension: [B, L] to [B, C, L]
    logger.info("Add channel dimension to the shape [B, L] to [B, C, L]")
    real_samples = real_samples[:num_samples].unsqueeze(1)
    logger.info(f"Sample shape: {real_samples.shape}, and dim: {real_samples.dim()}")

    logger.info("Starting inference...")
    timestep = config["diffusion"]["timesteps"]
    logger.info(f"Starting denoising from T={timestep}")

    # Test imputation if requested
    if args.test_imputation:
        logger.info("\n=== TESTING IMPUTATION FUNCTIONALITY ===")
        logger.info(f"Using mask ratio: {args.mask_ratio} (fraction of SNPs known)")

        # Create random masks for imputation testing
        torch.manual_seed(42)  # For reproducible results
        mask = torch.rand_like(real_samples) < args.mask_ratio
        mask = mask.float()

        known_snps_per_sample = mask.sum(dim=-1).mean().item()
        total_snps = mask.shape[-1]
        logger.info(
            f"Average known SNPs per sample: {known_snps_per_sample:.0f}/{total_snps} ({known_snps_per_sample/total_snps:.1%})"
        )

        # Generate samples with imputation using real samples as ground truth
        with torch.no_grad():
            logger.info("Generating samples WITH imputation...")
            generated_samples = generate_samples(
                model,
                num_samples=num_samples,
                start_timestep=timestep,
                discretize=args.discretize,
                true_x0=real_samples,
                mask=mask,
            )

        # Analyze imputation accuracy
        logger.info("\n=== IMPUTATION ACCURACY ANALYSIS ===")

        # Check accuracy at known positions
        known_positions = mask == 1.0
        if known_positions.any():
            imputation_diff = torch.abs(
                generated_samples[known_positions] - real_samples[known_positions]
            )
            max_diff = imputation_diff.max().item()
            mean_diff = imputation_diff.mean().item()
            std_diff = imputation_diff.std().item()

            logger.info("Imputation accuracy at known positions:")
            logger.info(f"  Max difference: {max_diff:.8f}")
            logger.info(f"  Mean difference: {mean_diff:.8f}")
            logger.info(f"  Std difference: {std_diff:.8f}")

            if mean_diff < 1e-6:
                logger.info(
                    "✅ Perfect imputation: Known positions exactly match ground truth"
                )
            elif mean_diff < 1e-4:
                logger.info(
                    "✅ Excellent imputation: Very small differences at known positions"
                )
            else:
                logger.warning(
                    f"⚠️  Imputation accuracy: Mean difference {mean_diff:.8f}"
                )

        # Compare generation quality at unknown vs known positions
        unknown_positions = mask == 0.0
        if unknown_positions.any() and known_positions.any():
            # MSE comparison
            mse_unknown = torch.mean(
                (generated_samples[unknown_positions] - real_samples[unknown_positions])
                ** 2
            ).item()
            mse_known = torch.mean(
                (generated_samples[known_positions] - real_samples[known_positions])
                ** 2
            ).item()

            # Value distribution comparison
            unknown_mean = generated_samples[unknown_positions].mean().item()
            known_mean = generated_samples[known_positions].mean().item()
            real_unknown_mean = real_samples[unknown_positions].mean().item()
            real_known_mean = real_samples[known_positions].mean().item()

            logger.info("\nGeneration quality comparison:")
            logger.info(f"  MSE at unknown positions: {mse_unknown:.6f}")
            logger.info(f"  MSE at known positions: {mse_known:.6f}")
            if mse_known > 0:
                logger.info(
                    f"  Quality ratio: {mse_unknown/mse_known:.2f}x (unknown/known MSE)"
                )

            logger.info("\nValue distribution comparison:")
            logger.info(f"  Generated mean at unknown positions: {unknown_mean:.4f}")
            logger.info(f"  Generated mean at known positions: {known_mean:.4f}")
            logger.info(f"  Real mean at unknown positions: {real_unknown_mean:.4f}")
            logger.info(f"  Real mean at known positions: {real_known_mean:.4f}")

        # Also generate samples without imputation for comparison
        logger.info("\nGenerating samples WITHOUT imputation for comparison...")
        with torch.no_grad():
            generated_samples_no_imputation = generate_samples(
                model,
                num_samples=num_samples,
                start_timestep=timestep,
                discretize=args.discretize,
                true_x0=None,
                mask=None,
            )

        # Compare imputation vs no-imputation results
        diff_with_without = (
            torch.abs(generated_samples - generated_samples_no_imputation).mean().item()
        )
        logger.info("\nImputation vs No-imputation comparison:")
        logger.info(f"  Mean absolute difference: {diff_with_without:.6f}")

        # Save imputation-specific results
        torch.save(mask, output_dir / "imputation_mask.pt")
        torch.save(
            generated_samples_no_imputation,
            output_dir / "generated_samples_no_imputation.pt",
        )
        logger.info("✅ Imputation mask and no-imputation samples saved")

    else:
        # Standard generation without imputation
        with torch.no_grad():
            generated_samples = generate_samples(
                model,
                num_samples=num_samples,
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
    print("\n📊 BASIC SAMPLE ANALYSIS")
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
    print("\n📊 SAMPLE DISTRIBUTION ANALYSIS")
    print("=" * 40)

    sample_distribution(
        real_samples,
        generated_samples,
        output_dir,
        genotype_values=genotype_values,
    )
    logger.info(
        f"✅ Sample distributions (sample_distribution.png) saved with genotype_values={genotype_values}"
    )

    sample_visualization(
        real_samples,
        generated_samples,
        output_dir,
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
    print("\n📊 QUALITY METRICS")
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
        output_dir,
        max_value=max_value,
        genotype_values=genotype_values,
    )

    print("\n🎯 QUALITY ASSESSMENT SUMMARY")
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
    print("📊 Visual metrics summary (quality_metrics.png) saved.\n")
    if args.test_imputation:
        print("\n🧬 IMPUTATION TEST SUMMARY")
        print("=" * 40)
        print("Imputation functionality has been tested successfully!")
        print(f"Known SNPs ratio: {args.mask_ratio:.1%}")
        print("Check the logs above for detailed imputation accuracy metrics.")
        print("Additional files saved:")
        print("  - imputation_mask.pt: The mask used for testing")
        print("  - generated_samples_no_imputation.pt: Samples without imputation")
        print("\nTo run comprehensive analysis, run:")
    else:
        print("To run comprehensive analysis, run:\n")
        print(f"python sample_analysis.py --checkpoint {args.checkpoint}\n")

    if args.test_imputation:
        logger.info("Inference with imputation testing completed!")
        logger.info("Imputation functionality verified successfully.")
    else:
        logger.info("Inference completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
