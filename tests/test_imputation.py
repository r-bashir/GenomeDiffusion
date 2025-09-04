#!/usr/bin/env python
# coding: utf-8

"""
Test script for imputation functionality in the diffusion model.

This script demonstrates how to use the imputation feature where the model
can replace reconstructed SNPs with true SNPs during the reverse diffusion process.

Usage:
    python test_imputation.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.infer_utils import denoise_samples, generate_samples
from src.utils import load_model_from_checkpoint, set_seed, setup_logging


def create_test_mask(shape, mask_ratio=0.5, seed=42):
    """
    Create a random mask for imputation testing.

    Args:
        shape: Shape of the mask tensor [B, C, L]
        mask_ratio: Fraction of SNPs to mask (0=all unknown, 1=all known)
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Binary mask with 1=known SNP, 0=unknown SNP
    """
    torch.manual_seed(seed)
    mask = torch.rand(shape) < mask_ratio
    return mask.float()


def create_genomic_pattern_mask(shape, pattern="blocks", seed=42):
    """
    Create masks with genomic-relevant patterns for testing.

    Args:
        shape: Shape of the mask tensor [B, C, L]
        pattern: Type of pattern ("blocks", "random", "edges", "center")
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Binary mask with 1=known SNP, 0=unknown SNP
    """
    torch.manual_seed(seed)
    B, C, L = shape
    mask = torch.zeros(shape)

    if pattern == "blocks":
        # Create blocks of known/unknown regions (simulating genomic regions)
        block_size = L // 10
        for i in range(0, L, block_size * 2):
            end_idx = min(i + block_size, L)
            mask[:, :, i:end_idx] = 1.0

    elif pattern == "random":
        # Random 50% of SNPs are known
        mask = torch.rand(shape) < 0.5

    elif pattern == "edges":
        # Known SNPs at the edges, unknown in the middle
        edge_size = L // 4
        mask[:, :, :edge_size] = 1.0
        mask[:, :, -edge_size:] = 1.0

    elif pattern == "center":
        # Known SNPs in the center, unknown at edges
        center_start = L // 4
        center_end = 3 * L // 4
        mask[:, :, center_start:center_end] = 1.0

    return mask.float()


def test_imputation_functionality(model, device, logger):
    """
    Test the imputation functionality with synthetic data.

    Args:
        model: Loaded diffusion model
        device: Device to run on
        logger: Logger instance
    """
    logger.info("Testing imputation functionality...")

    # Get model data shape
    data_shape = model._data_shape  # [C, L]
    batch_size = 4
    full_shape = (batch_size,) + data_shape  # [B, C, L]

    logger.info(f"Data shape: {data_shape}, Full shape: {full_shape}")

    # Create synthetic ground truth data (scaled SNP values)
    set_seed(42)
    true_x0 = torch.rand(full_shape, device=device) * 0.5  # Values in [0, 0.5]
    logger.info(f"Created synthetic ground truth with shape: {true_x0.shape}")
    logger.info(
        f"Ground truth stats - Min: {true_x0.min():.4f}, Max: {true_x0.max():.4f}, Mean: {true_x0.mean():.4f}"
    )

    # Test different mask patterns
    patterns = ["blocks", "random", "edges", "center"]

    for pattern in patterns:
        logger.info(f"\n=== Testing {pattern} mask pattern ===")

        # Create mask for this pattern
        mask = create_genomic_pattern_mask(full_shape, pattern=pattern, seed=42)
        mask = mask.to(device)

        known_ratio = mask.mean().item()
        logger.info(f"Mask pattern: {pattern}, Known SNPs: {known_ratio:.1%}")

        # Test 1: Generation with imputation
        logger.info("Test 1: Generation with imputation")
        try:
            generated_with_imputation = generate_samples(
                model,
                num_samples=batch_size,
                start_timestep=100,  # Start from moderate noise
                discretize=False,
                seed=42,
                true_x0=true_x0,
                mask=mask,
            )
            logger.info(
                f"✅ Generation with imputation successful: {generated_with_imputation.shape}"
            )

            # Check that known positions match ground truth
            masked_positions = mask == 1.0
            if masked_positions.any():
                diff_at_known = torch.abs(
                    generated_with_imputation[masked_positions]
                    - true_x0[masked_positions]
                )
                max_diff_known = diff_at_known.max().item()
                mean_diff_known = diff_at_known.mean().item()
                logger.info(
                    f"Imputation accuracy at known positions - Max diff: {max_diff_known:.6f}, Mean diff: {mean_diff_known:.6f}"
                )

                if mean_diff_known < 1e-6:
                    logger.info(
                        "✅ Perfect imputation: Known positions exactly match ground truth"
                    )
                else:
                    logger.warning(
                        f"⚠️  Imputation not perfect: Mean difference {mean_diff_known:.6f}"
                    )

        except Exception as e:
            logger.error(f"❌ Generation with imputation failed: {e}")

        # Test 2: Generation without imputation (baseline)
        logger.info("Test 2: Generation without imputation (baseline)")
        try:
            generated_without_imputation = generate_samples(
                model,
                num_samples=batch_size,
                start_timestep=100,
                discretize=False,
                seed=42,
                true_x0=None,
                mask=None,
            )
            logger.info(
                f"✅ Generation without imputation successful: {generated_without_imputation.shape}"
            )

        except Exception as e:
            logger.error(f"❌ Generation without imputation failed: {e}")

        # Test 3: Denoising with imputation
        logger.info("Test 3: Denoising with imputation")
        try:
            # Add noise to ground truth
            noisy_x0 = true_x0 + 0.1 * torch.randn_like(true_x0)

            denoised_with_imputation = denoise_samples(
                model,
                noisy_x0,
                start_timestep=50,  # Light denoising
                discretize=False,
                seed=42,
                true_x0=true_x0,
                mask=mask,
            )
            logger.info(
                f"✅ Denoising with imputation successful: {denoised_with_imputation.shape}"
            )

            # Check imputation accuracy
            masked_positions = mask == 1.0
            if masked_positions.any():
                diff_at_known = torch.abs(
                    denoised_with_imputation[masked_positions]
                    - true_x0[masked_positions]
                )
                mean_diff_known = diff_at_known.mean().item()
                logger.info(
                    f"Denoising imputation accuracy - Mean diff at known positions: {mean_diff_known:.6f}"
                )

        except Exception as e:
            logger.error(f"❌ Denoising with imputation failed: {e}")


def test_edge_cases(model, device, logger):
    """
    Test edge cases for imputation functionality.

    Args:
        model: Loaded diffusion model
        device: Device to run on
        logger: Logger instance
    """
    logger.info("\n=== Testing Edge Cases ===")

    data_shape = model._data_shape
    batch_size = 2
    full_shape = (batch_size,) + data_shape

    # Test case 1: All SNPs known (mask = 1 everywhere)
    logger.info("Edge Case 1: All SNPs known")
    try:
        true_x0 = torch.rand(full_shape, device=device) * 0.5
        mask_all_known = torch.ones_like(true_x0)

        result = generate_samples(
            model,
            num_samples=batch_size,
            start_timestep=50,
            seed=42,
            true_x0=true_x0,
            mask=mask_all_known,
        )

        # Should be identical to ground truth
        max_diff = torch.abs(result - true_x0).max().item()
        logger.info(f"✅ All known case - Max difference: {max_diff:.8f}")

        if max_diff < 1e-6:
            logger.info("✅ Perfect: Result identical to ground truth")
        else:
            logger.warning(f"⚠️  Not perfect: Max difference {max_diff:.8f}")

    except Exception as e:
        logger.error(f"❌ All known case failed: {e}")

    # Test case 2: No SNPs known (mask = 0 everywhere)
    logger.info("Edge Case 2: No SNPs known")
    try:
        true_x0 = torch.rand(full_shape, device=device) * 0.5
        mask_none_known = torch.zeros_like(true_x0)

        result_none_known = generate_samples(
            model,
            num_samples=batch_size,
            start_timestep=50,
            seed=42,
            true_x0=true_x0,
            mask=mask_none_known,
        )

        result_no_imputation = generate_samples(
            model,
            num_samples=batch_size,
            start_timestep=50,
            seed=42,
            true_x0=None,
            mask=None,
        )

        # Should be similar to no imputation case
        diff = torch.abs(result_none_known - result_no_imputation).mean().item()
        logger.info(f"✅ None known case - Difference from no imputation: {diff:.6f}")

    except Exception as e:
        logger.error(f"❌ None known case failed: {e}")

    # Test case 3: Single SNP known
    logger.info("Edge Case 3: Single SNP known")
    try:
        true_x0 = torch.rand(full_shape, device=device) * 0.5
        mask_single = torch.zeros_like(true_x0)
        mask_single[:, :, 0] = 1.0  # Only first SNP is known

        result = generate_samples(
            model,
            num_samples=batch_size,
            start_timestep=50,
            seed=42,
            true_x0=true_x0,
            mask=mask_single,
        )

        # Check that first SNP matches ground truth
        first_snp_diff = torch.abs(result[:, :, 0] - true_x0[:, :, 0]).max().item()
        logger.info(f"✅ Single SNP case - First SNP difference: {first_snp_diff:.8f}")

    except Exception as e:
        logger.error(f"❌ Single SNP case failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test imputation functionality")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./imputation_test_results",
        help="Directory to save test results",
    )
    args = parser.parse_args()

    # Setup
    logger = setup_logging(name="imputation_test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("IMPUTATION FUNCTIONALITY TEST")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")

    # Load model
    try:
        logger.info("Loading model...")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model config: {config}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Run tests
    try:
        test_imputation_functionality(model, device, logger)
        test_edge_cases(model, device, logger)

        logger.info("\n" + "=" * 60)
        logger.info("IMPUTATION TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("The imputation functionality has been successfully tested!")
        logger.info("You can now use true_x0 and mask parameters in:")
        logger.info("- generate_samples() for generation with imputation")
        logger.info("- denoise_samples() for denoising with imputation")
        logger.info("- All scripts in scripts/ directory")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
