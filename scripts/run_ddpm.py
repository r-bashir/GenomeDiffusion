#!/usr/bin/env python
# coding: utf-8

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps
and visualizes how data transforms during forward and reverse diffusion.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All imports after path modification
# We need to disable the import-not-at-top lint rule
# ruff: noqa: E402

from src.utils import load_model_from_checkpoint, set_seed, setup_logging
from utils.ddpm_utils import (
    get_noisy_sample,
    plot_denoising_comparison,
    plot_denoising_trajectory,
    run_denoising_process,
)

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DDPM Full-Cycle Diagnostic")
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--discretize",
        action="store_true",
        help="Discretize generated samples to 0, 0.5, and 1.0",
    )
    parser.add_argument(
        "--test_imputation",
        action="store_true",
        help="Test imputation functionality with known SNPs",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.5,
        help="Ratio of SNPs to keep as known for imputation testing (0.0-1.0)",
    )
    return parser.parse_args()


def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="reverse")
    logger.info("Starting run_ddpm script.")

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
    output_dir = Path(args.checkpoint).parent.parent / "ddpm_diffusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    test_batch = next(iter(test_loader)).to(device)
    logger.info(f"Batch shape: {test_batch.shape}, and dim: {test_batch.dim()}")

    # Select a single sample and ensure shape [1, 1, seq_len]
    logger.info(f"Adding channel dim, and selecting single sample")
    sample_idx = 0
    x0 = test_batch[sample_idx : sample_idx + 1].unsqueeze(1)
    logger.info(f"x0 shape: {x0.shape} and dim: {x0.dim()}")
    logger.info(f"x0 unique values: {torch.unique(x0)}")

    # === BEGIN: Reverse Diffusion ===
    logger.info(f"Running Markov reverse process from x_t at t=T...")
    diffusion_steps = 2

    # Generate noisy sample x_t at t=T
    x_t = get_noisy_sample(model, x0, diffusion_steps)
    # x_t = x0

    # Test imputation if requested
    true_x0_for_imputation = None
    mask_for_imputation = None

    if args.test_imputation:
        logger.info(f"Testing imputation with mask ratio: {args.mask_ratio}")

        # Use the original clean sample as ground truth
        true_x0_for_imputation = x0.clone()

        # Create a random mask (1 = known SNP, 0 = unknown SNP)
        torch.manual_seed(42)  # For reproducible masks
        mask_for_imputation = torch.rand_like(x0) < args.mask_ratio
        mask_for_imputation = mask_for_imputation.float()

        known_snps = mask_for_imputation.sum().item()
        total_snps = mask_for_imputation.numel()
        logger.info(
            f"Imputation mask: {known_snps}/{total_snps} SNPs known ({known_snps/total_snps:.1%})"
        )

        logger.info("Running denoising WITH imputation...")
    else:
        logger.info("Running denoising WITHOUT imputation...")

    # Run Reverse Diffusion Process (Markov Chain)
    samples_dict = run_denoising_process(
        model,
        x0,
        x_t,
        diffusion_steps,
        device,
        return_all_steps=True,
        print_mse=True,
        true_x0=true_x0_for_imputation,
        mask=mask_for_imputation,
    )

    # Plot and compare
    x_t_minus_1 = samples_dict[0]  # Denoised sample (x_{t-1} at t=0)
    mse_x0, corr_x0, mse_xt, corr_xt = plot_denoising_comparison(
        x0,
        x_t,
        x_t_minus_1,
        diffusion_steps,
        output_dir,
    )

    # Debug log
    logger.debug(
        f"MSE(x_t_minus_1, x0): {mse_x0:.6f}, r(x_t_minus_1, x0): {corr_x0:.6f} | MSE(x_t_minus_1, x_t): {mse_xt:.6f}, r(x_t_minus_1, x_t): {corr_xt:.6f}"
    )

    # Additional imputation analysis
    if args.test_imputation and mask_for_imputation is not None:
        logger.info("\n=== IMPUTATION ANALYSIS ===")

        # Check accuracy at known positions
        known_positions = mask_for_imputation == 1.0
        if known_positions.any():
            imputation_diff = torch.abs(
                x_t_minus_1[known_positions] - true_x0_for_imputation[known_positions]
            )
            max_diff = imputation_diff.max().item()
            mean_diff = imputation_diff.mean().item()

            logger.info(f"Imputation accuracy at known positions:")
            logger.info(f"  Max difference: {max_diff:.8f}")
            logger.info(f"  Mean difference: {mean_diff:.8f}")

            if mean_diff < 1e-6:
                logger.info(
                    "✅ Perfect imputation: Known positions exactly match ground truth"
                )
            else:
                logger.warning(
                    f"⚠️  Imputation not perfect: Mean difference {mean_diff:.8f}"
                )

        # Compare reconstruction quality at unknown vs known positions
        unknown_positions = mask_for_imputation == 0.0
        if unknown_positions.any() and known_positions.any():
            mse_unknown = torch.mean(
                (x_t_minus_1[unknown_positions] - x0[unknown_positions]) ** 2
            ).item()
            mse_known = torch.mean(
                (x_t_minus_1[known_positions] - x0[known_positions]) ** 2
            ).item()

            logger.info(f"Reconstruction quality comparison:")
            logger.info(f"  MSE at unknown positions: {mse_unknown:.6f}")
            logger.info(f"  MSE at known positions: {mse_known:.6f}")
            logger.info(
                f"  Improvement ratio: {mse_unknown/mse_known:.2f}x better at known positions"
            )

    # Plot denoising trajectory
    plot_denoising_trajectory(
        x0,
        x_t,
        samples_dict,
        diffusion_steps,
        output_dir,
    )

    # === END: Reverse Diffusion ===

    logger.info("DDPM complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
