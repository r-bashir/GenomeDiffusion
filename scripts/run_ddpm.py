#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps
and visualizes how data transforms during forward and reverse diffusion.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ddpm import build_mixing_mask
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
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to denoise",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Use this sample index from the batch",
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
    output_dir = Path(args.checkpoint).parent.parent / "denoising"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Dataset (Test)
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
    logger.info("Adding channel dimension ([B, L] to [B, C, L])...")
    real_samples = real_samples[:num_samples].unsqueeze(1)
    logger.info(f"Sample shape: {real_samples.shape}, and dim: {real_samples.dim()}")

    # === BEGIN: Reverse Diffusion ===
    logger.info("Starting denoising process...")
    diffusion_timestep = 10  # config["diffusion"]["timesteps"]
    logger.info(f"Starting at denoising from T={diffusion_timestep}")

    # Get real batch x_0
    x_0 = real_samples
    B, C, L = x_0.shape
    logger.info(f"x_0 shape: {x_0.shape} (B={B}, C={C}, L={L}) and dim: {x_0.dim()}")

    # Generate noisy batch x_t at t=T
    x_t = get_noisy_sample(model, x_0, diffusion_timestep)
    # x_t = x_0

    # Test imputation if requested
    true_x0_for_imputation = None
    mask_for_imputation = None

    if args.test_imputation:
        logger.info(f"Testing imputation with mask ratio: {args.mask_ratio}")

        # Use the original clean sample as ground truth
        true_x0_for_imputation = x_0.clone()

        # Create a random mask (1 = known SNP, 0 = unknown SNP)
        torch.manual_seed(42)  # For reproducible masks
        mask_for_imputation = torch.rand_like(x_0) < args.mask_ratio
        mask_for_imputation = mask_for_imputation.float()

        known_snps = mask_for_imputation.sum().item()
        total_snps = mask_for_imputation.numel()
        logger.info(
            f"Imputation mask: {known_snps}/{total_snps} SNPs known ({known_snps/total_snps:.1%})"
        )

        logger.info("Running denoising WITH imputation...")
    else:
        logger.info("Running denoising WITHOUT imputation...")

    # Optional mixing mask for staircase vs real metrics
    mixing_mask = None
    try:
        data_cfg = config.get("data", {})
        if data_cfg.get("mixing", False):
            pattern = data_cfg.get("mixing_pattern", [])
            interval = int(data_cfg.get("mixing_interval", 0))
            seq_length = int(config["data"]["seq_length"])
            mixing_mask = build_mixing_mask(
                seq_length, pattern, interval, device=device
            )
            # Ensure mask has shape [1,1,L] and boolean dtype
            mixing_mask = mixing_mask.to(device=device, dtype=torch.bool)
    except Exception as e:
        logger.warning(f"Could not build mixing mask: {e}")

    # Run Reverse Diffusion Process (Denoising)
    samples_dict = run_denoising_process(
        model,
        x_0,
        x_t,
        diffusion_timestep,
        device,
        return_all_steps=True,
        print_mse=True,
        true_x0=true_x0_for_imputation,
        imputation_mask=mask_for_imputation,
        mixing_mask=mixing_mask,
    )

    # Plotting: either selected sample or all samples
    x_t_minus_1 = samples_dict[0]  # Final denoised batch (x_{t-1} at t=0)
    if args.sample_idx is None:
        logger.info("Plotting all samples in batch...")
        for i in range(B):
            x0_i = x_0[i : i + 1]
            xt_i = x_t[i : i + 1]
            samples_i = {t: s[i : i + 1] for t, s in samples_dict.items()}
            plot_denoising_comparison(
                x0_i,
                xt_i,
                samples_i[0],
                diffusion_timestep,
                output_dir,
                filename_suffix=f"_sample{i}",
            )
            plot_denoising_trajectory(
                x0_i,
                xt_i,
                samples_i,
                diffusion_timestep,
                output_dir,
                filename_suffix=f"_sample{i}",
                print_step_metrics=True,
                mixing_mask=mixing_mask,
            )
    else:
        sample_idx = int(args.sample_idx)
        if not (0 <= sample_idx < B):
            logger.warning(
                f"--sample_idx {sample_idx} is out of range for batch size {B}. Defaulting to 0."
            )
            sample_idx = 0
        logger.info(f"Plotting only sample_idx={sample_idx}")
        x0_i = x_0[sample_idx : sample_idx + 1]
        xt_i = x_t[sample_idx : sample_idx + 1]
        samples_i = {t: s[sample_idx : sample_idx + 1] for t, s in samples_dict.items()}
        plot_denoising_comparison(
            x0_i,
            xt_i,
            samples_i[0],
            diffusion_timestep,
            output_dir,
            filename_suffix=f"_sample{sample_idx}",
        )
        plot_denoising_trajectory(
            x0_i,
            xt_i,
            samples_i,
            diffusion_timestep,
            output_dir,
            filename_suffix=f"_sample{sample_idx}",
            print_step_metrics=True,
            mixing_mask=mixing_mask,
        )

    # Test Imputation
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

            logger.info("Imputation accuracy at known positions:")
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
                (x_t_minus_1[unknown_positions] - x_0[unknown_positions]) ** 2
            ).item()
            mse_known = torch.mean(
                (x_t_minus_1[known_positions] - x_0[known_positions]) ** 2
            ).item()

            logger.info("Reconstruction quality comparison:")
            logger.info(f"  MSE at unknown positions: {mse_unknown:.6f}")
            logger.info(f"  MSE at known positions: {mse_known:.6f}")
            logger.info(
                f"  Improvement ratio: {mse_unknown/mse_known:.2f}x better at known positions"
            )

    # === END: Reverse Diffusion ===

    logger.info("DDPM complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
