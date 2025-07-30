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

from src import DiffusionModel
from src.mlp import IdentityNoisePredictor
from src.utils import load_config, set_seed, setup_logging
from utils.ddpm_utils import (
    compute_locality_metrics,
    format_metrics_report,
    get_noisy_sample,
    plot_denoising_comparison,
    plot_denoising_trajectory,
    plot_locality_analysis,
    run_markov_reverse_process,
)

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Loads a DiffusionModel from a checkpoint and moves it to the specified device.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        model: The loaded DiffusionModel (on the correct device, in eval mode)
        config: The config/hparams dictionary from the checkpoint
    """
    from src import DiffusionModel

    model = DiffusionModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    config = model.hparams
    model = model.to(device)
    model.eval()
    return model, config


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
        "--identity-model",
        action="store_true",
        help="Use identity noise predictor for debugging (ignore checkpoint)",
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
    if args.identity_model:
        logger.info("Using IdentityNoisePredictor for debugging")
        config = load_config(config_path="config.yaml")
        model = DiffusionModel(config).to(device)
        model.noise_predictor = IdentityNoisePredictor(
            seq_length=config["data"]["seq_length"],
            embedding_dim=config["unet"]["embedding_dim"],
            dim_mults=config["unet"]["dim_mults"],
            channels=config["unet"]["channels"],
            with_time_emb=config["unet"]["with_time_emb"],
            with_pos_emb=config["unet"]["with_pos_emb"],
            resnet_block_groups=config["unet"]["resnet_block_groups"],
        ).to(device)
        logger.info("Created IdentityNoisePredictor model")
        logger.info("Model config:")
        print(f"\n{config}\n")
    else:
        if not args.checkpoint:
            raise ValueError(
                "Must provide --checkpoint when not using --identity-model"
            )
        try:
            logger.info(f"Loading model from checkpoint: {args.checkpoint}")
            model, config = load_model_from_checkpoint(args.checkpoint, device)
            logger.info("Model loaded successfully from checkpoint on %s", device)
            logger.info("Model config loaded from checkpoint:")
            print(f"\n{config}\n")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    if args.identity_model:
        output_dir = Path("outputs/analysis") / "identity_model"
    else:
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
    diffusion_steps = 1000

    # Generate noisy sample x_t at t=T
    x_t = get_noisy_sample(model, x0, diffusion_steps)
    # x_t = x0

    # Run Reverse Diffusion Process (Markov Chain)
    samples_dict = run_markov_reverse_process(
        model, x0, x_t, diffusion_steps, device, return_all_steps=True, print_mse=True
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
        f"MSE(x_t_minus_1, x0): {mse_x0:.6f}, Corr(x_t_minus_1, x0): {corr_x0:.6f} | MSE(x_t_minus_1, x_t): {mse_xt:.6f}, Corr(x_t_minus_1, x_t): {corr_xt:.6f}"
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
