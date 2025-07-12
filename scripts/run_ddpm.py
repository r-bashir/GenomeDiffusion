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
from src.utils import set_seed, setup_logging
from utils.ddpm_utils import (
    get_noisy_sample,
    plot_denoising_comparison,
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
    logger.info("Starting run_reverse script.")

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
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    output_dir = base_dir / "ddpm_diffusion"
    output_dir.mkdir(exist_ok=True)

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    test_batch = next(iter(test_loader)).to(device)

    # Select a single sample and ensure shape [1, 1, seq_len]
    sample_idx = 0
    if test_batch.dim() == 2:
        # [batch, seq_len] -> [1, 1, seq_len]
        x0 = test_batch[sample_idx : sample_idx + 1].unsqueeze(1)
    elif test_batch.dim() == 3:
        # [batch, channels, seq_len] -> [1, channels, seq_len]
        x0 = test_batch[sample_idx : sample_idx + 1]
    else:
        raise ValueError(f"Unexpected test_batch shape: {test_batch.shape}")

    x0 = x0.to(device)
    logger.info(
        f"Selected x0 shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}"
    )
    logger.info(f"Sample unique values: {torch.unique(x0)}")
    logger.info(f"First 10 values: {x0[0, 0, :10]}")

    # --- New: Markov Reverse Process from Noisy x0 Diagnostic ---
    t_markov = 1000
    logger.info(f"Running Markov reverse process from noisy x0 at t={t_markov}...")

    # Generate noisy sample at t_markov
    x_t = get_noisy_sample(model, x0, t_markov)

    # Run Markov reverse process
    samples_dict = run_markov_reverse_process(
        model, x_t, t_markov, device, return_all_steps=True
    )
    x0_recon_markov = samples_dict[1]

    # Plot and compare
    mse, corr = plot_denoising_comparison(
        x0,
        x_t,
        x0_recon_markov,
        t_markov,
        output_dir / "markov_reverse_t{t_markov}.png",
    )
    logger.info(
        f"Markov reverse denoising plot saved to: {output_dir}/markov_reverse_t{t_markov}.png"
    )
    logger.info(f"MSE: {mse:.6f} | Corr: {corr:.4f}")

    logger.info("DDPM complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
