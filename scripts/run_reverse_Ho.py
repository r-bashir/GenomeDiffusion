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

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All imports after path modification
# We need to disable the import-not-at-top lint rule
# ruff: noqa: E402

from scripts.utils.reverse_utils import (
    generate_timesteps,
    plot_reverse_diagnostics,
    plot_reverse_mean_components,
    plot_schedule_parameters,
    print_diagnostic_statistics,
    print_reverse_statistics,
    run_reverse_process,
    visualize_diffusion_process_lineplot,
    visualize_diffusion_process_superimposed,
)
from src import DiffusionModel
from src.utils import set_seed, setup_logging

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
    parser = argparse.ArgumentParser(
        description="Test diffusion model noise prediction"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to analyze"
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

    try:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully from checkpoint on %s", device)
        logger.info("Model config loaded from checkpoint:")
        print(f"\n{config}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    output_dir = Path(args.checkpoint).parent.parent / "reverse_diffusion"
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

    # ===================== Analyze Reverse Diffusion =====================
    logger.info("Analyzing reverse diffusion...")

    # Generate timesteps for analysis
    tmin, tmax = model.forward_diffusion.tmin, model.forward_diffusion.tmax
    logger.info(f"Generating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)

    # Select timesteps for analysis
    timesteps = timestep_sets["boundary"]
    logger.info(f"Selected timesteps: {timesteps}")

    # Run reverse diffusion process with boundary timesteps
    logger.info("Reverse diffusion with boundary timesteps...")
    results = run_reverse_process(
        model=model,
        x0=x0,
        timesteps=timesteps,
    )

    # Print statistics for all timesteps
    logger.info("Printing statistics for all timesteps...")
    print_reverse_statistics(results, timesteps)

    # Schedule parameters for all timesteps
    logger.info("Plotting schedule parameters for all timesteps...")
    plot_schedule_parameters(results, timesteps, output_dir)

    # Special Diagnostics
    diagnostics = True
    if diagnostics:

        # Selected timesteps for analysis
        key_timesteps = [1, 998, 999, 1000]
        logger.info(f"Selected key timesteps: {key_timesteps}")

        # Print diagnostic statistics for key timesteps
        logger.info(f"Analyzing diagnostics for key timesteps: {key_timesteps}")
        print_diagnostic_statistics(results=results, timesteps=key_timesteps)

        # Plot diagnostics for key timesteps
        logger.info(f"Generating diagnostic plots for key timesteps: {key_timesteps}")
        plot_reverse_diagnostics(
            results=results, timesteps=key_timesteps, output_dir=output_dir
        )

    # Plotting
    plotting = True
    if plotting:

        # Selected timesteps for analysis
        key_timesteps = [1, 998, 999, 1000]
        logger.info(f"Selected key timesteps: {key_timesteps}")

        # Diffusion evolution with key timesteps
        logger.info(f"Plotting sample evolution for key timesteps: {key_timesteps}")
        visualize_diffusion_process_lineplot(
            results=results,
            timesteps=key_timesteps,
            output_dir=output_dir,
        )
        visualize_diffusion_process_superimposed(
            results=results,
            timesteps=key_timesteps,
            output_dir=output_dir,
        )

        # Plot reverse mean components for key timesteps
        logger.info(f"Plotting mean components for key timesteps: {key_timesteps}")
        plot_reverse_mean_components(
            results=results,
            timesteps=key_timesteps,
            output_dir=output_dir,
        )

    logger.info("Reverse diffusion complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
