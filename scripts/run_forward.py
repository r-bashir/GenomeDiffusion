#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Forward Diffusion Investigation Script

This script investigates the forward diffusion process as described in the forward_diffusion.py
module. It loads a sample, applies forward diffusion at various timesteps, and analyzes
the results to understand how noise is added during the diffusion process.

Usage:
    python scripts/run_fDiffusion.py --config config.yaml [--output-dir output] [--debug]
"""

import argparse
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ruff: noqa: E402
import torch

from scripts.utils.forward_utils import (
    generate_timesteps,
    plot_diffusion_parameters,
    plot_sample_evolution,
    plot_snr,
    run_forward_process,
)
from scripts.utils.schedule_utils import (
    analyze_schedule_parameters,
)
from src.dataset import SNPDataset
from src.forward_diffusion import ForwardDiffusion
from src.utils import load_config, set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Forward Diffusion Investigation")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/forward_diffusion"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output",
    )
    return parser.parse_args()


def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="forward")
    logger.info("Starting run_forward script.")

    # Set global seed
    set_seed(seed=42)

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = SNPDataset(config)

    # Select a sample
    sample_idx = 0
    logger.info(f"Selecting sample with index {sample_idx}...")
    x0 = dataset[sample_idx]
    logger.info(f"x0 sample shape: {x0.shape} and dimensions: {x0.dim()}")
    logger.info("Reshaping to [batch, channel, seq_length]")

    # Reshape sample for visualization
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length]
    elif x0.dim() == 2:
        x0 = x0.unsqueeze(0)  # [1, channels, seq_length]

    x0 = x0.to(device)
    logger.info(f"Sample shape: {x0.shape} and dimensions: {x0.dim()}")
    logger.info(f"Sample unique values: {torch.unique(x0)}")
    logger.info(f"First 10 values: {x0[0, 0, :10]}")

    # Prepare output directory
    output_dir = Path(config["output_path"]) / "forward_diffusion"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Analyze noise schedule
    forward_diff = ForwardDiffusion(
        time_steps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule_type=config["diffusion"]["schedule_type"],
    ).to(device)

    schedule_type = config["diffusion"]["schedule_type"]

    # ===================== Analyze Beta Schedulers =====================
    schedule_analysis = False
    if schedule_analysis:
        # Analyze schedule parameters
        logger.info("Analyzing beta schedule...")
        analyze_schedule_parameters(forward_diff, output_dir, schedule_type)

        # Print schedule parameters
        # logger.info("Printing schedule parameters...")
        # print_schedule_parameters(forward_diff, output_dir, schedule_type)

    # ===================== Analyze Forward Diffusion =====================

    forward_analysis = True
    if forward_analysis:
        logger.info("Analyzing forward diffusion...")

        # Generate timesteps for analysis
        tmin, tmax = forward_diff.tmin, forward_diff.tmax
        logger.info(f"Generating timesteps between {tmin} to {tmax}.")
        timestep_sets = generate_timesteps(tmin, tmax)

        # Select timesteps for analysis
        timesteps = timestep_sets["boundary"]
        logger.info(f"Selected timesteps: {timesteps}")

        # Forward diffusion process with boundary timesteps
        logger.info("Forward diffusion with boundary timesteps...")
        results = run_forward_process(forward_diff, x0, timesteps)

        # Print statistics for boundary timesteps
        # logger.info("Printing statistics for boundary timesteps...")
        # print_forward_statistics(results, timesteps)

        # Plot diffusion superposition
        key_timesteps = [1, 2, 500, 999, 1000]
        logger.info("Plotting sample evolution through timesteps...")
        plot_sample_evolution(results, x0, key_timesteps, output_dir)

        # Plot diffusion parameters
        logger.info("Plotting diffusion parameters...")
        plot_diffusion_parameters(results, x0, output_dir)

        # Plot signal noise ratio
        logger.info("Plotting signal noise ratio (SNR)...")
        plot_snr(forward_diff, results, output_dir, verbose=False)

        # Print schedule comparison
        # logger.info("Printing schedule comparison...")
        # print_schedule_comparison(forward_diff, results, timesteps, schedule_type)

        # Compare schedule parameters with actual values
        # logger.info("Plotting schedule parameters comparison...")
        # plot_schedule_comparison(forward_diff, results, output_dir, schedule_type)

    logger.info("Forward diffusion complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
