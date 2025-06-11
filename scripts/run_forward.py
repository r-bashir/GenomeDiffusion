#!/usr/bin/env python
# coding: utf-8

"""
Forward Diffusion Investigation Script

This script investigates the forward diffusion process as described in the forward_diffusion.py
module. It loads a sample, applies forward diffusion at various timesteps, and analyzes
the results to understand how noise is added during the diffusion process.

Usage:
    python scripts/run_fDiffusion.py --config config.yaml [--output-dir output] [--debug]
"""

import argparse

# Standard library imports
import sys
from pathlib import Path

# Add project root to path before any project imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# All imports after path modification
# Disable the import-not-at-top lint rule
# ruff: noqa: E402
import torch

from scripts.utils.forward_utils import (
    create_animation_frames,
    generate_timesteps,
    plot_diffusion_parameters,
    plot_forward_diffusion_sample,
    plot_signal_noise_ratio,
    print_forward_statistics,
    run_forward_process,
)
from src.dataset import SNPDataset
from src.forward_diffusion import ForwardDiffusion
from src.utils import load_config, set_seed

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

    # Set global seed for reproducibility
    set_seed(42)

    # Parse Arguments
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Get base output directory from config and create experiment-specific directory
    base_output_path = Path(config.get("output_path", "output"))
    output_dir = base_output_path / "forward_diffusion"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving results to {output_dir}")

    # Load dataset
    print("Loading dataset...")
    dataset = SNPDataset(config)

    # Select a sample
    sample_idx = 0
    x0 = dataset[sample_idx]

    # Reshape sample for visualization: add batch and channel dimensions [batch, channel, seq_length]
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length]
    elif x0.dim() == 2:
        x0 = x0.unsqueeze(0)  # [1, channels, seq_length]

    x0 = x0.to(device)
    print(f"Sample shape: {x0.shape}")
    print(f"Sample unique values: {torch.unique(x0)}")
    print(f"First 10 values: {x0[0, 0, :10]}")

    # Run forward diffusion process
    print("\nInitializing forward diffusion model...")
    forward_diff = ForwardDiffusion(
        diffusion_steps=config.get("diffusion", {}).get("timesteps", 1000),
        beta_start=config.get("diffusion", {}).get("beta_start", 0.0001),
        beta_end=config.get("diffusion", {}).get("beta_end", 0.02),
        schedule_type=config.get("diffusion", {}).get("schedule_type", "cosine"),
        max_beta=config.get("diffusion", {}).get("max_beta", 0.999),
    )

    # Move to GPU if available
    forward_diff = forward_diff.to(device)
    print(f"Using device: {device}")

    # Generate timesteps for analysis
    tmin = forward_diff.tmin
    tmax = forward_diff.tmax
    print(f"Generating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)

    # Run forward diffusion process with boundary timesteps
    print("\nForward diffusion process with boundary timesteps...")
    boundary_results = run_forward_process(
        forward_diff, x0, timestep_sets["boundary"], verbose=False
    )

    # Print statistics for key timesteps
    print("\nStatistics for key timesteps...")
    key_timesteps = [tmin, tmin + 1, tmax // 2, tmax - 1, tmax]
    print_forward_statistics(boundary_results, key_timesteps)

    print("\nPlots with boundary timesteps...")
    plot_forward_diffusion_sample(boundary_results, x0=x0, save_dir=output_dir)
    plot_signal_noise_ratio(
        boundary_results, x0=None, save_dir=output_dir, verbose=False
    )
    plot_diffusion_parameters(boundary_results, x0=None, save_dir=output_dir)

    # Additional plots
    additional_plots = False
    if additional_plots:
        print("\nExtra plots...")

        # Create animation frames
        print("\nAnimation frames...")
        create_animation_frames(x0, boundary_results, save_dir=output_dir)

        # Also run forward process with logarithmic timesteps for more comprehensive analysis
        print("\nLogarithmic timesteps...")
        log_results = run_forward_process(
            forward_diff, x0, timestep_sets["log"], verbose=False
        )

        # Save results to output directory
        log_output_dir = output_dir / "log_timesteps"
        log_output_dir.mkdir(exist_ok=True, parents=True)
        plot_forward_diffusion_sample(log_results, x0, save_dir=log_output_dir)
        plot_signal_noise_ratio(
            log_results, x0=x0, save_dir=log_output_dir, verbose=False
        )
        plot_diffusion_parameters(log_results, x0=x0, save_dir=log_output_dir)

    print(f"\nAll results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    main()
