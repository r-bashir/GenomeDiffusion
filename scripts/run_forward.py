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
    generate_timesteps,
    plot_diffusion_parameters,
    plot_sample_evolution,
    plot_snr,
    print_forward_statistics,
    run_forward_process,
)
from scripts.utils.schedule_utils import (
    analyze_schedule_parameters,
    compare_schedule_parameters,
    print_parameter_comparison,
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

    # ===================== Analyze Schedules =====================
    print("\nAnalyzing diffusion schedules...")

    # Analyze linear schedule
    print("\nAnalyzing linear schedule...")
    forward_diff_linear = ForwardDiffusion(
        diffusion_steps=config.get("diffusion", {}).get("timesteps", 1000),
        beta_start=config.get("diffusion", {}).get("beta_start", 0.0001),
        beta_end=config.get("diffusion", {}).get("beta_end", 0.02),
        schedule_type="linear",
        max_beta=config.get("diffusion", {}).get("max_beta", 0.999),
    )
    forward_diff_linear = forward_diff_linear.to(device)
    # analyze_schedule_parameters(forward_diff_linear, output_dir, schedule_type="linear")

    # Analyze cosine schedule
    print("\nAnalyzing cosine schedule...")
    forward_diff_cosine = ForwardDiffusion(
        diffusion_steps=config.get("diffusion", {}).get("timesteps", 1000),
        beta_start=config.get("diffusion", {}).get("beta_start", 0.0001),
        beta_end=config.get("diffusion", {}).get("beta_end", 0.02),
        schedule_type="cosine",
        max_beta=config.get("diffusion", {}).get("max_beta", 0.999),
    )
    forward_diff_cosine = forward_diff_cosine.to(device)
    analyze_schedule_parameters(forward_diff_cosine, output_dir, schedule_type="cosine")

    # Continue with the main analysis using the configured schedule
    schedule_type = config.get("diffusion", {}).get("schedule_type", "cosine")
    forward_diff = (
        forward_diff_cosine if schedule_type == "cosine" else forward_diff_linear
    )
    print(f"\nContinuing analysis with {schedule_type} schedule...")

    # ===================== Run Forward Diffusion =====================
    # Generate timesteps for analysis
    tmin, tmax = forward_diff.tmin, forward_diff.tmax
    print(f"\nGenerating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)

    # Run forward diffusion process with boundary timesteps
    print("Forward diffusion process with boundary timesteps...")
    boundary_results = run_forward_process(
        forward_diff, x0, timestep_sets["boundary"], verbose=False
    )

    # Print statistics for boundary timesteps
    print(f"\nStatistics for boundary timesteps: {timestep_sets['boundary']}")
    print_forward_statistics(boundary_results, timestep_sets["boundary"])

    # Compare schedule parameters with actual values
    print("\nComparing schedule parameters...")
    compare_schedule_parameters(
        forward_diff, boundary_results, output_dir, schedule_type
    )

    # Print parameter comparison
    print("\nParameter comparison...")
    print_parameter_comparison(
        forward_diff, boundary_results, timestep_sets["boundary"]
    )

    # Plot signal noise ratio
    print("\nSignal noise ratio (SNR)...")
    plot_snr(forward_diff, boundary_results, save_dir=output_dir, verbose=False)

    # Plot diffusion superposition
    print("\nSample evolution through timesteps...")
    plot_sample_evolution(
        forward_diff,
        boundary_results,
        x0,
        timesteps=[1, 2, 500, 999, 1000],
        save_dir=output_dir,
    )

    # Plot diffusion parameters
    print("\nDiffusion parameters...")
    plot_diffusion_parameters(boundary_results, x0, save_dir=output_dir)

    print("\nForward diffusion complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
