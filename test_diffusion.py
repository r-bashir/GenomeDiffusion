#!/usr/bin/env python
# coding: utf-8

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps
and visualizes how data transforms during forward and reverse diffusion.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import DiffusionModel
from src.utils import set_seed
from test_diffusion_utils import (
    test_diffusion_at_timestep,
    plot_noise_evolution,
    track_single_run_at_snp,
    calculate_variance_across_runs,
    plot_variance_statistics,
)

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test diffusion model noise prediction"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--snp_index",
        type=int,
        default=50,
        help="Index of the SNP to monitor (default: 50)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs for variance estimation (default: 3)",
    )
    return parser.parse_args()


def main():
    """Main function."""

    # Set global seed for reproducibility
    set_seed(42)

    # Parse Arguments
    args = parse_args()

    try:
        # Load the model from checkpoint
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        config = model.hparams  # model config used during training
        model = model.to(device)  # move model to device
        model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)

    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    print(f"\nResults will be saved to: {base_dir}")

    # Load test dataset
    print("\nLoading test dataset...")
    model.setup("test")  # Initialize test dataset
    test_loader = model.test_dataloader()

    # Get a batch of test data
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension

    print(f"Input shape: {x0.shape}")

    # === Task1: Analyze Diffusion ===
    output_dir = base_dir / "diffusion_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    timesteps_to_test = [1, 2, 250, 500, 750, 999, 1000]
    print(f"\nAnalyzing diffusion process at timesteps: {timesteps_to_test}")
    for t in timesteps_to_test:
        test_diffusion_at_timestep(
            model, x0, timestep=t, plot=True, save_plot=True, output_dir=output_dir
        )

    # === Task2: Analyze Noise ===
    print(f"\nAnalyzing noise at SNP index {args.snp_index}...")

    output_dir = base_dir / "noise_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Perform multiple runs
    all_runs = []
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}...")
        run_result = track_single_run_at_snp(model, x0, args.snp_index)
        all_runs.append(run_result)

        # Plot individual run with enhanced visualization
        plot_noise_evolution(
            run_result["timesteps"],
            run_result["true_noises"],
            run_result["pred_noises"],
            snp_index=args.snp_index,
            save_path=str(output_dir / f"noise_evolution_run{run+1}.png"),
        )

    # Calculate and plot variance statistics
    variance_stats = calculate_variance_across_runs(all_runs)
    if variance_stats:
        plot_variance_statistics(
            variance_stats, save_path=str(output_dir / "noise_variance_analysis.png")
        )

        # Save statistics to file
        stats_path = output_dir / "noise_statistics.txt"
        with open(stats_path, "w") as f:
            f.write("Noise Analysis Statistics\n")
            f.write("========================\n\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"SNP Index: {args.snp_index}\n")
            f.write(f"Number of runs: {args.num_runs}\n\n")
            f.write("Variance Analysis:\n")
            f.write(
                f"- Average true noise variance: {np.mean(variance_stats['true_variance']):.6f}\n"
            )
            f.write(
                f"- Average predicted noise variance: {np.mean(variance_stats['pred_variance']):.6f}\n"
            )
            f.write(f"- Average MSE: {np.mean(variance_stats['mse']):.6f}\n")
            f.write(
                f"- Max MSE at timestep {np.argmax(variance_stats['mse']) + 1}: {np.max(variance_stats['mse']):.6f}\n"
            )

        print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
