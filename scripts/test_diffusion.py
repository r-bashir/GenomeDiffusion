#!/usr/bin/env python
# coding: utf-8

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps
and visualizes how data transforms during forward and reverse diffusion.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from test_diff_utils import (
    display_diffusion_parameters,
    plot_diffusion_process,
    test_diffusion_at_timestep,
)
from test_noise_utils import (
    plot_error_heatmap,
    plot_error_statistics,
    plot_loss_vs_timestep,
    plot_noise_analysis_results,
    plot_noise_distributions,
    plot_noise_scales,
    run_noise_analysis,
    save_noise_analysis,
)
from test_snp_utils import analyze_single_snp, plot_noise_evolution
from torch import Tensor

from src import DiffusionModel
from src.utils import set_seed

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
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to analyze (default: 5)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs for variance estimation (default: 3)",
    )
    return parser.parse_args()


def main():
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

        # Get model config and move to device
        config = model.hparams
        model = model.to(device)
        model.eval()

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)

    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Setup output directory structure
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    print(f"\nResults will be saved to: {base_dir}")

    # Load test dataset
    print("\nLoading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Get a batch of test data
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    print(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # Run analyses
    base_dir = Path(base_dir)

    # 1. Noise analysis
    noise_analysis = True
    if noise_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING NOISE ANALYSIS ")
        print("=" * 70)

        # Create output directory
        noise_analysis_dir = base_dir / "noise_analysis"
        noise_analysis_dir.mkdir(exist_ok=True)

        # Use provided timesteps or default range
        timesteps = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        print(f"Running noise analysis at timesteps: {timesteps}")
        noise_results = run_noise_analysis(
            model, x0, num_samples=args.num_samples, timesteps=timesteps, verbose=False
        )

        # Plot and save noise analysis results
        plot_noise_analysis_results(noise_results, noise_analysis_dir)
        plot_loss_vs_timestep(noise_results, noise_analysis_dir)
        save_noise_analysis(noise_results, noise_analysis_dir)
        plot_noise_distributions(noise_results, noise_analysis_dir)
        # plot_spatial_correlations(noise_results, noise_analysis_dir)
        plot_error_heatmap(noise_results, noise_analysis_dir)
        plot_noise_scales(noise_results, noise_analysis_dir)
        plot_error_statistics(noise_results, noise_analysis_dir)
        print(f"\nNoise analysis complete! Results saved to {noise_analysis_dir}")

    # 2. Run diffusion analysis
    diffusion_analysis = True
    if diffusion_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING DIFFUSION ANALYSIS ")
        print("=" * 70)

        # Create output directory
        diffusion_analysis_dir = base_dir / "diffusion_analysis"
        diffusion_analysis_dir.mkdir(exist_ok=True)

        # Use provided timesteps or default range (fewer timesteps for diffusion analysis)
        diff_timesteps = [1, 10, 100, 200, 500, 600, 800, 900, 1000]

        print(f"Running diffusion analysis at timesteps: {diff_timesteps}")
        print(f"Using {args.num_samples} samples")

        # Run the diffusion analysis
        print("\nRunning diffusion analysis...")

        # Create results dictionary to store metrics
        diffusion_results = {}

        # Run analysis for each timestep
        for t in diff_timesteps:
            print(f"\n--- Analyzing timestep {t} ---")

            # Run the diffusion step and get results
            result = test_diffusion_at_timestep(
                model=model,
                x0=x0[: args.num_samples],  # Use specified number of samples
                timestep=t,
                plot=True,
                save_plot=True,
                output_dir=str(diffusion_analysis_dir),
            )

            # Generate comprehensive diffusion process plot
            if t in [1, diff_timesteps[len(diff_timesteps) // 2], diff_timesteps[-1]]:
                plot_diffusion_process(
                    x0=result["x0"][:1],  # Use x0 from result to ensure consistency
                    noise=result["noise"][:1],
                    x_t=result["xt"][:1],  # Changed from "x_t" to "xt"
                    predicted_noise=result["predicted_noise"][:1],
                    x_t_minus_1=result["x_t_minus_1"][:1],
                    timestep=t,
                    save_dir=str(diffusion_analysis_dir / "diffusion_plots"),
                )

            # Store results
            diffusion_results[t] = result

            # Print detailed results
            print(f"\nResults for timestep {t}:")
            print("-" * 40)
            print(f"MSE: {result['metrics']['mse']:.6f}")
            print(f"Weighted MSE: {result['metrics']['weighted_mse']:.6f}")
            print(f"Reconstruction MSE: {result['metrics']['x0_diff']:.6f}")

            # Display diffusion parameters for key timesteps
            if t in [1, diff_timesteps[len(diff_timesteps) // 2], diff_timesteps[-1]]:
                display_diffusion_parameters(model, t)

        print("\nDiffusion analysis complete!")
        print(f"Results saved to: {diffusion_analysis_dir}")
        print(f"- Individual step plots: {diffusion_analysis_dir}")
        print(
            f"- Combined process visualizations: {diffusion_analysis_dir / 'diffusion_plots'}"
        )

    # 3. Run SNP-specific analysis
    snp_analysis = True
    if snp_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING SNP-SPECIFIC ANALYSIS ")
        print("=" * 70)

        # Create output directory
        snp_analysis_dir = base_dir / "snp_analysis"
        snp_analysis_dir.mkdir(exist_ok=True)

        # Run SNP analysis
        print(f"\nAnalyzing SNP at index {args.snp_index}...")
        snp_results = analyze_single_snp(
            model=model,
            x0=x0,
            snp_index=args.snp_index,
            num_runs=args.num_runs,
            output_dir=snp_analysis_dir,
        )

        # Plot noise evolution for the first run
        if snp_results["all_runs"]:
            first_run = snp_results["all_runs"][0]
            plot_path = snp_analysis_dir / f"snp_{args.snp_index}_noise_evolution.png"
            plot_noise_evolution(
                timesteps=first_run["timesteps"],
                true_noises=first_run["true_noises"],
                pred_noises=first_run["pred_noises"],
                snp_index=args.snp_index,
                save_path=plot_path,
            )

        print(f"\nSNP analysis complete! Results saved to {snp_analysis_dir}")


if __name__ == "__main__":
    main()
