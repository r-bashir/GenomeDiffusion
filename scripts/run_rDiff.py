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

from src import DiffusionModel
from src.utils import set_seed
from utils.diff_utils import (
    display_diffusion_parameters,
    plot_diffusion_process,
    test_diffusion_at_timestep,
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

    # Output directory
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    output_dir = base_dir / "reverse_diffusion"
    output_dir.mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Load Dataset (Test)
    print("\nLoading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    print(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # ===================== Run Reverse Diffusion =====================

    # FIXME: Call test functions to analyze reverse diffusion process.
    diffusion_analysis = False
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


if __name__ == "__main__":
    main()
