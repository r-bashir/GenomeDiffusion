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

from src import DiffusionModel
from src.utils import set_seed
from utils.reverse_utils import (
    generate_timesteps,
    plot_diffusion_metrics,
    plot_diffusion_results,
    print_reverse_statistics,
    run_reverse_process,
    visualize_diffusion_process_heatmap,
    visualize_diffusion_process_lineplot,
    visualize_diffusion_process_superimposed,
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
        "--num_samples", type=int, default=3, help="Number of samples to analyze"
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
    # Generate timesteps for analysis
    tmin, tmax = model.forward_diffusion.tmin, model.forward_diffusion.tmax
    print(f"\nGenerating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)

    # Run reverse diffusion process with boundary timesteps
    print("Reverse diffusion process with boundary timesteps...")
    boundary_results = run_reverse_process(
        model=model,
        x0=x0,
        num_samples=args.num_samples,
        timesteps=timestep_sets["boundary"],
        verbose=False,
    )

    # Print statistics for boundary timesteps
    print(f"\nStatistics for boundary timesteps: {timestep_sets['boundary']}")
    print_reverse_statistics(boundary_results, timestep_sets["boundary"])

    # Diffusion evolution with key timesteps
    key_timesteps = [1, 2, 500, 998, 999, 1000]
    print(f"\nDiffusion evolution with key timesteps: {key_timesteps}")

    visualize_diffusion_process_heatmap(
        model=model,
        batch=x0,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )
    visualize_diffusion_process_lineplot(
        model=model,
        batch=x0,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )
    visualize_diffusion_process_superimposed(
        model=model,
        batch=x0,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )

    # Additional plots
    additional_plots = False
    if additional_plots:
        print("\nExtra plots...")

        # Create output directory for visualizations
        viz_dir = output_dir / "diffusion_analysis"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # Extract metrics across timesteps
        timesteps = sorted(boundary_results.keys())
        mse_values = [boundary_results[t]["metrics"]["noise_mse"] for t in timesteps]
        x0_diff_values = [boundary_results[t]["metrics"]["x0_diff"] for t in timesteps]

        # Print summary of results
        print("\n" + "=" * 70)
        print(" DIFFUSION ANALYSIS SUMMARY ")
        print("=" * 70)
        print(
            f"Analyzed {len(timesteps)} timesteps from {min(timesteps)} to {max(timesteps)}"
        )
        print(f"Average noise prediction MSE: {sum(mse_values) / len(mse_values):.6f}")
        print(
            f"Average reconstruction error: {sum(x0_diff_values) / len(x0_diff_values):.6f}"
        )

        # Generate plots for key timesteps (beginning, middle, end)
        key_timesteps = [
            min(timesteps),
            timesteps[len(timesteps) // 2],
            max(timesteps),
        ]
        print(f"\nGenerating visualizations for key timesteps: {key_timesteps}")

        for t in key_timesteps:
            t_dir = viz_dir / f"timestep_{t}"
            t_dir.mkdir(exist_ok=True, parents=True)
            plot_diffusion_results(boundary_results[t], save_dir=t_dir)

        # Generate summary metrics plot across all timesteps
        print("\nGenerating summary metrics plot across all timesteps")
        plot_diffusion_metrics(boundary_results, save_dir=viz_dir)

    print("\nReverse diffusion complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
