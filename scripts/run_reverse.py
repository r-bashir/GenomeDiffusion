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
from src.utils import set_seed, setup_logging
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
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="reverse")
    logger.info("Starting run_reverse script.")

    # Set global seed
    set_seed(seed=42)

    try:
        # Load the model from checkpoint
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        # Get model config and move to device
        config = model.hparams
        model = model.to(device)
        model.eval()

        logger.info("Model loaded successfully from checkpoint on {device}")
        logger.info("Model config loaded from checkpoint:")
        print(f"\n{config}\n")

    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    output_dir = base_dir / "reverse_diffusion"
    output_dir.mkdir(exist_ok=True)

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    logger.info(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # ===================== Run Reverse Diffusion =====================
    # Generate timesteps for analysis
    tmin, tmax = model.forward_diffusion.tmin, model.forward_diffusion.tmax
    logger.info(f"Generating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)

    # Select timesteps for analysis
    timesteps = timestep_sets["boundary"]
    logger.info(f"Selected timesteps: {timesteps}")

    # Run reverse diffusion process with boundary timesteps
    logger.info("Reverse diffusion process with boundary timesteps...")
    results = run_reverse_process(
        model=model,
        x0=x0,
        timesteps=timesteps,
        num_samples=args.num_samples,
    )

    # Print statistics for boundary timesteps
    logger.info(f"Statistics for boundary timesteps: {timesteps}")
    print_reverse_statistics(results, timesteps)

    # Diffusion evolution with key timesteps
    key_timesteps = [1, 2, 500, 998, 999, 1000]
    logger.info(f"Diffusion evolution with key timesteps: {key_timesteps}")

    visualize_diffusion_process_heatmap(
        results=results,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )
    visualize_diffusion_process_lineplot(
        results=results,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )
    visualize_diffusion_process_superimposed(
        results=results,
        timesteps=key_timesteps,
        output_dir=output_dir,
        sample_points=200,
    )

    # Additional plots
    additional_plots = False
    if additional_plots:
        logger.info("Extra plots...")

        # Create output directory for visualizations
        viz_dir = output_dir / "diffusion_analysis"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # Extract metrics across timesteps
        timesteps = sorted(results.keys())
        mse_values = [results[t]["metrics"]["noise_mse"] for t in timesteps]
        x0_diff_values = [results[t]["metrics"]["x0_diff"] for t in timesteps]

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
        logger.info(f"Generating visualizations for key timesteps: {key_timesteps}")

        for t in key_timesteps:
            t_dir = viz_dir / f"timestep_{t}"
            t_dir.mkdir(exist_ok=True, parents=True)
            plot_diffusion_results(results[t], save_dir=t_dir)

        # Generate summary metrics plot across all timesteps
        logger.info("Generating summary metrics plot across all timesteps")
        plot_diffusion_metrics(results, save_dir=viz_dir)

    logger.info("Reverse diffusion complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
