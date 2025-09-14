#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Script to run detailed noise analysis on a trained diffusion model.
Initial settings, argument parsing, model/data loading, and reproducibility follow run_diffusion.py.
"""
import argparse
import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DiffusionModel
from src.utils import set_seed, setup_logging
from utils.noise_utils import (
    plot_error_statistics,
    plot_loss_vs_timestep,
    plot_noise_analysis_results,
    plot_noise_correlation_scatter,
    plot_noise_histogram_grid,
    plot_noise_overlay_with_comparison,
    plot_noise_scales,
    run_noise_analysis,
    save_noise_analysis,
)
from utils.noise_utils_marker import (
    analyze_marker_noise_trajectory,
    plot_marker_noise_trajectory,
    plot_noise_evolution,
    track_single_run_at_marker,
)
from utils.noise_utils_sample import (
    analyze_sample_error_by_position,
    analyze_sample_noise_trajectory,
    plot_sample_error_by_position,
    plot_sample_noise_trajectory,
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
        "--marker_index", type=int, default=50, help="Index of marker to analyze"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to analyze"
    )
    return parser.parse_args()


def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="reverse")
    logger.info("Starting run_noise script.")

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
    output_dir = Path(args.checkpoint).parent.parent / "noise_analysis"
    output_dir.mkdir(exist_ok=True)

    # Load Dataset (Test)
    logger.info("Loading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    logger.info("Preparing a batch of test data...")
    test_batch = next(iter(test_loader)).to(device)
    x0 = test_batch.unsqueeze(1)  # Add channel dimension
    logger.info(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # ===================== Run Noise Analysis =====================

    # Model Loss vs Timestep
    loss_vs_timestep = True
    if loss_vs_timestep:
        print("\n" + "=" * 70)
        print(" RUNNING LOSS/SCALE VS TIMESTEP ")
        print("=" * 70)

        print(
            f"Running noise analysis for all timesteps with {args.num_samples} samples"
        )
        results = run_noise_analysis(
            model, x0, num_samples=args.num_samples, timesteps=None, verbose=False
        )

        # Save noise analysis results
        save_noise_analysis(results, output_dir)

        # Plot loss and noise scales vs timestep
        plot_loss_vs_timestep(results, output_dir)
        plot_noise_scales(results, output_dir)
        plot_noise_analysis_results(results, output_dir)
        plot_error_statistics(results, output_dir)
        print(
            f"\nLoss and noise scales vs timestep analysis complete! Results saved to {output_dir}"
        )

    # Batch Noise Analysis
    batch_noise_analysis = True
    if batch_noise_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING BATCH NOISE ANALYSIS ")
        print("=" * 70)

        # Use provided timesteps or default range
        timesteps = [1, 2, 10, 400, 500, 600, 979, 989, 999, 1000]

        print(
            f"Running noise analysis at selected timesteps with {args.num_samples} samples"
        )
        batch_results = run_noise_analysis(
            model, x0, num_samples=args.num_samples, timesteps=timesteps, verbose=False
        )

        # Plot and save noise analysis results
        plot_noise_histogram_grid(batch_results, output_dir, num_bins=50)
        plot_noise_overlay_with_comparison(
            batch_results, output_dir, num_bins=50, mode="difference"
        )
        plot_noise_correlation_scatter(batch_results, output_dir)
        print(f"\nBatch noise analysis complete! Results saved to {output_dir}")

    # Sample Noise Analysis
    sample_noise_analysis = True
    if sample_noise_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING SAMPLE NOISE ANALYSIS ")
        print("=" * 70)

        timesteps = [1, 2, 10, 400, 500, 600, 979, 989, 999, 1000]
        print("\nRunning noise analysis at selected timestep with single sample")

        sample_idx = 0

        noise_trajectory = analyze_sample_noise_trajectory(
            model, x0, sample_idx, position=None, timesteps=None
        )
        plot_sample_noise_trajectory(
            noise_trajectory, sample_idx, output_dir, position=None
        )

        errors = analyze_sample_error_by_position(model, x0, sample_idx, timestep=1000)
        plot_sample_error_by_position(errors, output_dir, sample_idx, timestep=1000)

        print(f"\nSample noise analysis complete! Results saved to {output_dir}")

    # Marker Noise Analysis
    marker_analysis = True
    if marker_analysis:
        print("\n" + "=" * 70)
        print(" RUNNING MARKER-SPECIFIC ANALYSIS ")
        print("=" * 70)

        # Run Marker analysis
        print(f"\nAnalyzing Marker at index {args.marker_index}...")

        marker_noise_trajectory = analyze_marker_noise_trajectory(
            model=model,
            x0=x0,
            sample_idx=0,
            marker_index=args.marker_index,
            timesteps=None,
        )
        plot_marker_noise_trajectory(
            marker_noise_trajectory,
            sample_idx=0,
            output_dir=output_dir,
            marker_index=args.marker_index,
        )

        x0_sample = x0[0:1].to(x0.device)
        marker_result = track_single_run_at_marker(
            model=model,
            x0=x0_sample,
            marker_index=args.marker_index,
        )

        # Plot noise evolution for the first run
        plot_noise_evolution(
            timesteps=marker_result["timesteps"],
            true_noises=marker_result["true_noises"],
            pred_noises=marker_result["pred_noises"],
            marker_index=args.marker_index,
            output_dir=output_dir,
        )

        print(f"\nMarker analysis complete! Results saved to {output_dir}")

    logger.info("Noise analysis complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
