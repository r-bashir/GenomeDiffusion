#!/usr/bin/env python
# coding: utf-8

"""Script for generating and analyzing samples from trained SNP diffusion models.

This script uses the utility functions from inference_utils.py to:
1. Generate samples from diffusion models
2. Analyze MAF distributions
3. Create visualizations of real vs. generated data
4. Compute genomic metrics

Examples:
    # Generate samples and analyze
    python inference_new.py --checkpoint path/to/checkpoint.ckpt

    # Generate specific number of samples
    python inference_new.py --checkpoint path/to/checkpoint.ckpt --num_samples 100

    # Generate discretized samples (0, 0.5, 1.0)
    python inference_new.py --checkpoint path/to/checkpoint.ckpt --discretize
"""

import argparse
import json
from pathlib import Path

import torch

from diffusion.diffusion_model import DiffusionModel
from diffusion.inference_utils import (analyze_maf_distribution,
                                       calculate_maf_stats,
                                       compute_genomic_metrics,
                                       plot_comparison,
                                       visualize_reverse_diffusion)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and analyze samples from SNP diffusion models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: match test set size)",
    )
    parser.add_argument(
        "--discretize",
        action="store_true",
        help="Discretize generated samples to 0, 0.5, and 1.0",
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    try:
        print("\nLoading model from checkpoint...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model from checkpoint
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
    if "checkpoints" in str(checkpoint_path):
        base_dir = checkpoint_path.parent.parent
    else:
        base_dir = checkpoint_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nInference results will be saved to: {output_dir}")

    # Load all real SNP data from test split (ground truth)
    print("\nLoading full test dataset...")
    model.setup("test")  # Ensure test dataset is initialized
    test_loader = model.test_dataloader()

    # Collect all test samples
    real_samples = []
    print("Loading all test batches...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to same device as model
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
            real_samples.append(batch)
    real_samples = torch.cat(real_samples, dim=0)

    print(f"\nReal samples shape: {real_samples.shape}")
    print(f"Real samples unique values: {torch.unique(real_samples)}")

    # Generate synthetic sequences
    try:
        with torch.no_grad():
            # Match to real sample shape if not specified
            num_samples = args.num_samples or real_samples.shape[0]
            print(f"\nGenerating {num_samples} synthetic sequences...")
            gen_samples = model.generate_samples(
                num_samples=num_samples, discretize=args.discretize
            )

            # Check for NaN values in generated samples
            if torch.isnan(gen_samples).any():
                print(
                    "Warning: Generated samples contain NaN values. Attempting to fix..."
                )
                gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

            # Save samples
            torch.save(gen_samples, output_dir / "synthetic_sequences.pt")

            # Print statistics
            print(f"\nGen samples shape: {gen_samples.shape}")
            print(f"Gen samples unique values: {torch.unique(gen_samples)}")
            print(f"First gen samples: {gen_samples[:, :1]}")

            # ----------------------------------------------------------------------

            # 1. Analyze MAF distribution
            print("\n1. Analyzing MAF distribution...")
            real_maf = analyze_maf_distribution(
                real_samples, output_dir / "real_maf_distribution.png"
            )
            gen_maf = analyze_maf_distribution(
                gen_samples, output_dir / "gen_maf_distribution.png"
            )

            # Calculate MAF correlation
            maf_corr = torch.corrcoef(
                torch.stack([torch.tensor(real_maf), torch.tensor(gen_maf)])
            )[0, 1]
            print(f"\nMAF correlation between real and generated data: {maf_corr:.4f}")

            # Save MAF statistics
            maf_stats = {
                "real": calculate_maf_stats(real_maf),
                "generated": calculate_maf_stats(gen_maf),
                "correlation": float(maf_corr),
            }

            with open(output_dir / "maf_statistics.json", "w") as f:
                json.dump(maf_stats, f, indent=4)
                print(f"MAF statistics saved to: {output_dir / 'maf_statistics.json'}")

            # 2. Generate comparison plots
            print("\n2. Generating comparison plots...")
            plot_path = output_dir / "sample_comparison.png"
            plot_comparison(real_samples.cpu(), gen_samples.cpu(), plot_path)
            print(f"Comparison plots saved to: {plot_path}")

            # 3. Compute genomic metrics
            print("\n3. Computing genomic metrics...")
            metrics = compute_genomic_metrics(
                real_samples.cpu(), gen_samples.cpu(), output_dir
            )

            # Save metrics to JSON file
            metrics_path = output_dir / "genomic_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Genomic metrics saved to: {metrics_path}")

            # Print summary of metrics
            print("\nGenome Diffusion Model Evaluation Metrics:")
            print("-" * 50)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")
            print("-" * 50)

            # 4. Generate reverse diffusion visualization
            print("\n4. Generating reverse diffusion visualization...")
            plot_path = visualize_reverse_diffusion(model, output_dir)
            print(f"Reverse diffusion visualization saved to: {plot_path}")

            print("\nAll inference tasks completed successfully!")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Sample generation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
