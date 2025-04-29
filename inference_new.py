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
from diffusion.inference_utils import (
    compare_samples,
    visualize_samples,
    calculate_maf_stats,
    analyze_maf_distribution,
    compare_maf_distributions,
    generate_mid_noise_samples,
    visualize_mid_noise_diffusion,
)


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

    # Parse Arguments
    args = parse_args()

    # === Prepare Environment ===
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
            # Move batch to device and add channel dimension
            batch = batch.to(device).unsqueeze(1)
            real_samples.append(batch)
    real_samples = torch.cat(real_samples, dim=0)

    print(f"\nReal samples shape: {real_samples.shape}")
    print(f"Real samples unique values: {torch.unique(real_samples)}")

    # Sample Generation
    num_samples = args.num_samples or real_samples.shape[0]
    print(f"Generating {num_samples} synthetic sequences from full noise...")
    with torch.no_grad():
        gen_samples = model.generate_samples(
            num_samples=num_samples, denoise_step=10, discretize=args.discretize
        )

        # Check for NaN values in generated samples
        if torch.isnan(gen_samples).any():
            print("Warning: Generated samples contain NaN values. Attempting to fix...")
            gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

        # Save samples
        torch.save(gen_samples, output_dir / "generated_samples.pt")

        # Print statistics
        print(f"Generated samples shape: {gen_samples.shape}")
        print(f"Generated samples unique values: {torch.unique(gen_samples)}")

    # === Perform Inference ===

    # 1. Sample Analysis
    print("\n1. Performing Sample Analysis...")
    compare_samples(
        real_samples,
        gen_samples,
        output_dir / "compare_samples.png",
    )
    visualize_samples(
        real_samples,
        gen_samples,
        output_dir / "visualize_samples.png",
        max_seq_len=1000,
    )

    # 2. MAF Analysis
    print("\n2. Performing MAF Analysis...")

    # Analyze real data MAF
    real_maf, _ = analyze_maf_distribution(
        real_samples,
        output_dir / "maf_real_distribution.png",
    )

    # Analyze generated data MAF
    gen_maf, _ = analyze_maf_distribution(
        gen_samples,
        output_dir / "maf_gen_distribution.png",
    )

    # Calculate MAF stats for real and generated data
    real_maf_stats = calculate_maf_stats(real_maf)
    gen_maf_stats = calculate_maf_stats(gen_maf)

    # Compare MAF distributions and get correlation
    maf_corr = compare_maf_distributions(real_maf, gen_maf, output_dir)
    print(f"MAF correlation between real and generated data: {maf_corr:.4f}")

    # Save MAF statistics
    maf_stats = {
        "real": real_maf_stats,
        "generated": gen_maf_stats,
        "correlation": float(maf_corr),
    }

    with open(output_dir / "maf_statistics.json", "w") as f:
        json.dump(maf_stats, f, indent=4)
        print(f"MAF statistics saved to: {output_dir / 'maf_statistics.json'}")

    # 4. Mid-Diffusion Analysis (single call, clean comparison)
    print("\n4. Analyzing Mid-Diffusion (t=500) Generated Samples...")

    # Generate mid-diffusion samples (t=500)
    mid_diff_samples = generate_mid_noise_samples(
        model,
        num_samples=num_samples,
        mid_timestep=500,  # Middle of diffusion process
        denoise_step=10,  # Choose denoise step
        discretize=args.discretize,
    )

    # Visualize mid-diffusion samples
    print("\nComparing mid-diff samples...")
    compare_samples(
        real_samples,
        mid_diff_samples,
        output_dir / "compare_midiff_samples.png",
    )
    
    print("\nVisualizing mid-diff samples...")
    visualize_samples(
        real_samples,
        mid_diff_samples,
        output_dir / "visualize_midiff_samples.png",
        max_seq_len=1000,
    )

    # Calculate MAF stats for mid-diffusion samples
    print("\nCalculating MAF statistics for mid-diffusion samples...")
    mid_diff_maf_stats = calculate_maf_stats(mid_diff_samples)
    print("\nFrequency analysis for mid-diffusion samples:")
    print(
        f"Raw frequency range: [{mid_diff_maf_stats['min_freq']:.3f}, {mid_diff_maf_stats['max_freq']:.3f}]"
    )
    print(f"Number of 0.5 frequencies: {mid_diff_maf_stats['num_half_freq']}")
    print(
        f"MAF range: [{mid_diff_maf_stats['min_maf']:.3f}, {mid_diff_maf_stats['max_maf']:.3f}]"
    )
    print(f"Number of MAF = 0.5: {mid_diff_maf_stats['num_half_maf']}")

    # --- Combined MAF comparison and summary ---
    print("\n=== MAF Comparison Summary ===")
    print(
        f"Real:    #0.5 = {real_maf_stats['num_half_maf']}, MAF range: [{real_maf_stats['min_maf']:.3f}, {real_maf_stats['max_maf']:.3f}]"
    )
    print(
        f"Generated:       #0.5 = {gen_maf_stats['num_half_maf']}, MAF range: [{gen_maf_stats['min_maf']:.3f}, {gen_maf_stats['max_maf']:.3f}]"
    )
    print(
        f"Mid-diffusion:   #0.5 = {mid_diff_maf_stats['num_half_maf']}, MAF range: [{mid_diff_maf_stats['min_maf']:.3f}, {mid_diff_maf_stats['max_maf']:.3f}]"
    )

    maf_stats_combined = {
        "real": real_maf_stats,
        "generated": gen_maf_stats,
        "mid_diffusion": mid_diff_maf_stats,
        "correlation": float(maf_corr),
    }
    with open(output_dir / "maf_statistics.json", "w") as f:
        json.dump(maf_stats_combined, f, indent=4)
        print(f"MAF statistics saved to: {output_dir / 'maf_statistics.json'}")

    print("\nAll inference tasks completed successfully!")


if __name__ == "__main__":
    main()
