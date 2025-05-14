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

from src.diffusion_model import DiffusionModel
from src.inference_utils import (
    analyze_maf_distribution,
    calculate_maf_stats,
    compare_maf_distributions,
    compare_samples,
    generate_samples_mid_step,
    visualize_reverse_denoising,
    visualize_samples,
)
from src.utils import set_seed


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

    # Set global seed for reproducibility
    set_seed(42)

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

    # Real samples
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

    # Generated samples
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

    # 1. Sample Analysis (Fully Denoised, T=0)
    print("\n1. Performing Sample Analysis (Fully Denoised, T=0)...")

    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # compare_samples(
    #     real_samples,
    #     gen_samples,
    #     output_dir / "compare_samples.png",
    #     genotype_values=[0.0, 0.5, 1.0],
    # )

    # For scaled genotype values (0.0, 0.25, 0.5)
    compare_samples(
        real_samples,
        gen_samples,
        output_dir / "compare_samples.png",
        genotype_values=[0.0, 0.25, 0.5],
    )

    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # visualize_samples(
    #     real_samples,
    #     gen_samples,
    #     output_dir / "visualize_samples.png",
    #     max_seq_len=1000,
    #     genotype_values=[0.0, 0.5, 1.0],
    # )

    # For scaled genotype values (0.0, 0.25, 0.5)
    visualize_samples(
        real_samples,
        gen_samples,
        output_dir / "visualize_samples.png",
        max_seq_len=1000,
        genotype_values=[0.0, 0.25, 0.5],
    )

    # 2. Sample Analysis (Mid-denoised,T=500)
    print("\n2. Performing Sample Analysis (Mid-denoised,T=500)...")

    T = 500
    gen_samples_mid = generate_samples_mid_step(
        model,
        num_samples=num_samples,
        mid_timestep=T,  # Middle of diffusion process
        denoise_step=10,  # Choose denoise step
        discretize=args.discretize,
    )

    compare_samples(
        real_samples,
        gen_samples_mid,
        output_dir / f"compare_samples_t{T}.png",
        genotype_values=[0.0, 0.25, 0.5],
    )
    visualize_samples(
        real_samples,
        gen_samples_mid,
        output_dir / f"visualize_samples_t{T}.png",
        max_seq_len=1000,
        genotype_values=[0.0, 0.25, 0.5],
    )

    # 3. Visualize Reverse Denoising
    print("\n3. Visualizing Reverse Denoising...")
    visualize_reverse_denoising(
        model,
        output_dir,
        start_timestep=100,  # None means full noise
        step_size=100,
        num_samples=1,
        save_prefix="viz_",
        discretize=False,
        seed=42,
    )

    # 4. MAF Analysis (Fully Denoised, T=0)
    print("\n4. Performing MAF Analysis (Fully Denoised, T=0)...")

    # Analyze real data MAF (scaled genotype values)
    real_maf, _ = analyze_maf_distribution(
        real_samples,
        output_dir / "maf_real_distribution.png",
        genotype_values=[0.0, 0.25, 0.5],
        max_value=0.5,
    )
    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # real_maf, _ = analyze_maf_distribution(
    #     real_samples,
    #     output_dir / "maf_real_distribution.png",
    # )

    # Analyze generated data MAF (scaled genotype values)
    gen_maf, _ = analyze_maf_distribution(
        gen_samples,
        output_dir / "maf_gen_distribution.png",
        genotype_values=[0.0, 0.25, 0.5],
        max_value=0.5,
    )
    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # gen_maf, _ = analyze_maf_distribution(
    #     gen_samples,
    #     output_dir / "maf_gen_distribution.png",
    # )

    # Calculate MAF stats for real and generated data (scaled genotype values)
    real_maf_stats = calculate_maf_stats(real_maf, genotype_values=[0.0, 0.25, 0.5])
    gen_maf_stats = calculate_maf_stats(gen_maf, genotype_values=[0.0, 0.25, 0.5])
    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # real_maf_stats = calculate_maf_stats(real_maf)
    # gen_maf_stats = calculate_maf_stats(gen_maf)

    # Compare MAF distributions and get correlation (scaled genotype values)
    maf_corr = compare_maf_distributions(
        real_maf, gen_maf, output_dir, genotype_values=[0.0, 0.25, 0.5], max_value=0.5
    )
    # For original genotype values (0.0, 0.5, 1.0), uncomment below:
    # maf_corr = compare_maf_distributions(real_maf, gen_maf, output_dir)
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

    # 5. MAF Analysis (Mid-denoised, T=500)
    print("\n5. Performing MAF Analysis (Mid-denoised, T=500)...")

    maf_stats_mid = calculate_maf_stats(
        gen_samples_mid, genotype_values=[0.0, 0.25, 0.5]
    )
    print("\nFrequency analysis for mid-diffusion samples:")
    print(
        f"Raw frequency range: [{maf_stats_mid['min_freq']:.3f}, {maf_stats_mid['max_freq']:.3f}]"
    )
    print(f"Number of 0.25 frequencies: {maf_stats_mid['num_half_freq']}")
    print(
        f"MAF range: [{maf_stats_mid['min_maf']:.3f}, {maf_stats_mid['max_maf']:.3f}]"
    )
    print(f"Number of MAF = 0.25: {maf_stats_mid['num_half_maf']}")

    # 6. Combined MAF comparison and summary
    print("\n6. MAF Comparison Summary")
    print(
        f"Real:          #0.25 = {real_maf_stats['num_half_maf']}, MAF range: [{real_maf_stats['min_maf']:.3f}, {real_maf_stats['max_maf']:.3f}]"
    )
    print(
        f"Generated:     #0.25 = {gen_maf_stats['num_half_maf']}, MAF range: [{gen_maf_stats['min_maf']:.3f}, {gen_maf_stats['max_maf']:.3f}]"
    )
    print(
        f"Mid-diffusion: #0.25 = {maf_stats_mid['num_half_maf']}, MAF range: [{maf_stats_mid['min_maf']:.3f}, {maf_stats_mid['max_maf']:.3f}]"
    )

    maf_stats_combined = {
        "real": real_maf_stats,
        "generated": gen_maf_stats,
        "mid_diffusion": maf_stats_mid,
        "correlation": float(maf_corr),
    }
    with open(output_dir / "maf_statistics.json", "w") as f:
        json.dump(maf_stats_combined, f, indent=4)
        print(f"MAF statistics saved to: {output_dir / 'maf_statistics.json'}")

    print("\nAll inference tasks completed successfully!")


if __name__ == "__main__":
    main()
