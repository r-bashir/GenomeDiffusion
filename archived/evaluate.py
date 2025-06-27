#!/usr/bin/env python
# coding: utf-8

"""
Comprehensive evaluation script for diffusion models on genomic data.

This script evaluates diffusion models using metrics appropriate for generative models
in genomic data, including:
1. Minor Allele Frequency (MAF) correlation
2. Linkage Disequilibrium (LD) patterns
3. Principal Component Analysis (PCA) visualization
4. Wasserstein distance between distributions
5. Genetic diversity metrics

Example:
    python evaluate_updated.py --checkpoint path/to/checkpoint.ckpt

Outputs:
- Evaluation metrics in JSON format
- Visualization plots
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.diffusion_model import DiffusionModel
from src.evaluation_utils import (
    calculate_genetic_diversity,
    plot_ld_decay,
    run_pca_analysis,
)
from src.inference_utils import (
    calculate_maf,
    calculate_maf_stats,
    compute_genomic_metrics,
    plot_maf_distribution,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion model on genomic data"
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

    output_dir = base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluation results will be saved to: {output_dir}")

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
        # Match to real sample size if not specified
        num_samples = args.num_samples or real_samples.shape[0]
        print(f"\nGenerating {num_samples} synthetic sequences...")
        gen_samples = model.generate_samples(
            num_samples=num_samples, discretize=args.discretize
        )

        # Check for NaN values in generated samples
        if torch.isnan(gen_samples).any():
            print(
                "Warning: Generated samples contain NaN values. Replacing with zeros..."
            )
            gen_samples = torch.nan_to_num(gen_samples, nan=0.0)

        # Save samples
        torch.save(gen_samples, output_dir / "synthetic_sequences.pt")

        # Print statistics
        print(f"\nGenerated samples shape: {gen_samples.shape}")
        print(f"Generated samples unique values: {torch.unique(gen_samples)}")

        # Initialize results dictionary
        results = {}

        # 1. MAF Analysis
        print("\n1. Analyzing Minor Allele Frequency (MAF) distribution...")
        real_maf = calculate_maf(real_samples)
        gen_maf = calculate_maf(gen_samples)

        # Plot MAF distributions and get correlation
        maf_corr = plot_maf_distribution(real_maf, gen_maf, output_dir)

        # Store MAF statistics
        results["maf"] = {
            "real": calculate_maf_stats(real_maf),
            "generated": calculate_maf_stats(gen_maf),
            "correlation": float(maf_corr),
        }

        # 2. Linkage Disequilibrium Analysis
        print("\n2. Analyzing Linkage Disequilibrium (LD) patterns...")
        ld_corr = plot_ld_decay(real_samples, gen_samples, output_dir)

        results["ld"] = {
            "correlation": float(ld_corr) if not np.isnan(ld_corr) else None,
        }

        # 3. PCA Analysis
        print("\n3. Running Principal Component Analysis (PCA)...")
        w_distance, component_distances = run_pca_analysis(
            real_samples, gen_samples, output_dir, n_components=5
        )

        results["pca"] = {
            "wasserstein_distance": float(w_distance),
            "component_distances": [float(d) for d in component_distances],
        }

        # 4. Genetic Diversity Metrics
        print("\n4. Calculating genetic diversity metrics...")
        real_diversity = calculate_genetic_diversity(real_samples)
        gen_diversity = calculate_genetic_diversity(gen_samples)

        # Calculate relative differences
        diversity_diffs = {}
        for key in real_diversity:
            if real_diversity[key] != 0:
                rel_diff = (gen_diversity[key] - real_diversity[key]) / real_diversity[
                    key
                ]
                diversity_diffs[key] = float(rel_diff)
            else:
                diversity_diffs[key] = None

        results["genetic_diversity"] = {
            "real": real_diversity,
            "generated": gen_diversity,
            "relative_difference": diversity_diffs,
        }

        # 5. Genomic Metrics (from inference_utils)
        print("\n5. Computing additional genomic metrics...")
        genomic_metrics = compute_genomic_metrics(
            real_samples.cpu(), gen_samples.cpu(), output_dir
        )
        results["genomic_metrics"] = genomic_metrics

        # Save all results to JSON
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        print(
            f"\nAll evaluation metrics saved to: {output_dir / 'evaluation_metrics.json'}"
        )

        # Print summary of key metrics
        print("\n=== EVALUATION SUMMARY ===")
        print(f"MAF Correlation: {maf_corr:.4f}")
        print(
            f"LD Pattern Correlation: {ld_corr:.4f}"
            if not np.isnan(ld_corr)
            else "LD Pattern Correlation: N/A"
        )
        print(f"PCA Distribution Distance: {w_distance:.4f}")
        print(
            f"Heterozygosity Preservation: {diversity_diffs['expected_heterozygosity']:.2%}"
        )
        print("==========================")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Evaluation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
