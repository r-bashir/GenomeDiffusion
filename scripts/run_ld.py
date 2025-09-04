#!/usr/bin/env python3
"""
Script to investigate Linkage Disequilibrium modeling in GenomeDiffusion.
This script helps diagnose why the model isn't capturing spatial correlations between SNPs.

Usage:
    python investigate_ld.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All imports after path modification
# We need to disable the import-not-at-top lint rule
# ruff: noqa: E402

from src.infer_utils import generate_samples
from src.utils import load_model_from_checkpoint, set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyze_receptive_field(model, real_samples):
    """Analyze the effective receptive field of the model using real data."""
    model.eval()

    # Use a real sample and add a spike in the middle
    sample_idx = len(real_samples) // 2  # Use middle sample
    x = real_samples[sample_idx : sample_idx + 1].clone()  # Shape: [1, 1, seq_length]

    # Add spike in the middle to see receptive field
    seq_length = x.shape[2]  # Get sequence length from tensor shape
    center = seq_length // 2
    original_value = x[0, 0, center].item()
    x[0, 0, center] = 1.0  # Add spike

    # Also test with original sample for comparison
    x_original = real_samples[sample_idx : sample_idx + 1].clone()

    # Forward pass with zero time
    time = torch.zeros(1, device=x.device)

    with torch.no_grad():
        output_with_spike = model(x, time)
        output_original = model(x_original, time)

    # Calculate difference to see influence of spike
    diff = torch.abs(output_with_spike - output_original)[0, 0].cpu().numpy()

    # Find significant differences to estimate receptive field
    significant_indices = np.where(diff > 1e-4)[0]

    if len(significant_indices) > 0:
        receptive_field = significant_indices[-1] - significant_indices[0] + 1
        print(f"🔍 Estimated Receptive Field: {receptive_field} positions")
        print(
            f"📍 Influence range: {significant_indices[0]} to {significant_indices[-1]}"
        )
        print(f"🎯 Spike position: {center}, original value: {original_value:.3f}")
    else:
        print("⚠️  No significant influence detected - model may have issues")

    return diff


def test_positional_sensitivity(model, real_samples):
    """Test if model is sensitive to position changes using real data."""
    model.eval()

    # Take different real samples and test model's response
    num_samples_to_test = min(5, len(real_samples))
    sample_indices = np.linspace(
        0, len(real_samples) - 1, num_samples_to_test, dtype=int
    )

    outputs = []
    time = torch.zeros(1, device=real_samples.device)

    print(f"🎯 Position Sensitivity Test (using {num_samples_to_test} real samples):")

    for i, idx in enumerate(sample_indices):
        x = real_samples[idx : idx + 1]  # Shape: [1, 1, seq_length]

        with torch.no_grad():
            output = model(x, time)
        outputs.append(output[0, 0].cpu().numpy())

        # Show some stats about this sample
        sample_mean = x[0, 0].mean().item()
        sample_std = x[0, 0].std().item()
        print(f"   Sample {i+1}: mean={sample_mean:.3f}, std={sample_std:.3f}")

    # Compare outputs between different real samples
    correlations = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            corr, _ = pearsonr(outputs[i], outputs[j])
            correlations.append(corr)

    avg_corr = np.mean(correlations)
    print(f"   Average correlation between different real samples: {avg_corr:.4f}")

    if avg_corr > 0.95:
        print("   ❌ Model outputs are too similar across different samples")
    elif avg_corr < 0.3:
        print("   ✅ Model is appropriately sensitive to input differences")
    else:
        print("   ⚠️  Model has moderate sensitivity to input differences")

    return outputs


def test_local_correlation_modeling(model, real_samples):
    """Test if model preserves local correlations using real genomic data."""
    model.eval()

    # Use multiple real samples to test correlation preservation
    num_samples_to_test = min(10, len(real_samples))
    sample_indices = np.random.choice(
        len(real_samples), num_samples_to_test, replace=False
    )

    print(f"🧬 Local Correlation Test (using {num_samples_to_test} real samples):")

    all_input_corrs = []
    all_output_corrs = []
    distances = [1, 2, 5, 10]

    time = torch.zeros(1, device=real_samples.device)

    for idx in sample_indices:
        x = real_samples[idx : idx + 1]  # Shape: [1, 1, seq_length]

        with torch.no_grad():
            output = model(x, time)

        input_np = x[0, 0].cpu().numpy()
        output_np = output[0, 0].cpu().numpy()

        # Calculate correlations at different distances for both input and output
        for d in distances:
            # Input correlations
            input_pairs_1 = input_np[:-d] if d < len(input_np) else []
            input_pairs_2 = input_np[d:] if d < len(input_np) else []

            if len(input_pairs_1) > 1:
                input_corr, _ = pearsonr(input_pairs_1, input_pairs_2)
                if not np.isnan(input_corr):
                    all_input_corrs.append((d, input_corr))

            # Output correlations
            output_pairs_1 = output_np[:-d] if d < len(output_np) else []
            output_pairs_2 = output_np[d:] if d < len(output_np) else []

            if len(output_pairs_1) > 1:
                output_corr, _ = pearsonr(output_pairs_1, output_pairs_2)
                if not np.isnan(output_corr):
                    all_output_corrs.append((d, output_corr))

    # Analyze correlations by distance
    for d in distances:
        input_corrs_at_d = [corr for dist, corr in all_input_corrs if dist == d]
        output_corrs_at_d = [corr for dist, corr in all_output_corrs if dist == d]

        if input_corrs_at_d and output_corrs_at_d:
            avg_input_corr = np.mean(input_corrs_at_d)
            avg_output_corr = np.mean(output_corrs_at_d)
            correlation_preservation = (
                avg_output_corr / avg_input_corr if avg_input_corr != 0 else 0
            )

            print(
                f"   Distance {d}: Input={avg_input_corr:.4f}, Output={avg_output_corr:.4f}, Preservation={correlation_preservation:.4f}"
            )

    # Overall assessment
    if all_input_corrs and all_output_corrs:
        overall_input = np.mean([corr for _, corr in all_input_corrs])
        overall_output = np.mean([corr for _, corr in all_output_corrs])
        print(
            f"   Overall: Input correlation={overall_input:.4f}, Output correlation={overall_output:.4f}"
        )

        if overall_output > 0.8 * overall_input:
            print("   ✅ Model preserves local correlations well")
        elif overall_output > 0.5 * overall_input:
            print("   ⚠️  Model partially preserves local correlations")
        else:
            print("   ❌ Model poorly preserves local correlations")

    return all_input_corrs, all_output_corrs


def visualize_model_behavior(model, real_samples, timestep, output_dir):
    """Visualize model behavior by comparing real samples with generated samples.

    Note: This function now properly uses reverse diffusion to generate samples
    instead of directly comparing real data with raw model output (predicted noise).
    """
    model.eval()

    # Select diverse real samples for visualization
    num_samples_to_viz = min(4, len(real_samples))

    # Select samples with different characteristics
    sample_means = [real_samples[i, 0].mean().item() for i in range(len(real_samples))]
    sorted_indices = np.argsort(sample_means)

    # Pick samples from different quartiles
    selected_indices = [
        sorted_indices[0],  # Lowest mean
        sorted_indices[len(sorted_indices) // 3],  # Low-medium mean
        sorted_indices[2 * len(sorted_indices) // 3],  # Medium-high mean
        sorted_indices[-1],  # Highest mean
    ][:num_samples_to_viz]

    # Generate samples using reverse diffusion (proper approach)
    print("🔄 Generating samples via reverse diffusion for comparison...")
    with torch.no_grad():
        # Generate samples using the correct generate_samples function
        generated_samples = generate_samples(
            model,
            num_samples=num_samples_to_viz,
            start_timestep=timestep,
            discretize=False,
        )

    # Test each real sample vs corresponding generated sample
    fig, axes = plt.subplots(
        num_samples_to_viz, 2, figsize=(15, 3 * num_samples_to_viz)
    )
    if num_samples_to_viz == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(selected_indices):
        real_sample = real_samples[idx, 0]  # Shape: [seq_length]
        generated_sample = generated_samples[i, 0]  # Shape: [seq_length]

        # Calculate sample statistics
        real_mean = real_sample.mean().item()
        real_std = real_sample.std().item()
        gen_mean = generated_sample.mean().item()
        gen_std = generated_sample.std().item()

        # Plot real and generated samples
        axes[i, 0].plot(
            real_sample.cpu().numpy(), label="Real Sample", alpha=0.7, color="blue"
        )
        axes[i, 0].set_title(
            f"Real Sample {idx} (mean={real_mean:.3f}, std={real_std:.3f})"
        )
        axes[i, 0].set_ylabel("Allele Frequency")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(-0.1, 0.6)

        axes[i, 1].plot(
            generated_sample.cpu().numpy(),
            label="Generated Sample",
            alpha=0.7,
            color="red",
        )
        axes[i, 1].set_title(
            f"Generated Sample {i} (mean={gen_mean:.3f}, std={gen_std:.3f})"
        )
        axes[i, 1].set_ylabel("Allele Frequency")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim(-0.1, 0.6)

        # Add correlation between real and generated samples
        corr, _ = pearsonr(real_sample.cpu().numpy(), generated_sample.cpu().numpy())
        axes[i, 1].text(
            0.02,
            0.98,
            f"Corr: {corr:.3f}",
            transform=axes[i, 1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    output_path = Path(output_dir) / "real_vs_generated_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"📊 Real vs Generated sample comparison saved as 'real_vs_generated_comparison.png'"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Investigate LD modeling in GenomeDiffusion"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )

    return parser.parse_args()


def main():
    # Parse Arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="LD Correlation")
    logger.info("Starting LD correlation investigation script.")

    # Set global seed
    set_seed(seed=42)

    # Load Model
    try:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, config = load_model_from_checkpoint(args.checkpoint, device)
        logger.info("Model loaded successfully from checkpoint on %s", device)
        logger.info("Model config loaded from checkpoint:")
        print(f"\n{config}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Output directory
    logger.info("Setting up output directory...")
    output_dir = Path(args.checkpoint).parent.parent / "investigate_ld"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory set to: %s", output_dir)

    # Load Dataset, gives shape of [B, L]
    logger.info("Loading real test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Collect all test batches
    real_samples = []
    logger.info("Loading all test batches...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = batch.to(device)
            real_samples.append(batch)
    real_samples = torch.cat(real_samples, dim=0)

    # Ensure shape [num_samples, 1, L] for consistency
    if real_samples.dim() == 2:
        real_samples = real_samples.unsqueeze(1)

    logger.info(
        f"Loaded {len(real_samples)} real test samples with shape: {real_samples.shape}"
    )

    print("🔍 INVESTIGATING LINKAGE DISEQUILIBRIUM MODELING")
    print("=" * 60)

    # Get sequence length from config
    seq_length = config.get("data", {}).get("seq_length", None)
    timestep = config["diffusion"]["timesteps"]

    print(f"🏗️  Model Configuration:")
    print(f"   Sequence length: {seq_length}")
    print(f"   Embedding dim: {config['unet'].get('embedding_dim', 'N/A')}")
    print(f"   Dim mults: {config['unet'].get('dim_mults', 'N/A')}")
    print(f"   Position embeddings: {config['unet'].get('with_pos_emb', 'N/A')}")
    print(f"   Time embeddings: {config['unet'].get('with_time_emb', 'N/A')}")
    print(f"   Diffusion Timesteps: {timestep}")
    print(f"   Edge padding: {config['unet'].get('edge_pad', 'N/A')}")
    print("\n" + "=" * 60)

    # Run investigations using real test data
    print("1️⃣ RECEPTIVE FIELD ANALYSIS")
    analyze_receptive_field(model, real_samples)

    print("\n2️⃣ POSITIONAL SENSITIVITY TEST")
    test_positional_sensitivity(model, real_samples)

    print("\n3️⃣ LOCAL CORRELATION MODELING TEST")
    test_local_correlation_modeling(model, real_samples)

    print("\n4️⃣ REAL DATA VISUALIZATION")
    visualize_model_behavior(model, real_samples, timestep, output_dir)

    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS:")

    if not config["unet"].get("with_pos_emb", False):
        print("❌ CRITICAL: Enable positional embeddings (with_pos_emb: True)")

    if len(config["unet"].get("dim_mults", [])) < 3:
        print("⚠️  Consider more downsampling levels for larger receptive field")

    if config["unet"].get("embedding_dim", 0) < 32:
        print("⚠️  Consider increasing embedding_dim for better feature capacity")

    print(f"\n✅ Investigation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
