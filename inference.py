#!/usr/bin/env python
# coding: utf-8

"""Script for generating and analyzing samples from trained SNP diffusion models.

Examples:
    # Generate default number of samples (from config)
    python inference.py --config config.yaml --checkpoint path/to/checkpoint.ckpt

    # Generate specific number of samples
    python inference.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --num_samples 100

Generated outputs are saved in the 'inference' directory, including:
- Generated samples (.pt file)
- Sample comparisons with real data
- Statistical analysis
- Visualization plots
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from diffusion.diffusion_model import DiffusionModel
from diffusion.dataset import SNPDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate samples using trained SNP diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
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
        help="Number of samples to generate (overrides config)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def plot_sample_grid(samples, save_path, title):
    """Plot a grid of samples showing the diffusion process."""
    n_samples = min(samples.shape[0], 10)  # Show at most 10 timesteps
    seq_length = samples.shape[-1]

    # Create figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 2 * n_samples))
    fig.suptitle(title)

    # If only one sample, axes won't be an array
    if n_samples == 1:
        axes = [axes]

    # Plot each sample
    for i in range(n_samples):
        axes[i].imshow(samples[i, 0, :].reshape(1, -1), aspect="auto", cmap="viridis")
        axes[i].set_title(f"Step {i}")
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(real_samples, generated_samples, save_path):
    """Plot comparison between real and generated samples."""
    # Convert to numpy for plotting
    real = real_samples.cpu().numpy()
    gen = generated_samples.cpu().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Real vs Generated Samples Comparison")

    # Plot sample sequences
    axes[0, 0].plot(real[0, 0, :], label="Real")
    axes[0, 0].plot(gen[0, 0, :], label="Generated")
    axes[0, 0].set_title("Sample Sequence Comparison")
    axes[0, 0].legend()

    # Plot distributions
    axes[0, 1].hist(real.flatten(), bins=50, alpha=0.5, label="Real")
    axes[0, 1].hist(gen.flatten(), bins=50, alpha=0.5, label="Generated")
    axes[0, 1].set_title("Value Distribution")
    axes[0, 1].legend()

    # Plot heatmaps
    axes[1, 0].imshow(real[0, 0, :100].reshape(10, -1), aspect="auto")
    axes[1, 0].set_title("Real Data Pattern (first 100 positions)")
    axes[1, 1].imshow(gen[0, 0, :100].reshape(10, -1), aspect="auto")
    axes[1, 1].set_title("Generated Data Pattern (first 100 positions)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    """Main inference function."""
    # Validate input files
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Load configuration
    config = load_config(args.config)

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    if "checkpoints" in str(checkpoint_path):
        base_dir = (
            checkpoint_path.parent.parent
        )  # Go up two levels: from checkpoint file to run directory
    else:
        base_dir = checkpoint_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create inference directory for generated samples and comparisons
    output_dir = base_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated samples and comparisons will be saved to: {output_dir}")

    # Load some real samples for comparison
    print("Loading test dataset for comparison...")
    test_dataset = SNPDataset(config["data"]["test_data_path"])
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    real_samples = next(iter(test_loader))

    # Initialize model from checkpoint
    try:
        print("Loading model from checkpoint...")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            strict=True,
            config=config,
        )
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully on {next(model.parameters()).device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Generate samples
    print("Generating samples...")
    try:
        with torch.no_grad():  # Disable gradient computation for inference
            # Get number of samples from config or args
            num_samples = args.num_samples or config["training"].get("num_samples", 10)

            # Generate samples
            print(f"Generating {num_samples} samples...")
            samples = model.generate_samples(num_samples=num_samples)

            # Save samples
            samples_path = output_dir / "generated_samples.pt"
            torch.save(samples, samples_path)
            print(f"Generated samples shape: {samples.shape}")
            print(f"Samples saved to {samples_path}")

            # Print statistics comparison
            print("\nStatistics Comparison:")
            print("Generated Samples:")
            print(f"  Mean: {samples.mean().item():.4f}")
            print(f"  Std: {samples.std().item():.4f}")
            print(f"  Min: {samples.min().item():.4f}")
            print(f"  Max: {samples.max().item():.4f}")

            print("\nReal Samples:")
            print(f"  Mean: {real_samples.mean().item():.4f}")
            print(f"  Std: {real_samples.std().item():.4f}")
            print(f"  Min: {real_samples.min().item():.4f}")
            print(f"  Max: {real_samples.max().item():.4f}")

            # Generate comparison plots
            print("\nGenerating comparison plots...")
            plot_path = output_dir / "sample_comparison.png"
            plot_comparison(real_samples, samples, plot_path)
            print(f"Comparison plots saved to: {plot_path}")

            # Generate reverse diffusion visualization
            print("\nGenerating reverse diffusion visualization...")
            reverse_samples = []
            x = torch.randn((1,) + model._data_shape, device=model.device)
            reverse_samples.append(x.cpu())

            for t in range(model._forward_diffusion.tmax, 0, -100):
                x = model._reverse_process_step(x, t)
                x = torch.clamp(x, -5.0, 5.0)  # Prevent explosion
                reverse_samples.append(x.cpu())

            # Final normalization
            x = torch.clamp(x, 0, 1)
            reverse_samples.append(x.cpu())

            # Plot reverse diffusion
            reverse_samples = torch.cat(reverse_samples, dim=0)
            plot_path = output_dir / "reverse_diffusion.png"
            plot_sample_grid(
                reverse_samples, plot_path, "Reverse Diffusion Process (t=T to 0)"
            )
            print(f"Reverse diffusion visualization saved to: {plot_path}")

    except Exception as e:
        raise RuntimeError(f"Sample generation failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
