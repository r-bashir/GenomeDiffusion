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
from pathlib import Path
from typing import Dict
import os
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pytorch_lightning as pl
from diffusion.diffusion_model import DiffusionModel



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


def plot_sample_grid(samples, save_path, title, timesteps=None):
    """Plot a grid of samples showing the diffusion process.

    Args:
        samples: Tensor of samples to plot
        save_path: Path to save the plot
        title: Title for the plot
        timesteps: List of timesteps corresponding to each sample
    """
    n_samples = min(samples.shape[0], 500)  # Show at most 500 timesteps
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
        if timesteps is not None:
            axes[i].set_title(f"t = {timesteps[i]}")
        else:
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

    # Print statistics for debugging
    # print("\nData Statistics:")
    # print(
    #    f"Real - shape: {real.shape}, range: [{real.min():.3f}, {real.max():.3f}], mean: {real.mean():.3f}, std: {real.std():.3f}"
    # )
    # print(
    #    f"Generated - shape: {gen.shape}, range: [{gen.min():.3f}, {gen.max():.3f}], mean: {gen.mean():.3f}, std: {gen.std():.3f}"
    # )

    # Check for NaN values
    if np.isnan(gen).any():
        print("Warning: Generated samples contain NaN values!")
        gen = np.nan_to_num(gen, nan=0.0)  # Replace NaN with 0

    # Add channel dimension if missing
    if len(real.shape) == 2:
        real = real.reshape(real.shape[0], 1, -1)
    if len(gen.shape) == 2:
        gen = gen.reshape(gen.shape[0], 1, -1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Real vs Generated Samples Comparison")

    try:
        # Plot sample sequences
        axes[0, 0].plot(real[0].flatten(), label="Real")
        axes[0, 0].plot(gen[0].flatten(), label="Generated")
        axes[0, 0].set_title("Sample Sequence Comparison")
        axes[0, 0].legend()

        # Plot distributions (skip NaN values)
        real_flat = real.flatten()
        gen_flat = gen.flatten()
        axes[0, 1].hist(
            real_flat[~np.isnan(real_flat)], bins=50, alpha=0.5, label="Real"
        )
        axes[0, 1].hist(
            gen_flat[~np.isnan(gen_flat)], bins=50, alpha=0.5, label="Generated"
        )
        axes[0, 1].set_title("Value Distribution")
        axes[0, 1].legend()

        # Plot heatmaps
        first_100 = min(1000, real.shape[-1])
        
        #axes[1, 0].imshow(
            #real[0].reshape(1, -1)[:, :first_100], aspect="auto", cmap="viridis")
        #axes[1, 0].set_title("Real Data Pattern (first 100 positions)")
        #axes[1, 1].imshow(
        #    gen[0].reshape(1, -1)[:, :first_100], aspect="auto", cmap="viridis")
        #axes[1, 1].set_title("Generated Data Pattern (first 100 positions)")

        # Custom colormap: 0 → blue, 0.5 → green, 1 → red
        cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
        # Plot real data
        axes[1, 0].imshow(
            real[0].reshape(1, -1)[:, :first_100], aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Real Data Pattern (first 100 positions)")
        # Plot generated data
        axes[1, 1].imshow(
            gen[0].reshape(1, -1)[:, :first_100], aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Generated Data Pattern (first 100 positions)")       
        plt.tight_layout()
        plt.savefig(save_path)
    except Exception as e:
        print(f"Warning: Error during plotting: {e}")
    finally:
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
    print(f"\nInference results will be saved to: {output_dir}")

    # Initialize model from checkpoint
    try:
        print("\nLoading model from checkpoint...")
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

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        # precision="bf16-mixed",
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Move model to the correct device
    model = model.to(trainer.strategy.root_device)

    # Load real SNP data from test split (ground truth)
    print("\nLoading test dataset...")
    model.setup("test")  # Ensure test dataset is initialized
    print(f"Selecting first batch of test dataset...")
    real_samples = next(iter(model.test_dataloader()))
    print(f"\nReal SNP data sample shape: {real_samples.shape}")
    print(f"Real unique SNP values: {torch.unique(real_samples)}")

    # Generate synthetic SNP sequences through reverse diffusion
    try:
        # print("\nGenerating synthetic SNP sequences via reverse diffusion...")
        with torch.no_grad():
            # Match number of synthetic samples with real data
            num_samples = real_samples.shape[0]
            print(f"\nGenerating {num_samples} synthetic sequences...")

            # Run reverse diffusion to generate synthetic SNP sequences
            # Starting from random noise, gradually denoise to get SNP-like data
            samples = model.generate_samples(num_samples=num_samples)

            # Check for NaN values in generated samples
            if torch.isnan(samples).any():
                print(
                    "Warning: Generated samples contain NaN values. Attempting to fix..."
                )
                samples = torch.nan_to_num(samples, nan=0.0)

            # Save synthetic SNP sequences
            samples_path = output_dir / "synthetic_snp_sequences.pt"
            torch.save(samples.cpu(), samples_path)
            print(f"\nSynthetic SNP data shape: {samples.shape}")
            print(
                f"Synthetic SNP values (comparing to real 0, 0.5, 1.0):\n{torch.unique(samples)}"
            )
            print(
                f"\nSynthetic SNP value range: [{samples.min().item():.3f}, {samples.max().item():.3f}]"
            )
            print(f"\nSamples saved to {samples_path}")

            # Generate comparison plots
            print("\nGenerating comparison plots...")
            plot_path = output_dir / "sample_comparison.png"
            plot_comparison(real_samples.cpu(), samples.cpu(), plot_path)
            print(f"Comparison plots saved to: {plot_path}")

            # Generate reverse diffusion visualization
            print("\nGenerating reverse diffusion visualization...")
            reverse_samples = []
            timesteps = []

            # Start with noise (t=T)
            x = torch.randn((1,) + model._data_shape, device=model.device)
            reverse_samples.append(x)
            timesteps.append(model._forward_diffusion.tmax)

            # Reverse diffusion process
            for t in range(model._forward_diffusion.tmax, 0, -100):
                x = model._reverse_process_step(x, t)
                x = torch.clamp(x, -5.0, 5.0)  # Prevent explosion
                reverse_samples.append(x)
                timesteps.append(t)

            # Final step (t=0)
            x = torch.clamp(x, 0, 1)
            reverse_samples.append(x)
            timesteps.append(0)

            # Plot reverse diffusion
            reverse_samples = torch.cat(reverse_samples, dim=0)
            plot_path = output_dir / "reverse_diffusion.png"
            plot_sample_grid(
                reverse_samples.cpu(),
                plot_path,
                "Reverse Diffusion Process",
                timesteps=timesteps,
            )
            print(f"Reverse diffusion visualization saved to: {plot_path}")

    except Exception as e:
        raise RuntimeError(f"Sample generation failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
