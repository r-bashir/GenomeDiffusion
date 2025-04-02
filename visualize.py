#!/usr/bin/env python
# coding: utf-8

"""Visualization script for diffusion model samples and process.

python visualize.py \
    --config config.yaml \
    --checkpoint path/to/checkpoint.ckpt \
    --samples path/to/generated_samples.pt \
    --output_dir visualizations

"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diffusion.diffusion_model import DiffusionModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--samples",
        type=str,
        required=True,
        help="Path to generated samples (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Output directory for plots",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def plot_sample_grid(samples: torch.Tensor, output_path: str, title: str = "Generated Samples"):
    """Plot a grid of samples.
    
    Args:
        samples: Tensor of shape [N, C, seq_len]
        output_path: Path to save plot
        title: Plot title
    """
    num_samples = min(16, samples.shape[0])  # Show max 16 samples
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_samples:
                # Plot sample as heatmap
                im = axes[i, j].imshow(
                    samples[idx].cpu().numpy(),
                    aspect='auto',
                    cmap='viridis',
                    interpolation='nearest'
                )
                axes[i, j].set_title(f"Sample {idx + 1}")
            axes[i, j].axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Value')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_diffusion_process(model: DiffusionModel, output_dir: str):
    """Visualize the forward and reverse diffusion process.
    
    Args:
        model: Trained diffusion model
        output_dir: Output directory for plots
    """
    # Generate a single sample and visualize the diffusion process
    with torch.no_grad():
        # Start with random noise
        x = torch.randn((1,) + model._data_shape, device=model.device)
        
        # Forward diffusion process
        forward_samples = []
        for t in range(0, model._forward_diffusion.tmax + 1, 100):
            xt = model._forward_diffusion.q_sample(x, t)
            forward_samples.append(xt.cpu())
        
        # Plot forward diffusion
        forward_samples = torch.cat(forward_samples, dim=0)
        plot_sample_grid(
            forward_samples,
            os.path.join(output_dir, "forward_diffusion.png"),
            "Forward Diffusion Process (t=0 to T)"
        )
        
        # Reverse diffusion process
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
        plot_sample_grid(
            reverse_samples,
            os.path.join(output_dir, "reverse_diffusion.png"),
            "Reverse Diffusion Process (t=T to 0)"
        )


def main(args):
    """Main visualization function."""
    # Load configuration and model
    config = load_config(args.config)
    model = DiffusionModel.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        hparams=config,
    )
    model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot generated samples
    samples = torch.load(args.samples, map_location="cpu")
    plot_sample_grid(
        samples,
        os.path.join(output_dir, "generated_samples.png"),
        "Generated Samples"
    )

    # Visualize diffusion process
    visualize_diffusion_process(model, str(output_dir))


if __name__ == "__main__":
    args = parse_args()
    main(args)
