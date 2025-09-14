#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SNPDataset
from src.ddpm import DiffusionModel
from src.forward_diffusion import ForwardDiffusion
from src.mlp import LinearNoisePredictor
from src.utils import load_config, set_seed, setup_logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Test MLPs")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


# Print data statistics
def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray | None:
    """Convert a PyTorch tensor to NumPy (safe for CPU/GPU, with/without grad)."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or None, got {type(tensor)}")
    return tensor.detach().cpu().numpy()


def plot_sample(sample: torch.Tensor, save_path: str) -> None:
    """Plot a SNP sample.

    Args:
        sample: A SNP sample tensor of shape [L], [1, L], or [C=1, L].
        save_path: Path to save the plot image.
    """
    # Remove batch dimension if present ([1, C, L] -> [C, L])
    if sample.dim() == 3 and sample.shape[0] == 1:
        sample = sample.squeeze(0)

    # Remove channel dimension if present ([1, L] -> [L])
    if sample.dim() == 2 and sample.shape[0] == 1:
        sample = sample.squeeze(0)

    # Ensure we now have a 1D sequence
    if sample.dim() != 1:
        raise ValueError(f"Expected sample to be 1D, got shape {tuple(sample.shape)}")

    # Convert to NumPy
    sample_np = torch_to_numpy(sample)

    # Limit number of points for plotting
    plot_points = min(1000, len(sample_np))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(sample_np[:plot_points], "o-", color="blue", alpha=0.7, markersize=3)
    ax.set_title(f"Sample visualization (first {plot_points} SNPs)")
    ax.set_xlabel("SNP Position")
    ax.set_ylabel("Genotype Values")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_tensor_stats(tensor, name):
    """Print statistics about a tensor.

    Args:
        tensor: PyTorch tensor
        name: Name to display for the tensor
    """
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    if tensor.dim() > 1:
        print(f"  Norm (L2): {torch.norm(tensor, p=2).item():.4f}")


def main():
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="MLP")
    logger.info("Starting `run_mlp.py` script.")

    # Set a seed for reproducibility
    set_seed(42)

    # Load configuration
    config = load_config(args.config)

    # Initialize model
    try:
        logger.info("Initializing model...")
        model = DiffusionModel(config)
        logger.info("Model initialized successfully...")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

    # Initialize dataloader
    logger.info("Loading test dataset...")
    model.setup("test")
    test_dataloader = model.test_dataloader()

    # Extract a Batch
    batch = next(iter(test_dataloader))  # Shape: [B, L]
    logger.info(f"Batch shape [B, L]: {batch.shape}, and dim: {batch.dim()}")
    batch = batch.unsqueeze(1).to(device)  # Shape: [B, C=1, L]
    logger.info(f"Batch shape [B, C=1, L]: {batch.shape}, and dim: {batch.dim()}")

    # Loop over batch examples
    for i in range(batch.size(0)):
        # Get example and keep batch dimension
        x0 = batch[i : i + 1]  # Keep batch dimension [1, C, L]

        # Print sample (squeeze batch dim for plotting)
        plot_sample(x0.squeeze(0).squeeze(0), f"before_forward_{i}.png")
        t = model.time_sampler.sample(shape=(x0.shape[0],))
        x0_pred = model.predict_added_noise(x0, t)

        # Print output (squeeze batch dim for plotting)
        plot_sample(x0_pred.squeeze(0).squeeze(0), f"after_forward_{i}.png")

        # Print statistics
        print(f"Example {i}:")
        print(f"  Input shape: {x0.shape}, Output shape: {x0_pred.shape}")
        print(
            "   Are they equal?",
            np.array_equal(x0.detach().numpy(), x0_pred.detach().numpy()),
        )
        print(x0.detach().numpy() == x0_pred.detach().numpy())

    logger.info("Done!")


if __name__ == "__main__":
    main()
