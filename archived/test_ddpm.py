#!/usr/bin/env python
# coding: utf-8

import matplotlib
import torch
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import DDPM, SNPDataset

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_ddpm():
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load config
    config = load_config("config.yaml")
    input_path = config.get("input_path")
    print("\nInitializing dataset...")

    # Initialize dataset
    dataset = SNPDataset(input_path)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Get batch
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"Batch shape [B, seq_len]: {batch.shape}")

    # Prepare input (ensure [B, C, L] format for both models)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    print(f"Batch shape [B, C, seq_len]: {batch.shape}")

    # ------------------ Test DDPM
    print("\nTesting DDPM...")
    forward_diffusion = DDPM(
        num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02
    )

    # Move forward diffusion tensors to device
    forward_diffusion._alphas = forward_diffusion._alphas.to(device)
    forward_diffusion._sigmas = forward_diffusion._sigmas.to(device)

    # Test different timesteps
    timesteps = [0, 250, 500, 750, 999]

    # 1. Test with single sample for visualization
    x0_single = batch[0:1].to(device)  # Take first sample [1, seq_len]
    print(f"1. Single sample shape: {x0_single.shape}")

    # Create figure for visualization
    plt.figure(figsize=(15, 5))

    for i, t in enumerate(timesteps):
        # Sample noise
        eps = torch.randn_like(x0_single)
        t_tensor = torch.tensor([t], device=device)

        # Apply forward diffusion
        xt = forward_diffusion.sample(x0_single, t_tensor, eps)

        print(f"\nt_tensor.shape: {t_tensor.shape}")
        print(f"eps.shape: {eps.shape}")
        print(f"xt.shape: {xt.shape}")

        # Plot results
        plt.subplot(1, len(timesteps), i + 1)
        plt.plot(xt[0, 0].cpu().detach().numpy(), linewidth=1, color="blue", alpha=0.8)
        plt.title(f"t={t}")

    plt.tight_layout()
    plt.savefig("ddpm_timesteps.png")
    plt.close()

    # 2. Test with full batch and different timesteps
    print("\n2. Testing with full batch...")
    batch = batch.to(device)

    # Assign different timesteps to each batch element
    batch_size = batch.shape[0]
    # TODO: use a time sampler instead of linspace
    varied_timesteps = torch.linspace(0, 999, batch_size).long().to(device)
    print(f"Varied timesteps shape: {varied_timesteps.shape}")

    # Sample noise
    eps = torch.randn_like(batch)

    # Apply forward diffusion
    xt_batch = forward_diffusion.sample(batch, varied_timesteps, eps)
    print(f"Output batch shape: {xt_batch.shape}")

    # 3. Validate noise levels
    print("\n3. Validating noise levels:")
    t_start = torch.tensor([0], device=device)
    t_mid = torch.tensor([500], device=device)
    t_end = torch.tensor([999], device=device)

    # Use same noise for fair comparison
    same_noise = torch.randn_like(x0_single)

    # Get samples at different timesteps
    x_start = forward_diffusion.sample(x0_single, t_start, same_noise)
    x_mid = forward_diffusion.sample(x0_single, t_mid, same_noise)
    x_end = forward_diffusion.sample(x0_single, t_end, same_noise)

    # Calculate signal-to-noise ratio
    # For 3D tensors, calculate variance along the sequence dimension
    if len(x_start.shape) == 3:
        # Calculate variance along the sequence dimension (dim=2)
        snr_start = (
            x_start.var(dim=2).mean() / (x_start - x0_single).var(dim=2).mean()
        ).item()
        snr_mid = (
            x_mid.var(dim=2).mean() / (x_mid - x0_single).var(dim=2).mean()
        ).item()
        snr_end = (
            x_end.var(dim=2).mean() / (x_end - x0_single).var(dim=2).mean()
        ).item()
    else:
        # Original calculation for 2D tensors
        snr_start = (x_start.var() / (x_start - x0_single).var()).item()
        snr_mid = (x_mid.var() / (x_mid - x0_single).var()).item()
        snr_end = (x_end.var() / (x_end - x0_single).var()).item()

    print(f"SNR at t=0: {snr_start:.4f}")
    print(f"SNR at t=500: {snr_mid:.4f}")
    print(f"SNR at t=999: {snr_end:.4f}")


if __name__ == "__main__":
    test_ddpm()
