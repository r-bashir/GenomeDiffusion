#!/usr/bin/env python
# coding: utf-8

"""Test script to verify device handling in the DiffusionModel."""

import torch
import yaml

from diffusion_model import DiffusionModel


def test_device_handling():
    """Test if device handling is properly implemented in the DiffusionModel."""
    print("Testing device handling in DiffusionModel...")

    # Check available devices
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    print("Initializing DiffusionModel...")
    model = DiffusionModel(config)
    print(f"Model device (from Lightning): {model.device}")

    # Create a sample batch
    batch_size = 2
    seq_length = config["data"]["seq_length"]
    channels = config["unet"]["channels"]

    # Create sample on CPU first
    print("\nCreating sample batch on CPU...")
    sample_batch = torch.randn(batch_size, channels, seq_length)
    print(f"Sample batch device: {sample_batch.device}")

    # Test prepare_batch
    print("\nTesting _prepare_batch...")
    prepared_batch = model._prepare_batch(sample_batch)
    print(f"Prepared batch device: {prepared_batch.device}")

    # Test forward_step
    print("\nTesting forward_step...")
    with torch.no_grad():
        output = model.forward_step(sample_batch)
    print(f"Output device: {output.device}")

    # Test compute_loss
    print("\nTesting compute_loss...")
    with torch.no_grad():
        loss = model.compute_loss(sample_batch)
    print(f"Loss device: {loss.device}")
    print(f"Loss value: {loss.item()}")

    # Test DDPM device handling
    print("\nTesting DDPM device handling...")
    # Create sample tensors
    x0 = torch.randn(2, channels, seq_length)
    t = torch.ones(2, dtype=torch.int32)
    eps = torch.randn_like(x0)

    # Test DDPM sample method
    xt = model._forward_diffusion.sample(x0, t, eps)
    print(f"DDPM input device: {x0.device}")
    print(f"DDPM output device: {xt.device}")

    # Move tensors to GPU if available and test again
    if torch.cuda.is_available():
        print("\nTesting with tensors on GPU...")
        x0_gpu = x0.to("cuda")
        t_gpu = t.to("cuda")
        eps_gpu = eps.to("cuda")
        xt_gpu = model._forward_diffusion.sample(x0_gpu, t_gpu, eps_gpu)
        print(f"DDPM GPU input device: {x0_gpu.device}")
        print(f"DDPM GPU output device: {xt_gpu.device}")

    print("\nDevice test completed successfully!")


if __name__ == "__main__":
    test_device_handling()
