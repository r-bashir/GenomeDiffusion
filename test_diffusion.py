#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid Wayland warning
import matplotlib.pyplot as plt
import yaml
from dataset import SNPDataset
from torch.utils.data import DataLoader
from test_models import ForwardDiffusion, UNet1D

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_models():
    """Test ForwardDiffusion and UNet1D with real SNP data from config."""
    # Load config
    config = load_config()
    input_path = config.get('input_path')
    print("\nInitializing dataset...")
    dataset = SNPDataset(input_path)
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"\nBatch shape: {batch.shape}")
    
    # Test ForwardDiffusion
    print("\nTesting ForwardDiffusion...")
    forward_diffusion = ForwardDiffusion(
        num_diffusion_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Test different timesteps
    timesteps = [0, 250, 500, 750, 999]
    plt.figure(figsize=(15, 5))
    
    # Select first sequence from batch
    x0 = batch[0].unsqueeze(0)  # [1, seq_len]
    
    for i, t in enumerate(timesteps):
        # Sample noise
        eps = torch.randn_like(x0)
        t_tensor = torch.tensor([t])
        
        # Apply forward diffusion
        xt = forward_diffusion.sample(x0, t_tensor, eps)
        
        # Plot
        plt.subplot(1, len(timesteps), i+1)
        plt.plot(xt[0].detach().numpy())
        plt.title(f't={t}')
        plt.ylim(-3, 3)
    
    plt.suptitle('Forward Diffusion Process on SNP Data')
    plt.tight_layout()
    plt.savefig('forward_diffusion_snp.png')
    plt.close()
    
    print("Forward diffusion test complete! Check forward_diffusion_snp.png")
    print(f"Alpha values at t=0, t=500, t=999: {forward_diffusion.alpha(torch.tensor([0, 500, 999])).tolist()}")
    print(f"Sigma values at t=0, t=500, t=999: {forward_diffusion.sigma(torch.tensor([0, 500, 999])).tolist()}")
    
    # Test UNet with memory optimizations from config
    print("\nTesting UNet1D...")
    config = load_config()
    unet_config = config.get('unet', {})
    
    unet = UNet1D(
        embedding_dim=unet_config.get('embedding_dim', 32),  # Memory-optimized: 32 instead of 64
        dim_mults=unet_config.get('dim_mults', [1, 2, 4]),  # Memory-optimized: removed 8
        channels=unet_config.get('channels', 1),
        with_time_emb=unet_config.get('with_time_emb', True),
        resnet_block_groups=unet_config.get('resnet_block_groups', 8)
    )
    
    # Prepare input (ensure [B, C, L] format)
    x = batch.unsqueeze(1)  # [B, 1, seq_len]
    t = torch.randint(0, 1000, (batch.shape[0],))
    
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        output = unet(x, t)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        assert x.shape == output.shape, "UNet output shape doesn't match input shape!"
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        has_grad = all(p.grad is not None for p in unet.parameters())
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in unet.parameters()]))
        print(f"Gradients present: {has_grad}")
        print(f"Gradient norm: {grad_norm:.4f}")
        print("\nUNet test successful!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOUT OF MEMORY ERROR!")
            print("Try reducing batch_size or model dimensions")
        else:
            print(f"\nError during UNet test: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error during UNet test: {str(e)}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_models()
