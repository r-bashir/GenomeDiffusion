#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from diffusion import SNPDataset
from test_models import DDPM, UNet1D

# Set global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_models():
    """Test ForwardDiffusion and UNet1D with real SNP data from config."""
    print(f"\nUsing device: {device}")
    
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
    
    # Analyze unique values in the data
    unique_values = torch.unique(batch)
    print(f"\nUnique values in data: {unique_values.tolist()}")
    
    # Count occurrences of each value
    value_counts = {}
    for value in [0.0, 0.5, 1.0, 9.0]:
        count = (batch == value).sum().item()
        percentage = (count / batch.numel()) * 100
        value_counts[value] = (count, percentage)
    
    print("\nValue distribution:")
    for value, (count, percentage) in value_counts.items():
        print(f"Value {value:.1f}: {count} occurrences ({percentage:.2f}%)")
    
    # Test ForwardDiffusion
    print("\nTesting ForwardDiffusion...")
    forward_diffusion = DDPM(
        num_diffusion_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # Test different timesteps
    timesteps = [0, 250, 500, 750, 999]
    # Create a new figure with high DPI for better quality
    plt.figure(figsize=(15, 5), dpi=100, facecolor='white')
    
    # Prepare input (ensure [B, C, L] format for both models)
    x0 = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    
    # Move forward diffusion tensors to device
    forward_diffusion._alphas = forward_diffusion._alphas.to(device)
    forward_diffusion._sigmas = forward_diffusion._sigmas.to(device)
    
    # For visualization, only use the first sequence
    x0_single = x0[0].unsqueeze(0)  # [1, 1, seq_len]
    print(f"\nVisualization shape: {x0_single.shape}")
    
    for i, t in enumerate(timesteps):
        # Sample noise
        eps = torch.randn_like(x0_single)
        t_tensor = torch.tensor([t], device=device)
        
        # Apply forward diffusion
        xt = forward_diffusion.sample(x0_single, t_tensor, eps)
        print(f"\nt={t}:")
        print(f"xt shape: {xt.shape}")
        plot_data = xt[0].detach().numpy()
        print(f"plot_data shape: {plot_data.shape}")
        print(f"plot_data range: [{plot_data.min():.4f}, {plot_data.max():.4f}]")
        
        # Plot with grid and labels
        plt.subplot(1, len(timesteps), i+1)
        plt.plot(plot_data[0], linewidth=1, color='blue', alpha=0.8)
        plt.title(f't={t}', pad=10)
        if t == 0:
            plt.ylim(-0.5, 9.5)  # Original data range with small padding
        else:
            # For noisy timesteps, use dynamic range based on data
            margin = 0.5
            plt.ylim(plot_data[0].min() - margin, plot_data[0].max() + margin)
        plt.grid(True, alpha=0.3)
        if i == 0:  # Add y-label only for the first subplot
            plt.ylabel('SNP Value')
        if i == len(timesteps)//2:  # Add x-label for the middle subplot
            plt.xlabel('Position')
    
    # Add title and adjust layout
    plt.suptitle('Forward Diffusion Process on SNP Data', y=1.05)
    plt.tight_layout()
    
    # Save figure with high quality
    plt.savefig('forward_diffusion_snp.png', bbox_inches='tight', pad_inches=0.2, dpi=100)
    plt.close('all')  # Close all figures to free memory
    
    print("Forward diffusion test complete! Check forward_diffusion_snp.png")
    test_times = torch.tensor([0, 500, 999], device=device)
    print(f"Alpha values at t=0, t=500, t=999: {forward_diffusion.alpha(test_times).tolist()}")
    print(f"Sigma values at t=0, t=500, t=999: {forward_diffusion.sigma(test_times).tolist()}")
    
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
    ).to(device)
    
    # Reuse the same input tensor from forward diffusion
    x = x0  # [B, 1, seq_len]
    
    # Use same timesteps for each batch element
    t = torch.tensor([500] * batch.shape[0], dtype=torch.float32, device=device)  # Use middle timestep
    
    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    
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
