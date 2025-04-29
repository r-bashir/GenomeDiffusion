#!/usr/bin/env python
# coding: utf-8

import math
from functools import partial
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Time Embeddings
class SinusoidalTimeEmbeddings(nn.Module):
    """Sinusoidal positional embedding (used for time steps in diffusion models)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Position Embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for 1D sequences.

    This class implements position embeddings using sine and cosine functions
    of different frequencies, similar to the Transformer architecture.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, positions):
        device = positions.device
        half_dim = self.dim // 2
        # Create position indices for the sequence
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Calculate position embeddings
        embeddings = positions.unsqueeze(-1) * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # Shape: [batch_size, seq_len, dim]


# Residual Join
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Convolutional Layer for Downsampling
class DownsampleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # For downsampling we need stride=2, but we'll handle padding carefully
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,  # This ensures proper dimension handling for even lengths
        )

    def forward(self, x):
        # Store original length for debugging
        original_length = x.size(-1)

        # For odd sequence lengths, use symmetric padding to better preserve edge information
        if x.size(-1) % 2 != 0:
            x = F.pad(
                x, (1, 0), mode="reflect"
            )  # Reflect padding preserves edge patterns better

        # Apply convolution
        x = self.conv(x)

        # Debug print
        # print(f"Downsample: {original_length} -> {x.size(-1)}")

        return x


# Convolutional Layer for Upsampling
class UpsampleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,  # This helps control the output size
        )

    def forward(self, x, target_size=None):
        # Store original length for debugging
        original_length = x.size(-1)

        # Apply transposed convolution
        x = self.conv(x)

        # If target size is provided, ensure output matches it
        if target_size is not None and x.size(-1) != target_size:
            if x.size(-1) > target_size:
                # If output is too long, center-crop
                start = (x.size(-1) - target_size) // 2
                x = x[:, :, start : start + target_size]
            else:
                # If output is too short, use reflect padding to better preserve edge information
                diff = target_size - x.size(-1)
                left_pad = diff // 2
                right_pad = diff - left_pad
                x = F.pad(x, (left_pad, right_pad), mode="reflect")

        # Debug print
        # print(f"Upsample: {original_length} -> {x.size(-1)}")

        return x


# Convolutional Block with Normalization and Activations
class ConvBlock(nn.Module):
    """1D Convolutional Block with GroupNorm and SiLU activation."""

    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=1,
            padding="same",
        )  # 1D Convolutional Layer
        self.norm = nn.GroupNorm(groups, dim_out)  # Group Normalization
        self.act = nn.SiLU()  # SiLU Activation

    def forward(self, x):
        x = self.conv(x)  # Convolution
        x = self.norm(x)  # Normalization
        x = self.act(x)  # Activation
        return x


# ResnetBlock
class ResnetBlock(nn.Module):
    """
    A residual block with two convolutions and optional time embedding.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # Time embedding
        self.mlp_time = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        # First convolution with normalization and activation - use 'same' padding
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim_out, 3, padding="same"),
        )

        # Second convolution with normalization and activation - use 'same' padding
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding="same"),
        )

        # Residual connection with projection if dimensions change
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        # Save input for residual connection
        identity = x

        # First convolution
        h = self.block1(x)

        # Add time embedding if available
        if self.mlp_time is not None and time_emb is not None:
            time_emb = self.mlp_time(time_emb)
            # Reshape time embedding to match feature map dimensions
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1)
            # Ensure time embedding is broadcast correctly
            if h.size(-1) > 1:
                time_emb = time_emb.expand(-1, -1, h.size(-1))
            h = h + time_emb

        # Second convolution
        h = self.block2(h)

        # No need for interpolation with 'same' padding
        # Residual connection
        return h + self.res_conv(identity)


# Noise Predictor
class UNet1D(nn.Module):
    """
    A 1D U-Net model with residual connections and sinusoidal time embeddings.

    Designed for processing SNP data, which is represented as a 1D sequence.
    """

    def __init__(
        self,
        embedding_dim=64,  # Embedding dimension for time embeddings
        dim_mults=(1, 2, 4, 8),  # Multipliers for feature dimensions at each UNet level
        channels=1,  # Input channels (SNP data has a single channel)
        with_time_emb=True,  # Whether to include time embeddings
        with_pos_emb=True,  # Whether to include position embeddings
        resnet_block_groups=8,  # Number of groups in ResNet blocks for GroupNorm
        seq_length=2067,  # Expected sequence length
    ):
        super().__init__()

        self.channels = channels
        self.seq_length = seq_length  # Make sequence length configurable
        self.use_gradient_checkpointing = False  # Gradient checkpointing flag
        self.with_pos_emb = with_pos_emb

        # Start with a small initial dimension
        init_dim = 16  # Fixed small initial dimension
        out_dim = channels  # Default output channels to match input

        # Initial convolutional layer with same padding
        kernel_size = 3
        padding = (kernel_size - 1) // 2
        self.init_conv = nn.Conv1d(
            channels, init_dim, kernel_size=kernel_size, padding=padding
        )

        # Compute feature dimensions for each UNet level
        dims = [init_dim]  # Start with initial dimension
        for mult in dim_mults:
            dims.append(min(init_dim * mult, 128))  # Cap maximum dimension
        in_out = list(zip(dims[:-1], dims[1:]))

        # Define ResNet block with grouped normalization
        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embeddings
        if with_time_emb:
            time_dim = embedding_dim
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbeddings(
                    embedding_dim
                ),  # Maps scalar time input to embedding
                nn.Linear(embedding_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Positional embeddings (use dedicated position embedding class)
        self.pos_emb = (
            SinusoidalPositionEmbeddings(embedding_dim) if with_pos_emb else None
        )

        # number of block iterators
        num_resolutions = len(in_out)

        # Downsampling Path
        self.downs = nn.ModuleList([])  # Downsampling layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)  # Check if last layer
            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                        resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                        nn.Identity(),  # Placeholder for Attention layer
                        (
                            DownsampleConv(dim_out) if not is_last else nn.Identity()
                        ),  # Downsampling unless last layer
                    ]
                )
            )

        # Bottleneck (Middle Block)
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling Path
        self.ups = nn.ModuleList([])  # Upsampling layers
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)  # Check if last layer
            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                        nn.Identity(),  # Placeholder for Attention layer
                        (
                            UpsampleConv(dim_out) if not is_last else nn.Identity()
                        ),  # Upsampling unless last layer
                    ]
                )
            )

        # Final Convolution with same padding
        self.final_conv = nn.Sequential(
            resnet_block(dims[0], dims[0]),  # ResNet block with same padding
            nn.Conv1d(
                dims[0], out_dim, kernel_size=1, padding="same"
            ),  # 1x1 conv to match channels
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_gradient_checkpointing = True

    def forward(self, x, time):
        """
        Forward pass for UNet1D.

        Args:
            x (torch.Tensor): Input SNP data of shape [batch, 1, seq_len].
            time (torch.Tensor): Diffusion timesteps of shape [batch].

        Returns:
            torch.Tensor: Denoised output.
        """
        # Check input dimensions
        batch, c, seq_len = x.shape
        assert c == self.channels, f"Expected {self.channels} channels, got {c}"

        # Store original sequence length
        original_len = x.size(-1)

        # Ensure input has shape [batch, 1, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Add positional embedding (sinusoidal)
        if self.with_pos_emb and self.pos_emb is not None:
            positions = torch.arange(seq_len, device=x.device).expand(
                batch, -1
            )  # [batch, seq_len]
            pos_encoding = self.pos_emb(positions)  # [batch, seq_len, embedding_dim]
            pos_encoding = pos_encoding.permute(
                0, 2, 1
            )  # [batch, embedding_dim, seq_len]
            # Add position encoding to input
            x = x + pos_encoding[:, : x.shape[1], :]

        # Apply edge padding to better preserve boundary information
        # This extra padding helps the model learn edge patterns better
        edge_pad = 4  # Small padding to help with edge preservation
        x_padded = F.pad(
            x, (edge_pad, edge_pad), mode="reflect"
        )  # Reflect padding preserves edge patterns better

        # Initial convolution
        x = self.init_conv(x_padded)
        t = self.time_mlp(time) if self.time_mlp else None

        # Store intermediate activations and their lengths for skip connections
        h = []

        # Track sequence lengths at each level for debugging
        seq_lengths = [original_len + 2 * edge_pad]

        # Downsampling
        for i, (block1, block2, _, downsample) in enumerate(self.downs):
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block1, x, t, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(block2, x, t, use_reentrant=False)
            else:
                x = block1(x, t)
                x = block2(x, t)
            h.append(x)  # Save features for skip connection

            # Track length before downsampling
            seq_lengths.append(x.size(-1))

            # Apply downsampling
            x = downsample(x)

            # Track length after downsampling
            seq_lengths.append(x.size(-1))

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Upsampling
        for i, (block1, block2, _, upsample) in enumerate(self.ups):
            # Get skip connection
            skip_x = h.pop()

            # Apply upsampling
            x = upsample(x)

            # Ensure dimensions match for concatenation
            if x.size(-1) != skip_x.size(-1):
                # If upsampled feature is longer, truncate
                if x.size(-1) > skip_x.size(-1):
                    # Center-aligned crop to preserve spatial information
                    start = (x.size(-1) - skip_x.size(-1)) // 2
                    x = x[:, :, start : start + skip_x.size(-1)]
                # If upsampled feature is shorter, pad symmetrically with reflection padding
                else:
                    diff = skip_x.size(-1) - x.size(-1)
                    left_pad = diff // 2
                    right_pad = diff - left_pad
                    x = F.pad(x, (left_pad, right_pad), mode="reflect")

            # Concatenate along channel dimension
            x = torch.cat((x, skip_x), dim=1)

            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block1, x, t, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(block2, x, t, use_reentrant=False)
            else:
                x = block1(x, t)
                x = block2(x, t)

        # Final convolution
        x = self.final_conv(x)

        # Remove the extra padding we added at the beginning
        if x.size(-1) > original_len + 2 * edge_pad:
            # If output is longer, center-crop
            start = (x.size(-1) - (original_len + 2 * edge_pad)) // 2
            x = x[:, :, start : start + (original_len + 2 * edge_pad)]

        # Remove the edge padding to get back to original size
        if edge_pad > 0:
            x = x[:, :, edge_pad:-edge_pad]

        # Final check to ensure output has exactly the same length as input
        if x.size(-1) != original_len:
            if x.size(-1) > original_len:
                # If still too long, center-crop
                start = (x.size(-1) - original_len) // 2
                x = x[:, :, start : start + original_len]
            else:
                # If too short, pad symmetrically with reflection
                diff = original_len - x.size(-1)
                left_pad = diff // 2
                right_pad = diff - left_pad
                x = F.pad(x, (left_pad, right_pad), mode="reflect")

        # Print final dimensions for verification
        # print(f"Input length: {original_len}, Output length: {x.size(-1)}")

        return x


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load config
    config = load_config("config.yaml")
    input_path = config.get("input_path")

    print(f"Using device: {device}")

    # Initialize dataset
    print("\nInitializing dataset...")
    dataset = SNPDataset(input_path)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Get batch
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"Batch shape [B, seq_len]: {batch.shape}")

    # Prepare input (ensure [B, C, L] format for UNet)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    print(f"Batch shape [B, C, seq_len]: {batch.shape}")

    # Initialize DDPM
    print("\nInitializing DDPM...")
    forward_diffusion = DDPM(
        num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02
    )

    # Move forward diffusion tensors to device
    forward_diffusion._alphas = forward_diffusion._alphas.to(device)
    forward_diffusion._sigmas = forward_diffusion._sigmas.to(device)

    # Initialize UNet
    print("\nInitializing UNet1D...")
    unet_config = config.get("unet", {})
    unet = UNet1D(
        embedding_dim=unet_config.get("embedding_dim", 32),
        dim_mults=unet_config.get("dim_mults", [1, 2, 4]),
        channels=unet_config.get("channels", 1),
        with_time_emb=unet_config.get("with_time_emb", True),
        resnet_block_groups=unet_config.get("resnet_block_groups", 8),
    ).to(device)

    # Test UNet with different timesteps
    timesteps = [0, 250, 500, 750, 999]

    # For visualization, use a single sample
    x0 = batch[0:1]  # [1, 1, seq_len]
    print(f"Single sample shape: {x0.shape}")

    # Create figure for visualization
    fig, axs = plt.subplots(len(timesteps), 3, figsize=(15, 15))

    for i, t in enumerate(timesteps):
        # Sample noise
        noise = torch.randn_like(x0)
        t_tensor = torch.tensor([t], device=device)

        # Apply forward diffusion to get noisy data
        noisy_data = forward_diffusion.sample(x0, t_tensor, noise)

        # UNet should predict the noise that was added
        noise_pred = unet(noisy_data, t_tensor)

        # Ensure dimensions match
        if noise_pred.shape != noise.shape:
            print(
                f"WARNING: Shape mismatch at t={t}: noise_pred {noise_pred.shape}, noise {noise.shape}"
            )
            if noise_pred.shape[2] != noise.shape[2]:
                noise_pred = F.interpolate(
                    noise_pred, size=noise.shape[2], mode="linear"
                )

        # Calculate MSE loss
        mse_loss = F.mse_loss(noise_pred, noise).item()
        print(f"t={t}, MSE Loss: {mse_loss:.6f}")

        # Plot original data, noisy data, and noise prediction
        axs[i, 0].plot(x0[0, 0].cpu().detach().numpy(), linewidth=1)
        axs[i, 0].set_title(f"t={t}: Original")

        axs[i, 1].plot(noisy_data[0, 0].cpu().detach().numpy(), linewidth=1)
        axs[i, 1].set_title(f"t={t}: Noisy")

        # Plot actual vs predicted noise
        axs[i, 2].plot(noise[0, 0].cpu().detach().numpy(), linewidth=1, label="Actual")
        axs[i, 2].plot(
            noise_pred[0, 0].cpu().detach().numpy(),
            linewidth=1,
            label="Predicted",
            alpha=0.7,
        )
        axs[i, 2].set_title(f"t={t}: Noise (MSE: {mse_loss:.4f})")
        axs[i, 2].legend()

    plt.tight_layout()
    plt.savefig("unet_noise_prediction.png")
    plt.close()

    # Test with full batch
    print("\nTesting with full batch...")
    t_batch = torch.tensor([500] * batch.shape[0], device=device)
    noise_batch = torch.randn_like(batch)

    # Apply forward diffusion
    noisy_batch = forward_diffusion.sample(batch, t_batch, noise_batch)

    # Predict noise with UNet
    pred_noise_batch = unet(noisy_batch, t_batch)

    # Calculate batch MSE loss
    batch_mse = F.mse_loss(pred_noise_batch, noise_batch).item()
    print(f"Batch MSE Loss: {batch_mse:.6f}")

    # Test gradient flow
    print("\nTesting gradient flow...")

    # Zero gradients
    if hasattr(unet, "zero_grad"):
        unet.zero_grad()

    # Forward pass
    pred_noise = unet(noisy_batch, t_batch)

    # Calculate loss
    loss = F.mse_loss(pred_noise, noise_batch)

    # Backward pass
    loss.backward()

    # Check if gradients exist
    has_gradients = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in unet.parameters()
    )
    print(f"UNet has gradients: {has_gradients}")

    print("\nUNet test complete!")


if __name__ == "__main__":
    main()
