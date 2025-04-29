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


def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Util function for broadcasting to the right."""
    if x.ndim > ndim:
        raise ValueError(f"Cannot broadcast a value with {x.ndim} dims to {ndim} dims.")
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(x.shape + (1,) * difference)
    else:
        return x


#  Discrete Time Sampling
class UniformDiscreteTimeSampler:

    def __init__(self, tmin: int, tmax: int):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randint(low=self._tmin, high=self._tmax, size=shape)


#  Continuous Time Sampling
class UniformContinuousTimeSampler:
    def __init__(self, tmin: float, tmax: float):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return self._tmin + (self._tmax - self._tmin) * torch.rand(size=shape)


class DDPM:
    """
    Implements the forward diffusion process that gradually adds noise to data.
    Following DDPM framework, at each timestep t, we add a controlled amount of
    Gaussian noise according to:
        q(xt|x0) = N(alpha(t) * x0, sigma(t)^2 * I)

    The forward process transitions from clean data x0 to noisy data xt via:
        xt = alpha(t) * x0 + sigma(t) * eps, where eps ~ N(0, I)

    As t increases, more noise is added until the data becomes pure noise.
    This creates the training pairs (xt, t, eps) that teach the UNet to
    predict the added noise at each timestep.
    """

    def __init__(
        self,
        diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Initializes the diffusion process.

        Args:
            diffusion_steps (int): Number of diffusion steps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
        """
        self._diffusion_steps = diffusion_steps
        self._beta_start = beta_start
        self._beta_end = beta_end

        # Use liner beta scheduler
        self._betas = np.linspace(
            self._beta_start, self._beta_end, self._diffusion_steps
        )

        # Use cosine beta scheduler
        self._betas = self._cosine_beta_schedule(self._diffusion_steps)
        alphas_bar = self._get_alphas_bar()

        # Register tensors as buffers so they move with the model
        self.register_buffer(
            "_alphas", torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)
        )
        self.register_buffer(
            "_sigmas", torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.

        Args:
            timesteps: Number of diffusion timesteps
            s: Offset parameter (default: 0.008)

        Returns:
            np.ndarray: Beta schedule array of shape (timesteps,)
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    @property
    def tmin(self) -> int:
        """Minimum timestep value."""
        return 1

    @property
    def tmax(self) -> int:
        """Maximum timestep value."""
        return self._diffusion_steps

    def _get_alphas_bar(self) -> np.ndarray:
        """Computes cumulative alpha values following the DDPM formula."""
        alphas_bar = np.cumprod(1.0 - self._betas)
        # Append 1 at the beginning for convenient indexing
        alphas_bar = np.concatenate(([1.0], alphas_bar))
        return alphas_bar

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves alpha(t) for the given time indices.

        Args:
            t (torch.Tensor): Timesteps (batch_size,).

        Returns:
            torch.Tensor: Alpha values corresponding to timesteps.
        """
        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)
        # Convert to indices (0-indexed)
        idx = (t - 1).long()

        # Ensure idx is on the same device as _alphas
        if idx.device != self._alphas.device:
            idx = idx.to(self._alphas.device)

        # Return values on the correct device
        return self._alphas[idx]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves sigma(t) for the given time indices.

        Args:
            t (torch.Tensor): Timesteps (batch_size,).

        Returns:
            torch.Tensor: Sigma values corresponding to timesteps.
        """
        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)
        # Convert to indices (0-indexed)
        idx = (t - 1).long()

        # Ensure idx is on the same device as _sigmas
        if idx.device != self._sigmas.device:
            idx = idx.to(self._sigmas.device)

        # Return values on the correct device
        return self._sigmas[idx]

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the forward diffusion process q(xt | x0). It gives you the
        noisy version xt from x0 and t, without looping through every step.

        Args:
            x0 (torch.Tensor): Original clean input (batch_size, [channels,] seq_len).
            t (torch.Tensor): Diffusion timesteps (batch_size,).
            eps (torch.Tensor): Gaussian noise with same shape as x0.

        Returns:
            torch.Tensor: Noisy sample xt.
        """
        # Get alpha and sigma values for the timesteps
        alpha_values = self.alpha(t)
        sigma_values = self.sigma(t)

        # Move alpha and sigma to the same device as x0
        alpha_values = alpha_values.to(x0.device)
        sigma_values = sigma_values.to(x0.device)

        # Reshape alpha_t and sigma_t according to the input shape
        if len(x0.shape) == 3:  # [batch_size, channels, seq_len]
            alpha_t = alpha_values.view(-1, 1, 1)
            sigma_t = sigma_values.view(-1, 1, 1)
        else:  # [batch_size, seq_len]
            alpha_t = alpha_values.view(-1, 1)
            sigma_t = sigma_values.view(-1, 1)

        xt = alpha_t * x0 + sigma_t * eps
        return xt

    def register_buffer(self, name, tensor):
        """
        Registers a tensor as a buffer (similar to PyTorch's nn.Module.register_buffer).
        This is a simple implementation for a non-nn.Module class.
        """
        setattr(self, name, tensor)

    def to(self, device):
        """
        Moves all tensors to the specified device.
        This mimics the behavior of nn.Module.to() for compatibility with PyTorch Lightning.
        """
        self._alphas = self._alphas.to(device)
        self._sigmas = self._sigmas.to(device)
        return self


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
