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


# Time Sampling
class UniformDiscreteTimeSampler:

    def __init__(self, tmin: int, tmax: int):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randint(low=self._tmin, high=self._tmax, size=shape)


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
        num_diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Initializes the diffusion process.

        Args:
            num_diffusion_timesteps (int): Number of diffusion steps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
        """
        self._num_diffusion_timesteps = num_diffusion_timesteps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._betas = np.linspace(
            self._beta_start, self._beta_end, self._num_diffusion_timesteps
        )
        alphas_bar = self._get_alphas_bar()
        self._alphas = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)
        self._sigmas = torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32)

    @property
    def tmin(self) -> int:
        """Minimum timestep value."""
        return 1

    @property
    def tmax(self) -> int:
        """Maximum timestep value."""
        return self._num_diffusion_timesteps

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
        return self._alphas[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves sigma(t) for the given time indices.

        Args:
            t (torch.Tensor): Timesteps (batch_size,).

        Returns:
            torch.Tensor: Sigma values corresponding to timesteps.
        """
        return self._sigmas[t.long()]

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the forward diffusion process q(xt | x0).

        Args:
            x0 (torch.Tensor): Original clean input (batch_size, [channels,] seq_len).
            t (torch.Tensor): Diffusion timesteps (batch_size,).
            eps (torch.Tensor): Gaussian noise with same shape as x0.

        Returns:
            torch.Tensor: Noisy sample xt.
        """
        # Reshape alpha_t and sigma_t according to the input shape
        if len(x0.shape) == 3:  # [batch_size, channels, seq_len]
            alpha_t = self.alpha(t).view(-1, 1, 1)  # Reshape for 3D tensor
            sigma_t = self.sigma(t).view(-1, 1, 1)
        else:  # [batch_size, seq_len]
            alpha_t = self.alpha(t).view(-1, 1)  # Reshape for 2D tensor
            sigma_t = self.sigma(t).view(-1, 1)

        xt = alpha_t * x0 + sigma_t * eps
        return xt


# Denoising Process
class SinusoidalPositionalEmbeddings(nn.Module):
    """Sinusoidal positional embedding (used for time steps in diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape [batch_size] containing scalar time steps.

        Returns:
            Tensor of shape [batch_size, dim], sinusoidal embeddings.
        """
        if time.dim() == 2 and time.shape[1] == 1:
            time = time.view(-1)  # Flatten if shape is [batch_size, 1]

        device = time.device
        half_dim = self.dim // 2
        e = math.log(10000.0) / (half_dim - 1)
        inv_freq = torch.exp(-e * torch.arange(half_dim, device=device).float())

        embeddings = time[:, None] * inv_freq[None, :]
        embeddings = torch.cat([torch.cos(embeddings), torch.sin(embeddings)], dim=-1)

        if self.dim % 2 == 1:
            embeddings = nn.functional.pad(
                embeddings, (0, 1)
            )  # Pad last dimension if odd

        return embeddings


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
        self.mlp = (
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
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
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


# Denoising Process: UNet
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
        resnet_block_groups=8,  # Number of groups in ResNet blocks for GroupNorm
        seq_length=2067,  # Expected sequence length
    ):
        super().__init__()

        self.channels = channels
        self.seq_length = seq_length  # Make sequence length configurable

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
                SinusoidalPositionalEmbeddings(
                    embedding_dim
                ),  # Maps scalar time input to embedding
                nn.Linear(embedding_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

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

        # Apply edge padding to better preserve boundary information
        # This extra padding helps the model learn edge patterns better
        edge_pad = 4  # Small padding to help with edge preservation
        x_padded = F.pad(x, (edge_pad, edge_pad), mode="reflect")

        # Initial convolution
        x = self.init_conv(x_padded)
        t = self.time_mlp(time) if self.time_mlp else None

        # Store intermediate activations and their lengths for skip connections
        h = []

        # Track sequence lengths at each level for debugging
        seq_lengths = [original_len + 2 * edge_pad]

        # Downsampling
        for i, (block1, block2, _, downsample) in enumerate(self.downs):
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

            # Apply blocks
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


# Diffusion Model: We wont use this model, since it is an nn.Module for PyTorch.
# To use it with PyTorch Lighting, we have transfered it to 'diffusion_model.py'
# as a LightingModule with additional hooks for training. Kept it for PyTorch.
class DiffusionModel(nn.Module):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse diffusion (denoising)
    processes for SNP data. The forward process gradually adds noise to the data following
    a predefined schedule, while the reverse process learns to denoise the data using a
    UNet1D architecture.
    """

    def __init__(self, hparams: Dict):
        """
        Initialize the diffusion model with hyperparameters.

        Args:
            hparams: Dictionary containing model hyperparameters.
        """
        super().__init__()

        # Set data shape
        self._data_shape = (hparams["unet"]["channels"], hparams["data"]["seq_length"])

        # Initialize components from hyperparameters
        self._forward_diffusion = DDPM(
            num_diffusion_timesteps=hparams["diffusion"]["num_diffusion_timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
        )

        self._time_sampler = UniformDiscreteTimeSampler(
            tmin=hparams["time_sampler"]["tmin"], tmax=hparams["time_sampler"]["tmax"]
        )

        self.unet = UNet1D(
            embedding_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            resnet_block_groups=hparams["unet"]["resnet_block_groups"],
            seq_length=hparams["data"]["seq_length"],
        )

    def predict_added_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added during forward diffusion."""
        # Ensure x has the correct shape for UNet input
        if len(x.shape) == 2:  # If shape is (batch_size, seq_len)
            x = x.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)

        # Print input shape for debugging
        print(f"Noise prediction input shape: {x.shape}")

        # Run through UNet
        pred_noise = self.unet(x, t)

        # Print output shape for debugging
        print(f"Noise prediction output shape: {pred_noise.shape}")

        return pred_noise

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            batch (torch.Tensor): Input batch of shape (batch_size, channels, seq_len).

        Returns:
            torch.Tensor: Predicted noise
        """
        return self.forward_step(batch)

    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model."""
        # Sample time and noise
        t = self._time_sampler.sample(shape=(batch.shape[0],))
        eps = torch.randn_like(batch)

        # Forward diffusion process
        xt = self._forward_diffusion.sample(batch, t, eps)

        # Debugging print statements
        print(
            f"Mean abs diff between x0 and xt: {(batch - xt).abs().mean().item()}"
        )  # Check noise level

        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(xt.shape) == 2:  # If shape is (batch_size, seq_len)
            xt = xt.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        elif xt.shape[1] != 1:  # If incorrect number of channels
            print(f"Unexpected number of channels: {xt.shape[1]}, reshaping...")
            xt = xt[:, :1, :]  # Force to 1 channel
        print(f"Final shape before UNet: {xt.shape}")

        # Predict noise added during forward diffusion
        return self.predict_added_noise(xt, t)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.
        xt = alpha(t) * x0 + sigma(t)**2 * eps

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: MSE loss
        """
        # Sample true noise
        eps = torch.randn_like(batch)

        # Get model predictions
        pred_eps = self.forward_step(batch)

        # Compute MSE loss
        return torch.mean((pred_eps - eps) ** 2)

    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Computes loss at specific timesteps."""
        losses = []
        for t in timesteps:
            t = int(t.item()) * torch.ones((x0.shape[0],), dtype=torch.int32)
            xt = self._forward_diffusion.sample(x0, t, eps)

            predicted_noise = self.predict_added_noise(xt, t)
            loss = torch.mean((predicted_noise - eps) ** 2)
            losses.append(loss)

        print(f"Loss across timesteps: {torch.stack(losses).detach().cpu().numpy()}")
        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion step to estimate x_{t-1} given x_t.
        Computes parameters of a Gaussian p(x_{t-1}| x_t, x0_pred),
        DDPM sampling method - algorithm 1: It formalizes the whole generative procedure.
        """

        xt = xt.to(device)
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=device)

        eps_pred = self.predict_added_noise(xt, t)

        if t > 1:
            sqrt_a_t = self._forward_diffusion.alpha(t) / self._forward_diffusion.alpha(
                t - 1
            )
        else:
            sqrt_a_t = self._forward_diffusion.alpha(t)

        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t**2
        inv_sigma_t = 1.0 / self._forward_diffusion.sigma(t)

        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)

        # DDPM instructs to use either the variance of the forward process
        # or the variance of posterior q(x_{t-1}|x_t, x_0). Former is easier.

        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt, device=device)

        # The reparameterization trick: N(mean, variance^2) = mean + std(sigma) * epsilon
        return mean + std * z

    def sample(self, sample_size: int) -> torch.Tensor:
        """Samples from the learned reverse diffusion process without conditioning."""
        with torch.no_grad():
            x = torch.randn((sample_size,) + self._data_shape)

            for t in range(self._forward_diffusion.tmax, 0, -1):
                x = self._reverse_process_step(x, t)

                if t % 100 == 0:
                    print(
                        f"Sampling at timestep {t}, mean: {x.mean().item()}, std: {x.std().item()}"
                    )

            x = torch.clamp(x, 0, 1)

        print(f"Final sample mean: {x.mean().item()}, std: {x.std().item()}")
        return x
