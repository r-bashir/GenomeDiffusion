#!/usr/bin/env python
# coding: utf-8

import math
from functools import partial
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


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


# DDPMProcess (Diffusion/Noising Step)
class DDPMProcess:
    """
    A Gaussian diffusion process following the DDPM framework:
    q(xt|x0) = N(alpha(t) * x0, sigma(t)^2 * I)

    Transition from x0 to xt:
        xt = alpha(t) * x0 + sigma(t) * eps, where eps ~ N(0, I).

    This implementation supports SNP data with 1D convolutional processing.
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
            x0 (torch.Tensor): Original clean input (batch_size, seq_len).
            t (torch.Tensor): Diffusion timesteps (batch_size,).
            eps (torch.Tensor): Gaussian noise (batch_size, seq_len).

        Returns:
            torch.Tensor: Noisy sample xt.
        """
        alpha_t = self.alpha(t).view(-1, 1)  # Ensure proper broadcasting
        sigma_t = self.sigma(t).view(-1, 1)
        return alpha_t * x0 + sigma_t * eps


# --------------------------------------- Denoising Process
# Positional Embedding

# - Takes in a batch of scalar time steps and ensures they are 1D.
# - Computes frequency scales using an exponential decay function.
# - Generates sinusoidal embeddings by applying sin and cos transformations.
# - Ensures correct feature dimension by adding padding if necessary.
# - Returns structured time embeddings that can be used in U-Net-based diffusion models.


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
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


# Convolutional Layer for Upsampling
class UpsampleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        return self.conv(x)


# Convolutional Block with Normalization and Activations
class ConvBlock(nn.Module):
    """1D Convolutional Block with GroupNorm and SiLU activation."""

    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, 3, padding=1)  # 1D Convolutional Layer
        self.norm = nn.GroupNorm(groups, dim_out)  # Group Normalization
        self.act = nn.SiLU()  # SiLU Activation

    def forward(self, x):
        x = self.conv(x)  # Convolution
        x = self.norm(x)  # Normalization
        x = self.act(x)  # Activation
        return x


# ResnetBlock
class ResnetBlock(nn.Module):
    """1D CNN with residual connections and optional time embedding."""

    def __init__(self, in_dim, out_dim, time_emb_dim=None, groups=8):
        super().__init__()

        # Time embedding layer (optional)
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_dim))
        else:
            self.mlp = None

        # Convolutional Blocks (use ConvBlock or nn.Sequential)
        # self.block1 = ConvBlock(in_dim, out_dim, groups=groups)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
        )

        # self.block2 = ConvBlock(out_dim, out_dim, groups=groups)
        self.block2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(groups, out_dim),
            nn.SiLU(),
        )

        # Residual connection
        self.res_conv = (
            nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        # First convolution
        h = self.block1(x)

        # Add time embedding if available
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(
                time_emb.shape[0], time_emb.shape[1], 1
            )  # Ensure correct shape
            h = h + time_emb

        # Second convolution
        h = self.block2(h)

        # Residual connection
        return h + self.res_conv(x)


# Denoising Process: UNet
class UNet1D(nn.Module):
    """
    A 1D U-Net model with residual connections and optional sinusoidal time embeddings.

    Designed for processing SNP data, which is represented as a 1D sequence.
    """

    def __init__(
        self,
        embedding_dim=64,  # Embedding dimension for time embeddings
        dim_mults=(1, 2, 4, 8),  # Multipliers for feature dimensions at each UNet level
        channels=1,  # Input channels (SNP data has a single channel)
        with_time_emb=True,  # Whether to include time embeddings
        resnet_block_groups=8,  # Number of groups in ResNet blocks for GroupNorm
    ):
        super().__init__()

        self.channels = channels
        init_dim = (2 * embedding_dim) // 3  # Ensure integer division
        init_dim = init_dim - (init_dim % resnet_block_groups)  # Ensure divisibility
        out_dim = channels  # Default output channels to match input

        # Initial convolutional layer (expands input channels to init_dim)
        self.init_conv = nn.Conv1d(channels, init_dim, kernel_size=7, padding=3)

        # Compute feature dimensions for each UNet level
        dims = [init_dim, *map(lambda m: embedding_dim * m, dim_mults)]
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
                            UpsampleConv(dim_in) if not is_last else nn.Identity()
                        ),  # Upsampling unless last layer
                    ]
                )
            )

        # Final Convolution
        # self.final_conv = nn.Sequential(
        #    resnet_block(embedding_dim, embedding_dim),
        #    nn.Conv1d(embedding_dim, out_dim, 1)
        # )

        # New: final convolution should match the last upsampled feature dimension instead of blindly assuming embedding_dim
        self.final_conv = nn.Sequential(
            resnet_block(
                dims[0], dims[0]
            ),  # dims[0] is the first level of feature map size
            nn.Conv1d(
                dims[0], out_dim, 1
            ),  # dims[0] should match the final upsampled layer output
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
        # Ensure input has shape [batch, 1, seq_len]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Convert [batch, seq_len] → [batch, 1, seq_len]

        # Rest of Forward pass
        x = self.init_conv(x)
        t = self.time_mlp(time) if self.time_mlp else None
        h = []  # Store intermediate activations for skip connections

        # Downsampling
        for block1, block2, _, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)  # Save for skip connection
            x = downsample(x)  # Downsample

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, _, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)  # Concatenate with skip connection
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)  # Upsample

        return self.final_conv(x)  # Final output


# Final Diffusion Model
class DiffusionModel(nn.Module):
    """Diffusion model with 1D Convolutional network for SNP data."""

    def __init__(self, diffusion_process, time_sampler, net_config, data_shape):
        super(DiffusionModel, self).__init__()
        self._process = diffusion_process
        self._time_sampler = time_sampler
        self._net_config = net_config
        self._data_shape = data_shape
        self.net_fwd = UNet1D(net_config)  # Uses Net with ResidualConv1D

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.

        Args:
            x0 (torch.Tensor): Original clean input data (batch_size, seq_len)

        Returns:
            torch.Tensor: MSE loss
        """
        t = self._time_sampler.sample(shape=(x0.shape[0],))  # Sample time
        eps = torch.randn_like(x0, device=x0.device)  # Sample noise
        xt = self._process.sample(x0, t, eps)  # Corrupt the data
        net_outputs = self.net_fwd(xt, t)  # Pass through Conv1D model
        loss = torch.mean((net_outputs - eps) ** 2)  # Compute MSE loss
        return loss

    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes loss at specific timesteps.

        Args:
            x0 (torch.Tensor): Original clean input data.
            eps (torch.Tensor): Sampled noise.
            timesteps (torch.Tensor): Selected timesteps.

        Returns:
            torch.Tensor: Loss values for each timestep.
        """
        losses = []
        timesteps = timesteps.to(x0.device)
        for t in timesteps:
            t = int(t.item()) * torch.ones(
                (x0.shape[0],), dtype=torch.int32, device=x0.device
            )
            xt = self._process.sample(x0, t, eps)
            net_outputs = self.net_fwd(xt, t)
            loss = torch.mean((net_outputs - eps) ** 2)
            losses.append(loss)
        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion step to estimate x_{t-1} given x_t.

        Args:
            xt (torch.Tensor): Noisy input at time t.
            t (int): Current timestep.

        Returns:
            torch.Tensor: Estimated previous timestep data.
        """
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)
        eps_pred = self.net_fwd(xt, t)  # Predict epsilon
        if t > 1:
            sqrt_a_t = self._process.alpha(t) / self._process.alpha(t - 1)
        else:
            sqrt_a_t = self._process.alpha(t)
        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t**2
        inv_sigma_t = 1.0 / self._process.sigma(t)
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt)
        return mean + std * z

    def sample(self, x0, sample_size):
        """
        Samples from the learned reverse diffusion process without conditioning.

        Args:
            x0 (torch.Tensor): Initial input (not used, only for device reference).ples.
            torch.Tensor: Generated samples.
        """
        with torch.no_grad():
            x = torch.randn((sample_size,) + self._data_shape, device=x0.device)
            for t in range(self._process.tmax, 0, -1):
                if t > 1:
                    x = self._reverse_process_step(x, t)ss.tmax, 0, -1):
                else:
                    x = self._reverse_process_step(x, t)x = self._reverse_process_step(x, t)
        return x 
