#!/usr/bin/env python
# coding: utf-8

import math
from functools import partial
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from .network_base import NetworkBase

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
        # Final convolution should match the last upsampled feature
        # dimension instead of blindly assuming embedding_dim
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
class DiffusionModel(NetworkBase):
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
            hparams: Dictionary containing model hyperparameters with the following structure:
                    {
                        'input_path': str,  # Path to input data
                        'diffusion': {'num_diffusion_timesteps': int, 'beta_start': float, 'beta_end': float},
                        'time_sampler': {'tmin': int, 'tmax': int},
                        'unet': {'embedding_dim': int, 'dim_mults': List[int], ...},
                        'data': {'seq_length': int, 'batch_size': int, 'num_workers': int}
                    }
        """
        super().__init__(hparams=hparams)

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
        )

    def predict_added_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added during forward diffusion.
        This is a key part of the DDPM's reverse process - the UNet learns to
        predict what noise was added, which helps us denoise the data.

        Args:
            x (torch.Tensor): Noisy input tensor at timestep t
            t (torch.Tensor): Current timestep tensor

        Returns:
            torch.Tensor: Predicted noise that was added during forward diffusion
        """
        return self.unet(x, t)

    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: Predicted noise
        """
        # Sample time and noise
        t = self._time_sampler.sample(shape=(batch.shape[0],))
        eps = torch.randn_like(batch)

        # Forward diffusion process
        xt = self._forward_diffusion.sample(batch, t, eps)

        # Predict noise added during forward diffusion
        return self.predict_added_noise(xt, t)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.

        Args:
            batch: Input batch from dataloader of shape (batch_size, channels, seq_len)

        Returns:
            torch.Tensor: MSE loss
        """
        # Sample noise
        eps = torch.randn_like(batch)

        # Get model predictions
        pred_eps = self.forward_step(batch)

        # Compute MSE loss
        return torch.mean((pred_eps - eps) ** 2)

    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes loss at specific timesteps.

        Args:
            x0 (torch.Tensor): Original clean input data (batch_size, channels, seq_len).
            eps (torch.Tensor): Sampled noise of same shape as x0.
            timesteps (torch.Tensor): Timesteps to evaluate.

        Returns:
            torch.Tensor: Loss values for each timestep.
        """
        losses = []
        for t in timesteps:
            t = int(t.item()) * torch.ones((x0.shape[0],), dtype=torch.int32)
            # Forward diffusion at timestep t
            xt = self._forward_diffusion.sample(x0, t, eps)

            # Predict noise that was added
            predicted_noise = self.predict_added_noise(xt, t)

            # Compute loss at this timestep
            loss = torch.mean((predicted_noise - eps) ** 2)
            losses.append(loss)

        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion step to estimate x_{t-1} given x_t.

        Args:
            xt (torch.Tensor): Noisy input at time t of shape (batch_size, channels, seq_len).
            t (int): Current timestep.

        Returns:
            torch.Tensor: Estimated previous timestep data.
        """
        # Move input to device and create timestep tensor
        xt = xt.to(device)
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=device)

        # Predict noise that was added during forward diffusion
        eps_pred = self.predict_added_noise(xt, t)

        # Compute reverse process parameters
        if t > 1:
            sqrt_a_t = self._forward_diffusion.alpha(t) / self._forward_diffusion.alpha(
                t - 1
            )
        else:
            sqrt_a_t = self._forward_diffusion.alpha(t)

        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t**2
        inv_sigma_t = 1.0 / self._forward_diffusion.sigma(t)

        # Compute mean and standard deviation
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        std = torch.sqrt(beta_t)

        # Add noise scaled by standard deviation
        z = torch.randn_like(xt, device=device)
        return mean + std * z

    def sample(self, sample_size: int) -> torch.Tensor:
        """
        Samples from the learned reverse diffusion process without conditioning.
        Implements the full reverse diffusion chain, starting from pure noise and
        gradually denoising to generate SNP data.

        Args:
            sample_size (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (sample_size, channels, seq_len).
        """
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn((sample_size,) + self._data_shape)

            # Gradually denoise
            for t in range(self._forward_diffusion.tmax, 0, -1):
                x = self._reverse_process_step(x, t)

            # Ensure output is in correct range (typically [0,1] for SNP data)
            x = torch.clamp(x, 0, 1)
        return x
