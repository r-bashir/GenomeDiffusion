#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import time
import numpy as np
import pandas as pd

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

# Pytorch Lightening
import pytorch_lightning as pl


# In[ ]:


def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Util function for broadcasting to the right."""
    if x.ndim > ndim:
        raise ValueError(f'Cannot broadcast a value with {x.ndim} dims to {ndim} dims.')
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(x.shape + (1,) * difference)
    else:
        return x


# ## _Time Embedding_
# - Takes in a batch of scalar time steps and ensures they are 1D.
# - Computes frequency scales using an exponential decay function.
# - Generates sinusoidal embeddings by applying sin and cos transformations.
# - Ensures correct feature dimension by adding padding if necessary.
# - Returns structured time embeddings that can be used in U-Net-based diffusion models.



class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding (used for time steps in diffusion models)."""
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape [batch_size] containing scalar time steps.

        Returns:
            Tensor of shape [batch_size, num_features], sinusoidal embeddings.
        """
        if inputs.dim() == 2 and inputs.shape[1] == 1:
            inputs = inputs.view(-1)  # Flatten if shape is [batch_size, 1]
        
        device = inputs.device
        half_dim = self.num_features // 2
        e = math.log(10000.0) / (half_dim - 1)
        inv_freq = torch.exp(-e * torch.arange(half_dim, device=device).float())

        emb = inputs[:, None] * inv_freq[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        if self.num_features % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))  # Pad last dimension if odd

        return emb


# ## _Sampling_

class UniformDiscreteTimeSampler:

    def __init__(self, tmin: int, tmax: int):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randint(low=self._tmin, high=self._tmax, size=shape)


# ## _Residual Block_


class ResidualConv1D(nn.Module):
    """1D CNN with residual connections for SNP data."""

    def __init__(
        self,
        in_channels: int,   # Number of input channels (e.g., 1 if SNPs are single-channel)
        out_channels: int,  # Number of output channels (same as in_channels for residual)
        kernel_size: int = 3,  # Size of the 1D convolution kernel
        activation: str = 'relu'
    ):
        super(ResidualConv1D, self).__init__()
        
        self.activation = getattr(F, activation)
        padding = kernel_size // 2  # Ensure same spatial size output
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Optional embedding for categorical labels (if needed)
        self.label_emb = nn.Embedding(3, out_channels)

    def forward(self, xt: torch.Tensor, time: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        xt: (batch, in_channels, sequence_length)
        time: (batch, embedding_dim)
        label: (batch,)
        """
        # Label embedding
        c = self.label_emb(label).unsqueeze(2)  # (batch, channels) -> (batch, channels, 1)
        
        # First Conv + BN + Activation
        h = self.conv1(xt)  
        h = self.bn1(h)
        h = self.activation(h)
        
        # Second Conv + BN
        h = self.conv2(h)
        h = self.bn2(h)
        
        # Residual connection
        x = xt + h
        
        return x


# ## _U-Net_

@dataclasses.dataclass
class NetConfig:
    resnet_n_blocks: int = 2
    resnet_n_hidden: int = 256
    resnet_n_out: int = 6
    activation: str = 'elu'
    time_embedding_dim: int = 256


# In[ ]:


class Net(nn.Module):
    """Combines 1D CNN and time embeddings for SNP data."""
    
    def __init__(self, net_config: NetConfig, name: str = None):
        super(Net, self).__init__()

        self._time_encoder = SinusoidalTimeEmbedding(net_config.time_embedding_dim)
        
        self._predictor = ResidualConv1D(
            in_channels=1,  # Assuming SNP data has 1 input channel
            out_channels=net_config.resnet_n_hidden,  # Number of hidden channels
            kernel_size=3,  # Typical for SNP data
            activation=net_config.activation
        )

    def forward(self, noisy_data: torch.Tensor, time: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SNP data.

        Args:
            noisy_data (torch.Tensor): (batch_size, sequence_length)
            time (torch.Tensor): (batch_size,)
            label (torch.Tensor): (batch_size,)

        Returns:
            torch.Tensor: (batch_size, out_channels, sequence_length)
        """
        # Reshape input for Conv1D: (batch_size, 1, sequence_length)
        noisy_data = noisy_data.unsqueeze(1)

        # Encode time
        time_embedding = self._time_encoder(time)

        # Pass through 1D CNN predictor
        outputs = self._predictor(noisy_data, time_embedding, label)

        return outputs


# ## _DDPM Process_

# In[ ]:


class DiscreteDDPMProcess:
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
        self._betas = np.linspace(self._beta_start, self._beta_end, self._num_diffusion_timesteps)

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
        alphas_bar = np.concatenate(([1.], alphas_bar))

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

    def sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
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


# ## _Diffusion Model_

class DiffusionModel(nn.Module):
    """Diffusion model with 1D Convolutional network for SNP data."""

    def __init__(self, diffusion_process, time_sampler, net_config, data_shape):
        super(DiffusionModel, self).__init__()

        self._process = diffusion_process
        self._time_sampler = time_sampler
        self._net_config = net_config
        self._data_shape = data_shape
        self.net_fwd = Net(net_config)  # Uses Net with ResidualConv1D

    def loss(self, x0: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Computes MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.

        Args:
            x0 (torch.Tensor): Original clean input data (batch_size, seq_len)
            label (torch.Tensor): Label tensor (batch_size,)

        Returns:
            torch.Tensor: MSE loss
        """
        t = self._time_sampler.sample(shape=(x0.shape[0],))  # Sample time

        eps = torch.randn_like(x0, device=x0.device)  # Sample noise

        xt = self._process.sample(x0, t, eps)  # Corrupt the data

        net_outputs = self.net_fwd(xt, t, label)  # Pass through Conv1D model

        loss = torch.mean((net_outputs - eps) ** 2)  # Compute MSE loss

        return loss

    def loss_per_timesteps(self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Computes loss at specific timesteps.

        Args:
            x0 (torch.Tensor): Original clean input data.
            eps (torch.Tensor): Sampled noise.
            timesteps (torch.Tensor): Selected timesteps.
            label (torch.Tensor): Label tensor.

        Returns:
            torch.Tensor: Loss values for each timestep.
        """
        losses = []
        for t in timesteps:
            t = int(t.item()) * torch.ones((x0.shape[0],), dtype=torch.int32, device=x0.device)
            xt = self._process.sample(x0, t, eps)
            net_outputs = self.net_fwd(xt, t, label)
            loss = torch.mean((net_outputs - eps) ** 2)
            losses.append(loss)
        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int, label: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion step to estimate x_{t-1} given x_t.

        Args:
            xt (torch.Tensor): Noisy input at time t.
            t (int): Current timestep.
            label (torch.Tensor): Labels for conditioning.

        Returns:
            torch.Tensor: Estimated previous timestep data.
        """
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)

        eps_pred = self.net_fwd(xt, t, label)  # Predict epsilon

        sqrt_a_t = self._process.alpha(t) / self._process.alpha(t - 1)
        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t ** 2
        inv_sigma_t = 1.0 / self._process.sigma(t)

        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)

        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt)

        return mean + std * z

    def sample(self, x0, sample_size, label):
        """
        Samples from the learned reverse diffusion process.

        Args:
            x0 (torch.Tensor): Initial input (not used, only for device reference).
            sample_size (int): Number of samples.
            label (torch.Tensor): Labels for conditioning.

        Returns:
            torch.Tensor: Generated samples.
        """
        with torch.no_grad():
            x = torch.randn((sample_size,) + self._data_shape, device=x0.device)

            for t in range(self._process.tmax, 0, -1):
                x = self._reverse_process_step(x, t, label)

        return x