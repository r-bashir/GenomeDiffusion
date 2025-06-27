#!/usr/bin/env python
# coding: utf-8

"""Forward Diffusion Process for Denoising Diffusion Probabilistic Models (DDPM).

This module implements the FORWARD (noising) process of the DDPM framework as described in:
'Denoising Diffusion Probabilistic Models' (Ho et al., 2020)
https://arxiv.org/abs/2006.11239

The implementation includes both cosine and linear noise schedules, with the cosine
schedule following the improvements from 'Improved Denoising Diffusion Probabilistic Models'
(Nichol & Dhariwal, 2021) https://arxiv.org/abs/2102.09672

Typical usage:
    forward_diff = ForwardDiffusion(time_steps=1000, schedule_type='cosine')
    noisy_sample = forward_diff.sample(x0, timestep, noise)

Note: This class only implements the forward (noising) process q(x_t|x_0), NOT the full DDPM model.
"""

from typing import Optional, Union

import numpy as np
import torch

from .utils import bcast_right, prepare_batch_shape, tensor_to_device


class ForwardDiffusion:
    """Implements the forward diffusion process that gradually adds noise to data.

    This class implements the DDPM framework where the forward process gradually
    adds Gaussian noise to data according to a fixed schedule, and the reverse
    process learns to denoise the data. The forward process is defined by:
        q(x_t | x_0) = N(x_t; √(ᾱ_t) * x_0, (1-ᾱ_t)I)
    where ᾱ_t = Π_{s=1}^t (1-β_s) and β_s is the noise schedule.

    The forward process can be sampled using the reparameterization trick:
        x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0, I)

    In this implementation, reverse diffusion is handled by DiffusionModel.

    Attributes:
        _time_steps (int): Total number of diffusion steps T
        _beta_start (float): Starting value for β schedule (β_1)
        _beta_end (float): Ending value for β schedule (β_T)
        _schedule_type (str): Type of β schedule ('cosine' or 'linear')
        _betas_t (torch.Tensor): β values for each timestep, shape (T,)
        _alphas_t (torch.Tensor): α_t = 1-β_t values, shape (T,)
        _alphas_bar_t (torch.Tensor): Cumulative product ᾱ_t = Π_{s=1}^t α_s, shape (T+1,)
        _sigmas_t (torch.Tensor): Noise scale σ_t = √(1-ᾱ_t), shape (T+1,)
    """

    def __init__(
        self,
        time_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        """
        Initialize the diffusion process with specified noise schedule parameters.

        Args:
            time_steps (int, optional): Total number of diffusion timesteps T. Defaults to 1000.
            beta_start (float, optional): Starting value for β schedule. Defaults to 0.0001.
            beta_end (float, optional): Final value for β schedule. Defaults to 0.02.
            schedule_type (str, optional): Type of β schedule to use ('cosine' or 'linear'). Defaults to 'cosine'.
            max_beta (float, optional): Maximum value for beta in any schedule. Defaults to 0.999.

        Raises:
            ValueError: If schedule_type is not 'cosine' or 'linear'.
        """

        # Initialize parameters
        self._time_steps = time_steps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._schedule_type = schedule_type

        # Select beta schedule
        if schedule_type == "linear":
            betas_np = self._linear_beta_schedule(
                self._time_steps, self._beta_start, self._beta_end
            )
        elif schedule_type == "cosine":
            betas_np = self._cosine_beta_schedule(self._time_steps)
        else:
            raise ValueError(
                f"Unknown schedule_type '{schedule_type}'. Use 'cosine' or 'linear'."
            )

        # alpha = α_t = 1 - β_t
        alphas_np = 1.0 - betas_np

        # cumulative product of alphas: ᾱ_t = Π_{s=1}^{t} α_s,
        # prepend 1 for t=0 (no noise) for convenient indexing
        alphas_bar_np = np.concatenate(([1.0], np.cumprod(alphas_np)))

        # sigmas: σ_t = √(1-ᾱ_t)
        sigmas_np = np.sqrt(1.0 - alphas_bar_np)

        # Convert to tensors and register as buffers
        self.register_buffer("_betas_t", torch.tensor(betas_np, dtype=torch.float32))
        self.register_buffer("_alphas_t", torch.tensor(alphas_np, dtype=torch.float32))
        self.register_buffer(
            "_alphas_bar_t", torch.tensor(alphas_bar_np, dtype=torch.float32)
        )
        self.register_buffer("_sigmas_t", torch.tensor(sigmas_np, dtype=torch.float32))

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the forward diffusion process q(x_t | x_0).

        Implements:
            q(x_t|x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1-ᾱ_t)I)
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε, ε ~ N(0, I)

        Args:
            x0: Clean data sample of shape [B, C, seq_len].
            t: Timestep indices, shape (batch_size,). Should be in [1, time_steps].
            eps: Pre-generated noise of same shape as x0.

        Returns:
            torch.Tensor: Noisy sample x_t.
        """
        device = x0.device
        # Ensure x0 has the correct shape [B, C, L]
        x0 = prepare_batch_shape(x0)

        # Move t to the same device as x0 before indexing
        t = tensor_to_device(t, device).long()

        # Get parameters and ensure they're on the same device as x0
        alpha_bar_t = self.alpha_bar(t)
        sigma_t = self.sigma(t)

        # Broadcast parameters to match x0's dimensions
        ndim = x0.ndim
        alpha_bar_t = bcast_right(alpha_bar_t, ndim)
        sigma_t = bcast_right(sigma_t, ndim)

        # Ensure noise is on the same device
        eps = tensor_to_device(eps, device)

        # Compute noisy sample
        x_t = torch.sqrt(alpha_bar_t) * x0 + sigma_t * eps
        return x_t

    # ====================== nn.Module like Methods ======================
    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """
        Register a tensor as a buffer (similar to PyTorch's nn.Module.register_buffer).

        This is a simplified implementation for a non-nn.Module class that allows the
        DDPM to maintain persistent state that should be saved and restored in the
        state_dict, but not trained.

        Args:
            name (str): Name of the buffer to register.
            tensor (torch.Tensor): Tensor to register as a buffer.
        """
        setattr(self, name, tensor)

    def to(self, device: Union[str, torch.device]) -> "ForwardDiffusion":
        """Move all internal tensors to the specified device.

        This method mimics the behavior of nn.Module.to() for compatibility with
        PyTorch Lightning and general PyTorch operations. It ensures all internal
        tensors (betas, alphas, etc.) are on the same device.

        Args:
            device (Union[str, torch.device]): Target device to move tensors to.
                Can be either a string (e.g., 'cuda:0', 'cpu') or torch.device.

        Returns:
            ForwardDiffusion: Returns self for method chaining.
        """
        # Move all internal tensors to the target device
        self._betas_t = tensor_to_device(self._betas_t, device)
        self._alphas_t = tensor_to_device(self._alphas_t, device)
        self._alphas_bar_t = tensor_to_device(self._alphas_bar_t, device)
        self._sigmas_t = tensor_to_device(self._sigmas_t, device)
        return self

    # ====================== Methods ======================
    def _check_timestep(self, t: torch.Tensor) -> None:
        """Ensure t is within valid bounds [1, time_steps]."""
        if torch.any(t < 1) or torch.any(t > self._time_steps):
            raise ValueError(
                f"Timestep t={t} out of valid range [1, {self._time_steps}]"
            )

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Retrieves β_t for the given timesteps.

        The β_t values determine how much noise is added at each timestep.

        Args:
            t (torch.Tensor): Timestep indices, shape (batch_size,). Should be in [1, time_steps].

        Returns:
            torch.Tensor: β_t values of shape (batch_size,), where:
                - For timestep t, returns β_t
                - t=1 corresponds to the first diffusion step (β_1)
                - The returned values are on the same device as the internal tensors
        """
        # Check if t is within valid range
        self._check_timestep(t)

        # Move betas to the same device as t before indexing
        device = t.device
        betas = tensor_to_device(self._betas_t, device)

        # Convert to 0-indexed for tensor access (t=1 -> index 0)
        t_idx = t.long() - 1
        return betas[t_idx]

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Retrieves α_t = 1-β_t for the given timesteps.

        The α_t values determine how much signal is preserved at each timestep.

        Args:
            t (torch.Tensor): Timestep indices, shape (batch_size,). Should be in [1, time_steps].

        Returns:
            torch.Tensor: α_t values of shape (batch_size,), where:
                - For timestep t, returns α_t = 1-β_t
                - t=1 corresponds to the first diffusion step (β_1)
                - The returned values are on the same device as the internal tensors
        """
        # Check if t is within valid range
        self._check_timestep(t)

        # Move alphas to the same device as t before indexing
        device = t.device
        alphas = tensor_to_device(self._alphas_t, device)

        # Convert to 0-indexed for tensor access (t=1 -> index 0)
        t_idx = t.long() - 1
        return alphas[t_idx]

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Retrieves ᾱ_t = Π_{s=1}^t α_s for the given timesteps.

        The ᾱ_t values represent the total signal preservation up to timestep t.

        Args:
            t (torch.Tensor): Timestep indices, shape (batch_size,). Should be in [0, time_steps].

        Returns:
            torch.Tensor: ᾱ_t values of shape (batch_size,), where:
                - For timestep t, returns ᾱ_t = Π_{s=1}^t α_s
                - t=0 corresponds to the original data (ᾱ_0 = 1.0)
                - t=1 corresponds to the first diffusion step (ᾱ_1 = α_1)
                - The returned values are on the same device as the internal tensors
        """
        # Allow t in [0, time_steps] for theoretical calculations
        if not torch.all((t >= 0) & (t <= self._time_steps)):
            raise ValueError(
                f"Timestep t={t} out of valid range [0, {self._time_steps}]"
            )
        device = t.device
        alphas_bar = tensor_to_device(self._alphas_bar_t, device)
        return alphas_bar[t.long()]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Retrieves σ_t = √(1-ᾱ_t) for the given timesteps.

        The σ_t values represent the standard deviation of the noise added at each timestep.

        Args:
            t (torch.Tensor): Timestep indices, shape (batch_size,). Should be in [0, time_steps].

        Returns:
            torch.Tensor: σ_t values of shape (batch_size,), where:
                - For timestep t, returns σ_t = √(1-ᾱ_t)
                - t=0 corresponds to the original data (σ_0 = 0.0)
                - t=1 corresponds to the first diffusion step (σ_1 = √(1-ᾱ_1))
                - The returned values are on the same device as the internal tensors
        """
        # Allow t in [0, time_steps] for theoretical calculations
        if not torch.all((t >= 0) & (t <= self._time_steps)):
            raise ValueError(
                f"Timestep t={t} out of valid range [0, {self._time_steps}]"
            )
        device = t.device
        sigmas = tensor_to_device(self._sigmas_t, device)
        return sigmas[t.long()]

    # ====================== Properties ======================
    @property
    def tmin(self) -> int:
        """Minimum valid timestep value (always 1 since t=0 is the original data).

        Returns:
            int: Minimum timestep value (1).
        """
        return 1

    @property
    def tmax(self) -> int:
        """Maximum valid timestep value (total number of time steps).

        Returns:
            int: Maximum timestep value (time_steps).
        """
        return self._time_steps

    @property
    def betas(self) -> torch.Tensor:
        """β values for the noise schedule.

        These control how much noise is added at each timestep.

        Returns:
            torch.Tensor: β values of shape (time_steps,).
        """
        return self._betas_t

    @property
    def alphas(self) -> torch.Tensor:
        """α values for the noise schedule, where α_t = 1 - β_t.

        These represent how much of the original signal is preserved at each timestep.

        Returns:
            torch.Tensor: α values of shape (time_steps,).
        """
        return self._alphas_t

    @property
    def alphas_bar(self) -> torch.Tensor:
        """Returns the cumulative product of α values: ᾱ_t = Π_{s=1}^t α_s.

        The ᾱ_t values represent the total signal preservation up to timestep t.

        Returns:
            torch.Tensor: ᾱ_t values of shape (time_steps + 1,), where:
                - ᾱ_0 = 1.0 (no noise)
                - ᾱ_1 = α_1 = 1-β_1
                - ...
                - ᾱ_T = α_1 * α_2 * ... * α_T
        """
        return self._alphas_bar_t

    @property
    def sigmas(self) -> torch.Tensor:
        """Returns the noise scale values σ_t = √(1 - ᾱ_t) for all timesteps.

        The σ_t values represent the standard deviation of the noise added up to timestep t.

        Returns:
            torch.Tensor: σ_t values of shape (time_steps + 1,), where:
                - σ_0 = 0.0 (no noise)
                - σ_1 = √(1-ᾱ_1)
                - ...
                - σ_T = √(1-ᾱ_T)
        """
        return self._sigmas_t

    # ====================== Static Methods ======================
    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
        """
        Improved cosine schedule from Nichol & Dhariwal, 2021.
        Args:
            timesteps (int): Number of diffusion steps.
            s (float): Small offset to prevent singularity.
        Returns:
            np.ndarray: Beta schedule of shape (timesteps,).
        """
        steps = timesteps + 1
        t = np.linspace(0, timesteps, steps) / timesteps
        alphas_bar = np.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = np.clip(betas, a_min=0, a_max=0.999)
        return betas

    @staticmethod
    def _linear_beta_schedule(
        timesteps: int, beta_start: float, beta_end: float
    ) -> np.ndarray:
        """
        Linear beta schedule as in Ho et al., 2020.
        Args:
            timesteps (int): Number of diffusion steps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
        Returns:
            np.ndarray: Beta schedule of shape (timesteps,).
        """
        return np.linspace(beta_start, beta_end, timesteps)
