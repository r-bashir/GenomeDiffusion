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
    forward_diff = ForwardDiffusion(diffusion_steps=1000, schedule_type='cosine')
    noisy_sample = forward_diff.sample(x0, timestep, noise)

Note: This class only implements the forward (noising) process q(x_t|x_0), NOT the full DDPM model.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .utils import bcast_right, prepare_batch_shape, tensor_to_device


# Alternative implementation of ForwardDiffusion class. Difference is that it uses torch.nn.Module
# and registers the schedule tensors as buffers using register_buffer method. There is no need to
# explicitly move the tensors to the same device as the input data since the module will move them
# automatically. Maybe used in future in diffusion_models.py.
class NoiseSchedule(nn.Module):
    """
    DDPM forward (noising) process scheduler.

    Adds Gaussian noise to data using a fixed beta schedule (linear or improved cosine).
    All schedule tensors are registered as buffers and move automatically with the module.

    Args:
        diffusion_steps (int): Number of diffusion steps (T).
        beta_start (float): Start value for linear beta schedule.
        beta_end (float): End value for linear beta schedule.
        schedule_type (str): 'cosine' (Nichol & Dhariwal, 2021) or 'linear' (Ho et al., 2020).

    Attributes:
        _betas_t: Beta schedule, shape (T,)
        _alphas_t: Alpha schedule (1-beta), shape (T,)
        _alphas_bar_t: Cumulative product of alphas, shape (T+1,)
        _sigmas_t: Noise stddev at each step, shape (T+1,)
    """

    def __init__(
        self,
        time_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        """
        Args:
            time_steps: Number of diffusion steps (T)
            beta_start: Start value for linear schedule
            beta_end: End value for linear schedule
            schedule_type: 'cosine' or 'linear'
        """

        # Initialize parameters
        self._time_steps = time_steps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._schedule_type = schedule_type

        # Select beta schedule
        if schedule_type == "linear":
            betas_np = self.linear_beta_schedule(
                self._time_steps, self._beta_start, self._beta_end
            )
        elif schedule_type == "cosine":
            betas_np = self.cosine_beta_schedule(self._time_steps)
        else:
            raise ValueError(
                f"Unknown schedule_type '{schedule_type}'. Use 'cosine' or 'linear'."
            )

        # Calculate alphas and cumulative alphas
        alphas_np = 1.0 - betas_np

        # Calculate cumulative product of alphas, prepend 1 for t=0 (no noise) for convenient indexing
        alphas_bar_np = np.concatenate(([1.0], np.cumprod(alphas_np)))

        # Calculate sigmas
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
            eps: Optional pre-generated noise of same shape as x0. If None, noise is sampled.

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

        # Generate noise if not provided
        eps = tensor_to_device(eps, device)

        # Compute noisy sample
        x_t = torch.sqrt(alpha_bar_t) * x0 + sigma_t * eps
        return x_t

    # ========================== Properties
    @property
    def tmin(self) -> int:
        """Minimum valid timestep value (always 1 since t=0 is the original data).

        Returns:
            int: Minimum timestep value (1).
        """
        return 1

    @property
    def tmax(self) -> int:
        """Maximum valid timestep value (total number of diffusion steps).

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

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
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
        betas = np.clip(betas, a_min=1e-8, a_max=0.999)
        return betas

    @staticmethod
    def linear_beta_schedule(
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
