#!/usr/bin/env python
# coding: utf-8

"""Diffusion model implementation for SNP data."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .all_models import UniformContinuousTimeSampler
from .forward_diffusion import ForwardDiffusion
from .mlp import (
    ComplexNoisePredictor,
    LinearNoisePredictor,
    SimpleNoisePredictor,
)
from .network_base import NetworkBase
from .reverse_diffusion import ReverseDiffusion
from .unet import UNet1D
from .utils import bcast_right, tensor_to_device


class DiffusionModel(NetworkBase):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse
    diffusion (denoising) processes for SNP data. The forward process
    gradually adds noise to the data following a predefined schedule,
    while the reverse process learns to denoise the data using a UNet.

    Inherits from NetworkBase to leverage PyTorch Lightning functionality.
    """

    def __init__(self, hparams: Dict):
        super().__init__(hparams)

        # Set data shape
        self._data_shape = (hparams["unet"]["channels"], hparams["data"]["seq_length"])

        # Continuous time sampler
        self.time_sampler = UniformContinuousTimeSampler(
            tmin=hparams["time_sampler"]["tmin"], tmax=hparams["time_sampler"]["tmax"]
        )

        # ForwardDiffusion: Forward diffusion process
        self.forward_diffusion = ForwardDiffusion(
            time_steps=hparams["diffusion"]["timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
            schedule_type=hparams["diffusion"]["schedule_type"],
        )

        # Noise Predictor
        self.noise_predictor = LinearNoisePredictor(
            embedding_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            with_pos_emb=hparams["unet"].get("with_pos_emb", True),
            norm_groups=hparams["unet"]["norm_groups"],
            seq_length=hparams["data"]["seq_length"],
        )

        # ReverseDiffusion: Reverse diffusion process
        self.reverse_diffusion = ReverseDiffusion(
            self.forward_diffusion,
            self.noise_predictor,
            self._data_shape,
            denoise_step=hparams["diffusion"].get("denoise_step", 1),
            discretize=hparams["diffusion"].get("discretize", False),
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.noise_predictor, "gradient_checkpointing_enable"):
            self.noise_predictor.gradient_checkpointing_enable()
        else:
            print("Warning: `gradient_checkpointing_enable()` not found. Skipping...")

    # ==================== Training Methods ====================
    def forward(self, batch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            batch: Input data of shape [B, 1, L].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, 1, L].
        """
        return self.predict_added_noise(batch, t)

    def predict_added_noise(self, batch: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise that was added during forward diffusion.

        Args:
            batch: Noisy input data of shape [B, 1, L].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, 1, L].
        """
        # Ensure tensors have the correct shape of [B, 1, L]
        device = batch.device
        batch = self._prepare_batch(batch)
        t = tensor_to_device(t, device)

        # Pass through the noise predictor to predict noise
        return self.noise_predictor(batch, t)

    def compute_loss_Ho(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between true noise and predicted noise for a batch.
        This method performs the forward diffusion process, predicts noise,
        and calculates the loss in a single, clear function.

        Implements DDPM Eq. 4:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        and the loss:
            L = E[||eps - eps_theta(x_t, t)||^2]

        Args:
            batch: Input batch from dataloader of shape [B, 1, L].
        Returns:
            torch.Tensor: MSE loss.
        """
        # Ensure batch has the correct shape of [B, 1, L]
        device = batch.device
        batch = self._prepare_batch(batch)

        # Sample random timesteps for each batch element
        t = tensor_to_device(self.time_sampler.sample(shape=(batch.shape[0],)), device)

        # Generate Gaussian noise
        eps = torch.randn_like(batch, device=device)

        # Forward diffusion: add noise to the batch
        xt = self.forward_diffusion.sample(batch, t, eps)

        # ε_θ(x_t, t): Model's prediction of the noise added at timestep t
        eps_theta = self.predict_added_noise(xt, t)

        # Get σ_t = √(1 - ᾱ_t) at each timestep
        sigma_t = self.forward_diffusion.sigma(t)

        # Broadcast sigma_t to match dimensions of pred_eps
        sigma_t = bcast_right(sigma_t, eps_theta.ndim)

        # Scale predicted noise by 1/σ_t before computing MSE
        # i.e. MSE(true_noise, ε_θ(x_t, t)/σ_t)
        scaled_pred_eps = eps_theta / sigma_t

        # Compute and return MSE loss
        return F.mse_loss(eps, scaled_pred_eps)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between (xt - x0) and predicted noise for a batch.
        This method performs the forward diffusion process, predicts noise,
        and calculates the loss. One can scale the MSE with a scale factor.

        Args:
            batch: Input batch from dataloader of shape [B, 1, L].
        Returns:
            torch.Tensor: MSE loss.
        """
        # Ensure batch has the correct shape of [B, 1, L]
        device = batch.device
        batch = self._prepare_batch(batch)

        # Sample random timesteps for each batch element
        t = tensor_to_device(self.time_sampler.sample(shape=(batch.shape[0],)), device)

        # Generate Gaussian noise
        eps = torch.randn_like(batch, device=device)

        # Forward diffusion: add noise to the batch
        xt = self.forward_diffusion.sample(batch, t, eps)

        # ε_θ(x_t, t): Model's prediction of the noise added at timestep t
        eps_theta = self.predict_added_noise(xt, t)

        # Elementwise MSE loss
        mse = F.mse_loss(eps_theta, (xt - batch), reduction="none")  # shape: [B, 1, L]

        # Scale MSE by (1 - ᾱ_t)
        alpha_bar_t = self.forward_diffusion.alpha_bar(t)
        alpha_bar_t = bcast_right(alpha_bar_t, eps_theta.ndim)  # get shape: [B, 1, L]
        scaled_mse = mse / (1 - alpha_bar_t)

        # Aggregate to scalar (mean over all elements)
        scaled_loss = scaled_mse.mean()

        return mse.mean()

    # ==================== Inference Methods ====================
    def loss_per_timesteps_Ho(
        self, batch: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes loss at specific timesteps.
        Args:
            batch: Clean input data of shape [B, 1, L].
            eps: Noise of shape [B, 1, L].
            timesteps: Timesteps to compute loss at.
        Returns:
            torch.Tensor: Loss at each timestep.
        """
        # Ensure tensors have the correct shape of [B, 1, L]
        device = batch.device
        batch = self._prepare_batch(batch)
        eps = self._prepare_batch(eps)

        losses = []
        for t in timesteps:
            # Create tensor of timestep t for all batch elements
            t_tensor = tensor_to_device(
                torch.full((batch.shape[0],), int(t.item()), dtype=torch.int32), device
            )

            # Apply forward diffusion at timestep t
            xt = self.forward_diffusion.sample(batch, t_tensor, eps)

            # Predict noise
            eps_theta = self.predict_added_noise(xt, t_tensor)

            # Get sigma_t for the current timestep
            sigma_t = self.forward_diffusion.sigma(t_tensor)

            # Broadcast sigma_t to match dimensions of predicted_noise
            sigma_t = bcast_right(sigma_t, eps_theta.ndim)

            # Scale predicted noise by 1/sigma_t before computing MSE
            scaled_pred_noise = eps_theta / sigma_t

            # Compute loss using scaled predicted noise
            loss = F.mse_loss(scaled_pred_noise, eps)
            losses.append(loss)

        return torch.stack(losses)

    def generate_samples(
        self,
        num_samples: int = 10,
        denoise_step: int = 10,
        discretize: bool = False,
        seed: int = 42,
        device=None,
    ) -> torch.Tensor:
        """
        Generate new samples from random noise using the reverse diffusion process.

        This method implements Algorithm 2 from Ho et al., 2020 (DDPM paper):
            For t = T,...,1:
                x_{t-1} ~ p_θ(x_{t-1}|x_t)
            where p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), β_t I)

        Args:
            num_samples: Number of samples to generate.
            denoise_step: Number of timesteps to skip in reverse diffusion (default: 10).
            discretize: If True, discretize the output to 0, 0.5, and 1.0 values (SNP genotypes).
            seed: Optional seed for reproducible sample generation.
            device: torch.device

        Returns:
            torch.Tensor: Generated samples of shape [num_samples, C, seq_len].
        """
        # Determine device to use
        if device is None:
            device = (
                self.device
                if hasattr(self, "device")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # Use the reverse diffusion process to generate samples
        return self.reverse_diffusion.generate_samples(
            num_samples=num_samples,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=seed,
            device=device,
        )

    def denoise_sample(
        self,
        batch: torch.Tensor,
        denoise_step: int = 10,
        discretize: bool = False,
        seed: int = 42,
        device=None,
    ) -> torch.Tensor:
        """
        Denoise an input batch using the reverse diffusion process.

        This method starts the reverse diffusion process from the provided batch
        (which should be noisy or real data). It does not generate new noise, but
        instead denoises the actual input batch by iteratively applying the reverse
        diffusion steps. This is useful for denoising specific data samples, e.g.,
        for evaluation or restoration tasks.

        Args:
            batch: Input batch to denoise. Shape: [B, C, seq_len]. This can be noisy or real data.
            denoise_step: Number of timesteps to skip in reverse diffusion. If None, uses the instance default.
            discretize: If True, discretize output to SNP values. If None, uses the instance default.
            seed: Random seed for reproducibility (not used in this function, kept for API compatibility).
            device: Device to run the computation on. If None, uses CUDA if available, else CPU.

        Returns:
            Denoised batch of the same shape as input.
        """
        # Determine device to use
        if device is None:
            device = (
                self.device
                if hasattr(self, "device")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # Use the reverse diffusion process to denoise the batch
        return self.reverse_diffusion.denoise_sample(
            batch=batch,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=seed,
            device=device,
        )
