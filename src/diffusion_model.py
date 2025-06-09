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
from .mlp import MLP, SimpleLinearModel, zero_out_model_parameters
from .network_base import NetworkBase
from .reverse_diffusion import ReverseDiffusion
from .unet import UNet1D
from .utils import bcast_right, prepare_batch_shape, set_seed, tensor_to_device


class DiffusionModel(NetworkBase):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse diffusion (denoising)
    processes for SNP data. The forward process gradually adds noise to the data following
    a predefined schedule, while the reverse process learns to denoise the data using a
    UNet1D architecture.

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

        # UNet1D or MLP or SimpleLinearModel for noise prediction
        self.unet = SimpleLinearModel(
            embedding_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            with_pos_emb=hparams["unet"].get("with_pos_emb", True),
            resnet_block_groups=hparams["unet"]["resnet_block_groups"],
            seq_length=hparams["data"]["seq_length"],
        )

        # ForwardDiffusion: Forward diffusion process
        self.forward_diffusion = ForwardDiffusion(
            diffusion_steps=hparams["diffusion"]["timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
            schedule_type=hparams["diffusion"]["schedule_type"],
        )

        # ReverseDiffusion: Reverse diffusion process
        self.reverse_diffusion = ReverseDiffusion(
            self.forward_diffusion,
            self.unet,
            self._data_shape,
            denoise_step=hparams["diffusion"].get("denoise_step", 10),
            discretize=hparams["diffusion"].get("discretize", False),
        )

        # Zero noise model
        # zero_out_model_parameters(self.unet)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.unet, "gradient_checkpointing_enable"):
            self.unet.gradient_checkpointing_enable()
        else:
            print("Warning: `gradient_checkpointing_enable()` not found. Skipping...")

    # ==================== Training Methods ====================
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x: Input data of shape [B, C, seq_len].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        return self.predict_added_noise(x, t)

    def predict_added_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise that was added during forward diffusion.

        Args:
            x: Noisy input data of shape [B, C, seq_len].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Ensure tensors have the correct shape and are on the right device
        device = x.device
        x = prepare_batch_shape(x)
        t = tensor_to_device(t, device)

        # Pass through the UNet model to predict noise
        return self.unet(x, t)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between true noise and predicted noise for a batch.
        This method performs the forward diffusion process, predicts noise,
        and calculates the loss in a single, clear function.

        Implements DDPM Eq. 4:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        and the loss:
            L = E[||eps - eps_theta(x_t, t)||^2]

        Args:
            batch: Input batch from dataloader of shape [B, C, seq_len].
        Returns:
            torch.Tensor: MSE loss.
        """
        # Ensure batch has the correct shape and is on the right device
        device = batch.device
        batch = prepare_batch_shape(batch)

        # Sample random timesteps for each batch element
        t = tensor_to_device(self.time_sampler.sample(shape=(batch.shape[0],)), device)

        # Generate Gaussian noise
        eps = torch.randn_like(batch, device=device)

        # Forward diffusion: add noise to the batch
        xt = self.forward_diffusion.sample(batch, t, eps)

        # Predict the noise using the model
        pred_eps = self.predict_added_noise(xt, t)

        # Get sigma_t for each timestep
        sigma_t = self.forward_diffusion.sigma(t)

        # Broadcast sigma_t to match dimensions of pred_eps
        sigma_t = bcast_right(sigma_t, pred_eps.ndim)

        # Scale predicted noise by 1/sigma_t before computing MSE
        # This implements the supervisor's recommendation: MSE(true_noise, predicted_noise / sigma_t)
        scaled_pred_eps = pred_eps / sigma_t

        # Compute and return MSE loss
        return F.mse_loss(scaled_pred_eps, eps)

    # ==================== Inference Methods ====================
    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes loss at specific timesteps.
        Args:
            x0: Clean input data of shape [B, C, seq_len].
            eps: Noise of shape [B, C, seq_len].
            timesteps: Timesteps to compute loss at.
        Returns:
            torch.Tensor: Loss at each timestep.
        """
        # Ensure tensors have the correct shape and are on the right device
        device = x0.device
        x0 = prepare_batch_shape(x0)
        eps = prepare_batch_shape(eps)

        losses = []
        for t in timesteps:
            # Create tensor of timestep t for all batch elements
            t_tensor = tensor_to_device(
                torch.full((x0.shape[0],), int(t.item()), dtype=torch.int32), device
            )

            # Apply forward diffusion at timestep t
            xt = self.forward_diffusion.sample(x0, t_tensor, eps)

            # Predict noise
            predicted_noise = self.predict_added_noise(xt, t_tensor)

            # Get sigma_t for the current timestep
            sigma_t = self.forward_diffusion.sigma(t_tensor)
            # Broadcast sigma_t to match dimensions of predicted_noise
            sigma_t = bcast_right(sigma_t, predicted_noise.ndim)

            # Scale predicted noise by 1/sigma_t before computing MSE
            scaled_pred_noise = predicted_noise / sigma_t

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
