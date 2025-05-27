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
from .mlp import MLP, zero_out_model_parameters
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

        # UNet1D or MLP for noise prediction
        self.unet = MLP(
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
            self.forward_diffusion, self.unet, self._data_shape
        )

        # Zero noise model
        # zero_out_model_parameters(self.unet)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.unet, "gradient_checkpointing_enable"):
            self.unet.gradient_checkpointing_enable()
        else:
            print("Warning: `gradient_checkpointing_enable()` not found. Skipping...")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x: Input data of shape [B, C, seq_len].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Ensure input and timestep are on the same device and have correct shape
        device = x.device
        x = prepare_batch_shape(x)
        t = tensor_to_device(t, device)
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
        # Ensure input and timestep are on the same device and have correct shape
        device = x.device
        x = prepare_batch_shape(x)
        t = tensor_to_device(t, device)

        # Pass through the UNet model to predict noise
        return self.unet(x, t)

    def diffuse_and_predict(self, batch: torch.Tensor) -> tuple:
        """
        Apply forward diffusion and predict noise.

        This method handles the complete process of:
        1. Sampling timesteps
        2. Sampling noise
        3. Adding noise to the input (forward diffusion)
        4. Predicting the noise

        Args:
            batch: Clean input batch of shape [B, C, seq_len].

        Returns:
            tuple: (predicted_noise, true_noise, timesteps, noisy_input)
        """
        # Ensure batch has the correct shape and is on the right device
        device = batch.device
        batch = prepare_batch_shape(batch)

        # Sample time and noise
        t = tensor_to_device(self.time_sampler.sample(shape=(batch.shape[0],)), device)

        # Generate Gaussian noise
        eps = torch.randn_like(batch, device=device)

        # Forward diffusion process (DDPM Eq. 4):
        #   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        xt = self.forward_diffusion.sample(batch, t, eps)

        # Predict the noise using the model
        pred_eps = self.predict_added_noise(xt, t)

        return pred_eps, eps, t, xt

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.
        Implements DDPM Eq. 4:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        and the loss:
            L = E[||eps - eps_theta(x_t, t)||^2]
        Args:
            batch: Input batch from dataloader of shape [B, C, seq_len].
        Returns:
            torch.Tensor: MSE loss.
        """
        pred_eps, eps, _, _ = self.diffuse_and_predict(batch)
        loss = F.mse_loss(pred_eps, eps)
        return loss

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
        # Ensure inputs have the correct shape and are on the right device
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

            # Predict noise and compute loss
            predicted_noise = self.forward(xt, t_tensor)
            loss = F.mse_loss(predicted_noise, eps)
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
        Denoise an existing batch using the reverse diffusion process.

        This method takes an existing batch (which could be noisy data or even
        clean data that you want to process through the model) and applies the
        reverse diffusion process to it.

        Args:
            batch: Input batch to denoise, shape [B, C, seq_len].
            denoise_step: Number of timesteps to skip in reverse diffusion (default: 10).
            discretize: If True, discretize the output to 0, 0.5, and 1.0 values (SNP genotypes).
            seed: Optional seed for reproducible sample generation.
            device: torch.device

        Returns:
            torch.Tensor: Denoised samples of shape [B, C, seq_len].
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
