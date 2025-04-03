#!/usr/bin/env python
# coding: utf-8

"""Diffusion model implementation for SNP data."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import DDPM, UNet1D, UniformContinuousTimeSampler
from .network_base import NetworkBase


class DiffusionModel(NetworkBase):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse diffusion (denoising)
    processes for SNP data. The forward process gradually adds noise to the data following
    a predefined schedule, while the reverse process learns to denoise the data using a
    UNet1D architecture.

    Inherits from NetworkBase to leverage PyTorch Lightning functionality.
    """

    def __init__(self, config: Dict):
        """Initialize the diffusion model with configuration.

        Args:
            config: Dictionary containing model configuration.
        """
        super().__init__(config)

        # Set data shape
        self._data_shape = (config["unet"]["channels"], config["data"]["seq_length"])

        # Initialize components from configuration
        self._forward_diffusion = DDPM(
            num_diffusion_timesteps=config["diffusion"]["num_diffusion_timesteps"],
            beta_start=config["diffusion"]["beta_start"],
            beta_end=config["diffusion"]["beta_end"],
        )

        self._time_sampler = UniformContinuousTimeSampler(
            tmin=config["time_sampler"]["tmin"], tmax=config["time_sampler"]["tmax"]
        )

        self.unet = UNet1D(
            embedding_dim=config["unet"]["embedding_dim"],
            dim_mults=config["unet"]["dim_mults"],
            channels=config["unet"]["channels"],
            with_time_emb=config["unet"]["with_time_emb"],
            resnet_block_groups=config["unet"]["resnet_block_groups"],
            seq_length=config["data"]["seq_length"],
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.unet, "gradient_checkpointing_enable"):
            self.unet.gradient_checkpointing_enable()
        else:
            print(
                "Warning: `gradient_checkpointing_enable()` not found in UNet1D. Skipping..."
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the diffusion model.

        Args:
            batch: Input batch of shape [B, C, seq_len].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Sample time and noise
        t = self._time_sampler.sample(shape=(batch.shape[0],))
        # Move time tensor to correct device
        t = t.to(batch.device)
        eps = torch.randn_like(batch)
        # Forward diffusion process
        xt = self._forward_diffusion.sample(batch, t, eps)
        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(xt.shape) == 2:  # If shape is (batch_size, seq_len)
            xt = xt.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        elif xt.shape[1] != 1:  # If incorrect number of channels
            xt = xt[:, :1, :]  # Force to 1 channel
        # Predict noise added during forward diffusion
        return self.predict_added_noise(xt, t)

    def predict_added_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise that was added during forward diffusion.

        Args:
            x: Noisy input data of shape [B, C, seq_len].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Ensure x has the correct shape for UNet input
        if len(x.shape) == 2:  # If shape is (batch_size, seq_len)
            x = x.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        # print(f"Noise prediction input shape: {x.shape}")
        # Run through UNet
        pred_noise = self.unet(x, t)
        # print(f"Noise prediction output shape: {pred_noise.shape}")
        return pred_noise

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute MSE between true noise and predicted noise.
        The network's goal is to correctly predict noise (eps) from noisy observations.
        xt = alpha(t) * x0 + sigma(t)**2 * eps

        Args:
            batch: Input batch from dataloader of shape [B, C, seq_len].

        Returns:
            torch.Tensor: MSE loss.
        """
        # t = self._time_sampler.sample(shape=(batch.shape[0],))  # sample time
        # eps = torch.randn_like(batch)                           # sample noise
        # xt = self._forward_diffusion.sample(batch, t, eps)      # add noise
        # pred_noise = self.unet(xt, t)                           # predict noise
        # loss = torch.mean((pred_noise - eps) ** 2)              # compute loss

        # Sample true noise (sample noise same as in forward function)
        eps = torch.randn_like(batch)

        # Get model predicted noise (sample time, sample noise, add & predict noise)
        pred_eps = self.forward(batch)

        # Compute MSE loss
        loss = F.mse_loss(pred_eps, eps)

        return loss

    def loss_per_timesteps(
        self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Computes loss at specific timesteps.

        Args:
            x0: Clean input data of shape [B, C, seq_len].
            eps: Noise of shape [B, C, seq_len].
            timesteps: Timesteps to compute loss at.

        Returns:
            torch.Tensor: Loss at each timestep.
        """
        losses = []
        for t in timesteps:
            t = int(t.item()) * torch.ones((x0.shape[0],), dtype=torch.int32)
            xt = self._forward_diffusion.sample(x0, t, eps)
            predicted_noise = self.predict_added_noise(xt, t)
            loss = F.mse_loss(predicted_noise, eps)
            losses.append(loss)
        return torch.stack(losses)

    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion step to estimate x_{t-1} given x_t.

        Computes parameters of a Gaussian p(x_{t-1}| x_t, x0_pred),
        DDPM Sampling Algorithm 2: It formalizes the whole generative procedure.

        Args:
            xt: Noisy data at timestep t.
            t: Current timestep.

        Returns:
            torch.Tensor: Denoised data at timestep t-1.
        """
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=xt.device)
        # Get predicted noise from the U-Net
        eps_pred = self.predict_added_noise(xt, t)
        # Handle the case where t > 1 for all elements in batch
        is_t_greater_than_one = (t > 1).all()
        if is_t_greater_than_one:
            sqrt_a_t = self._forward_diffusion.alpha(t) / self._forward_diffusion.alpha(
                t - 1
            )
        else:
            sqrt_a_t = self._forward_diffusion.alpha(t)
        
        # Ensure all parameters are on the same device as xt
        sqrt_a_t = sqrt_a_t.to(xt.device)
        
        # Perform the denoising step to take the snp from t to t-1
        inv_sqrt_a_t = (1.0 / sqrt_a_t).to(xt.device)
        beta_t = (1.0 - sqrt_a_t**2).to(xt.device)
        inv_sigma_t = (1.0 / self._forward_diffusion.sigma(t)).to(xt.device)

        # Add proper broadcasting for all scalar tensors
        inv_sqrt_a_t = inv_sqrt_a_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        beta_t = beta_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        inv_sigma_t = inv_sigma_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]

        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        # DDPM instructs to use either the variance of the forward process
        # or the variance of posterior q(x_{t-1}|x_t, x_0). Former is easier.
        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt)
        # The reparameterization trick: N(mean, variance^2) = mean + std(sigma) * epsilon
        return mean + std * z

    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the learned reverse diffusion process.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        with torch.no_grad():
            # Start with random noise
            x = torch.randn((num_samples,) + self._data_shape, device=self.device)

            # Track statistics for debugging
            print(
                f"Initial noise stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}"
            )

            # Reverse diffusion process
            for t in range(self._forward_diffusion.tmax, 0, -1):
                if t % 100 == 0:  # Print progress every 100 steps
                    print(
                        f"Step {t} stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}"
                    )
                x = self._reverse_process_step(x, t)

                # Clamp intermediate values to prevent explosion
                x = torch.clamp(x, -5.0, 5.0)

            # Final normalization to [0, 1]
            x = torch.clamp(x, 0, 1)

            print(
                f"Final sample stats - mean: {x.mean().item():.4f}, std: {x.std().item():.4f}"
            )
            return x
