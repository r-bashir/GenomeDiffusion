#!/usr/bin/env python
# coding: utf-8

"""Diffusion model implementation for SNP data."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_base import NetworkBase
from .models import DDPM, UNet1D, UniformContinuousTimeSampler
from .mlp import MLP

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

        # DDPM: Forward diffusion process
        self.ddpm = DDPM(
            num_diffusion_timesteps=hparams["diffusion"]["num_diffusion_timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
        )

        # Replace UNet with DeepMLP for noise prediction
        self.unet = MLP(
            embedding_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            with_pos_emb=hparams["unet"].get("with_pos_emb", True),
            resnet_block_groups=hparams["unet"]["resnet_block_groups"],
            seq_length=hparams["data"]["seq_length"],
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.unet, "gradient_checkpointing_enable"):
            self.unet.gradient_checkpointing_enable()
        else:
            print("Warning: `gradient_checkpointing_enable()` not found. Skipping...")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the diffusion model.

        Args:
            x: Input data of shape [B, C, seq_len].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Predict noise
        return self.predict_added_noise(x, t)

    def diffuse_and_predict(self, batch: torch.Tensor) -> tuple:
        """Apply forward diffusion and predict noise.

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
        # Sample time and noise
        t = self.time_sampler.sample(shape=(batch.shape[0],))

        # Move time tensor to correct device
        t = t.to(batch.device)
        eps = torch.randn_like(batch)

        # Forward diffusion process
        xt = self.ddpm.sample(batch, t, eps)

        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(xt.shape) == 2:  # If shape is (batch_size, seq_len)
            xt = xt.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        elif xt.shape[1] != 1:  # If incorrect number of channels
            xt = xt[:, :1, :]  # Force to 1 channel

        # Predict noise added during forward diffusion
        pred_eps = self.predict_added_noise(xt, t)

        return pred_eps, eps, t, xt

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
        # Get predicted noise, true noise, timesteps, and noisy input
        pred_eps, eps, _, _ = self.diffuse_and_predict(batch)

        # Compute MSE loss between predicted and true noise
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
            # Create tensor of timestep t for all batch elements
            t_tensor = int(t.item()) * torch.ones(
                (x0.shape[0],), dtype=torch.int32, device=x0.device
            )

            # Forward diffusion to timestep t
            xt = self.ddpm.sample(x0, t_tensor, eps)

            # Predict noise
            predicted_noise = self.forward(xt, t_tensor)

            # Compute loss
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
        # Get predicted noise using the forward method
        eps_pred = self.forward(xt, t)

        # Handle the case where t > 1 for all elements in batch
        is_t_greater_than_one = (t > 1).all()
        if is_t_greater_than_one:
            sqrt_a_t = self.ddpm.alpha(t) / self.ddpm.alpha(t - 1)
        else:
            sqrt_a_t = self.ddpm.alpha(t)

        # [NaN-FIX] Add numerical stability to prevent NaN values
        sqrt_a_t = sqrt_a_t.to(xt.device)
        eps = 1e-12  # [NaN-FIX] Small constant for numerical stability

        # Perform the denoising step to take the snp from t to t-1
        # [NaN-FIX] Add eps to denominators to prevent division by zero
        inv_sqrt_a_t = (1.0 / (sqrt_a_t + eps)).to(xt.device)  # [NaN-FIX] Added eps
        beta_t = (1.0 - sqrt_a_t**2).to(xt.device)
        sigma_t = self.ddpm.sigma(t)
        inv_sigma_t = (1.0 / (sigma_t + eps)).to(xt.device)  # [NaN-FIX] Added eps

        # Add proper broadcasting for all scalar tensors
        inv_sqrt_a_t = inv_sqrt_a_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        beta_t = beta_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]
        inv_sigma_t = inv_sigma_t.view(-1, 1, 1)  # Shape: [batch_size, 1, 1]

        # [NaN-FIX] Calculate mean with gradient clipping to prevent extreme values
        # mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        noise_pred_term = beta_t * inv_sigma_t * eps_pred
        noise_pred_term = torch.clamp(
            noise_pred_term, -100, 100
        )  # [NaN-FIX] Prevent extreme values
        mean = inv_sqrt_a_t * (xt - noise_pred_term)

        # [NaN-FIX] Replace any non-finite values with zeros
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))

        # DDPM instructs to use either the variance of the forward process
        # or the variance of posterior q(x_{t-1}|x_t, x_0). Former is easier.
        std = torch.sqrt(
            torch.clamp(beta_t, min=eps)
        )  # [NaN-FIX] Ensure positive value before sqrt
        z = torch.randn_like(xt)

        # The reparameterization trick: N(mean, variance^2) = mean + std(sigma) * epsilon
        result = mean + std * z

        # [NaN-FIX] Final safety check and debugging info
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"Warning: NaN/Inf detected at step {t[0].item()}")
            print(
                f"Stats - mean: {mean.mean().item():.4f}, std: {std.mean().item():.4f}"
            )
            result = torch.where(
                torch.isfinite(result), result, torch.zeros_like(result)
            )

        return result

    def denoise_batch(
        self, batch: torch.Tensor, denoise_step: int = 10, discretize: bool = False
    ) -> torch.Tensor:
        """Run the reverse diffusion process to generate denoised samples.

        Performs the full reverse diffusion process starting from pure noise
        and progressively denoising through all timesteps to generate clean samples.

        Args:
            batch: Input batch of shape [B, C, seq_len], used for shape and device reference.
            discretize: If True, discretize the output to 0, 0.5, and 1.0 values (SNP genotypes).
            denoise_step: Number of timesteps to skip in reverse diffusion (default: 10). Smaller values
                         give more fine-grained denoising but take longer.

        Returns:
            torch.Tensor: Denoised (reconstructed) output of shape [B, C, seq_len].
        """
        with torch.no_grad():
            # Start from pure noise (or optionally from batch if you want conditional denoising)
            x = torch.randn_like(batch)

            # Print initial noise statistics
            # print(f"Initial noise stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

            # Reverse diffusion process
            print(
                f"Starting reverse diffusion from t={self.ddpm.tmax} to t={self.ddpm.tmin} with step {denoise_step}"
            )

            for t in reversed(range(self.ddpm.tmin, self.ddpm.tmax + 1, denoise_step)):
                t_tensor = torch.full(
                    (x.size(0),), t, device=x.device, dtype=torch.long
                )
                x = self._reverse_process_step(x, t_tensor)

                # Print statistics every 100 steps
                if t % 100 == 0 or t == self.ddpm.tmin:
                    # print(f"Step {t} stats - mean: {x.mean():.4f}, std: {x.std():.4f}")

                    # Print unique values to see if we're getting discrete distribution
                    if t <= 100:  # Only in final stages
                        unique_vals = torch.unique(x)
                        if len(unique_vals) < 20:  # Only if there aren't too many
                            print(f"Unique values at step {t}: {unique_vals}")

            # Clamp to valid range [0, 1]
            x = torch.clamp(x, 0, 1)

            # Discretize to SNP values if requested
            if discretize:
                # Round to nearest genotype (0, 0.5, 1.0)
                x = torch.round(x * 2) / 2

            # print(f"Final sample stats - mean: {x.mean():.4f}, std: {x.std():.4f}")
            return x

    def generate_samples(
        self, num_samples: int = 10, denoise_step: int = 10, discretize: bool = False
    ) -> torch.Tensor:
        """Generate samples from the learned reverse diffusion process.

        Args:
            num_samples: Number of samples to generate.
            denoise_step: Number of timesteps to skip in reverse diffusion (default: 10).
            discretize: If True, discretize the output to 0, 0.5, and 1.0 values (SNP genotypes).

        Returns:
            torch.Tensor: Generated samples.
        """
        # Create a dummy batch with the right shape for denoise_batch
        dummy_batch = torch.zeros((num_samples,) + self._data_shape, device=self.device)

        # Use the improved denoise_batch method
        return self.denoise_batch(
            dummy_batch, denoise_step=denoise_step, discretize=discretize
        )
