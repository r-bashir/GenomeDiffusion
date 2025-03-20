#!/usr/bin/env python
# coding: utf-8

"""Diffusion model implementation for SNP data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional

from network_base import NetworkBase
from test_models import DDPM, UNet1D, UniformDiscreteTimeSampler
from diffusion import SNPDataset


class DiffusionModel(NetworkBase):
    """Diffusion model with 1D Convolutional network for SNP data.
    
    Implements both forward diffusion (data corruption) and reverse diffusion (denoising)
    processes for SNP data. The forward process gradually adds noise to the data following
    a predefined schedule, while the reverse process learns to denoise the data using a
    UNet1D architecture.
    
    Inherits from NetworkBase to leverage PyTorch Lightning functionality.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the diffusion model with configuration.
        
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
        
        self._time_sampler = UniformDiscreteTimeSampler(
            tmin=config["time_sampler"]["tmin"], 
            tmax=config["time_sampler"]["tmax"]
        )
        
        self.unet = UNet1D(
            embedding_dim=config["unet"]["embedding_dim"],
            dim_mults=config["unet"]["dim_mults"],
            channels=config["unet"]["channels"],
            with_time_emb=config["unet"]["with_time_emb"],
            resnet_block_groups=config["unet"]["resnet_block_groups"],
            seq_length=config["data"]["seq_length"],
        )
        
        # Enable gradient checkpointing if specified
        if config['training'].get('gradient_checkpointing', True):
            if hasattr(self.unet, "gradient_checkpointing_enable"):
                self.unet.gradient_checkpointing_enable()
            else:
                print("Warning: `gradient_checkpointing_enable()` not found in UNet1D. Skipping...")
    
    def _create_dataset(self) -> Dataset:
        """Create and return the SNP dataset.
        
        Returns:
            Dataset: The created SNP dataset.
        """
        return SNPDataset(self.config['input_path'])
    
    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Prepare a batch for model input.
        
        Ensures the batch has the correct shape [B, C, seq_len].
        
        Args:
            batch: Input batch from dataloader.
            
        Returns:
            torch.Tensor: Prepared batch with shape [B, C, seq_len].
        """
        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(batch.shape) == 2:  # If shape is (batch_size, seq_len)
            batch = batch.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        
        return batch.to(self.device)
    
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
        
        # Print input shape for debugging
        # print(f"Noise prediction input shape: {x.shape}")
        
        # Run through UNet
        pred_noise = self.unet(x, t)
        
        # Print output shape for debugging
        # print(f"Noise prediction output shape: {pred_noise.shape}")
        
        return pred_noise
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the diffusion model.
        
        Args:
            batch: Input batch of shape [B, C, seq_len].
            
        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        return self.forward_step(batch)
    
    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.
        
        Args:
            batch: Input batch of shape [B, C, seq_len].
            
        Returns:
            torch.Tensor: Predicted noise of shape [B, C, seq_len].
        """
        # Sample time and noise
        t = self._time_sampler.sample(shape=(batch.shape[0],))
        eps = torch.randn_like(batch)
        
        # Forward diffusion process
        xt = self._forward_diffusion.sample(batch, t, eps)
        
        # Debugging print statements
        # print(f"Mean abs diff between x0 and xt: {(batch - xt).abs().mean().item()}")  # Check noise level
        
        # Ensure input has correct shape (batch_size, 1, seq_len)
        if len(xt.shape) == 2:  # If shape is (batch_size, seq_len)
            xt = xt.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        elif xt.shape[1] != 1:  # If incorrect number of channels
            # print(f"Unexpected number of channels: {xt.shape[1]}, reshaping...")
            xt = xt[:, :1, :]  # Force to 1 channel
        
        # print(f"Final shape before UNet: {xt.shape}")
        
        # Predict noise added during forward diffusion
        return self.predict_added_noise(xt, t)
    
    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute MSE between true noise and predicted noise.
        
        The network's goal is to correctly predict noise (eps) from noisy observations.
        xt = alpha(t) * x0 + sigma(t)**2 * eps
        
        Args:
            batch: Input batch from dataloader of shape [B, C, seq_len].
            
        Returns:
            torch.Tensor: MSE loss.
        """
        # Sample true noise
        eps = torch.randn_like(batch)
        
        # Get model predictions
        pred_eps = self.forward_step(batch)
        
        # Compute MSE loss
        return torch.mean((pred_eps - eps) ** 2)
    
    def loss_per_timesteps(self, x0: torch.Tensor, eps: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
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
            loss = torch.mean((predicted_noise - eps) ** 2)
            losses.append(loss)
        
        # print(f"Loss across timesteps: {torch.stack(losses).detach().cpu().numpy()}")
        return torch.stack(losses)
    
    def _reverse_process_step(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse diffusion step to estimate x_{t-1} given x_t.
        
        Computes parameters of a Gaussian p(x_{t-1}| x_t, x0_pred),
        DDPM sampling method - algorithm 1: It formalizes the whole generative procedure.
        
        Args:
            xt: Noisy data at timestep t.
            t: Current timestep.
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1.
        """
        device = self.device
        xt = xt.to(device)
        t = t * torch.ones((xt.shape[0],), dtype=torch.int32, device=device)
        
        eps_pred = self.predict_added_noise(xt, t)
        
        if t > 1:
            sqrt_a_t = self._forward_diffusion.alpha(t) / self._forward_diffusion.alpha(t - 1)
        else:
            sqrt_a_t = self._forward_diffusion.alpha(t)
        
        inv_sqrt_a_t = 1.0 / sqrt_a_t
        beta_t = 1.0 - sqrt_a_t**2
        inv_sigma_t = 1.0 / self._forward_diffusion.sigma(t)
        
        mean = inv_sqrt_a_t * (xt - beta_t * inv_sigma_t * eps_pred)
        
        # DDPM instructs to use either the variance of the forward process
        # or the variance of posterior q(x_{t-1}|x_t, x_0). Former is easier.
        std = torch.sqrt(beta_t)
        z = torch.randn_like(xt, device=device)
        
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
            x = torch.randn((num_samples,) + self._data_shape, device=self.device)
            
            for t in range(self._forward_diffusion.tmax, 0, -1):
                x = self._reverse_process_step(x, t)
                
                if t % 100 == 0:
                    print(f"Sampling at timestep {t}, mean: {x.mean().item()}, std: {x.std().item()}")
            
            x = torch.clamp(x, 0, 1)
        
        print(f"Final sample mean: {x.mean().item()}, std: {x.std().item()}")
        return x
