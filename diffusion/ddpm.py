#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Initializes the diffusion process.

        Args:
            diffusion_steps (int): Number of diffusion steps.
            beta_start (float): Initial beta value.
            beta_end (float): Final beta value.
        """
        self._diffusion_steps = diffusion_steps
        self._beta_start = beta_start
        self._beta_end = beta_end

        # Use liner beta scheduler
        self._betas = np.linspace(
            self._beta_start, self._beta_end, self._diffusion_steps
        )

        # Use cosine beta scheduler
        self._betas = self._cosine_beta_schedule(self._diffusion_steps)
        alphas_bar = self._get_alphas_bar()

        # Register tensors as buffers so they move with the model
        self.register_buffer(
            "_alphas", torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)
        )
        self.register_buffer(
            "_sigmas", torch.tensor(np.sqrt(1 - alphas_bar), dtype=torch.float32)
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.

        Args:
            timesteps: Number of diffusion timesteps
            s: Offset parameter (default: 0.008)

        Returns:
            np.ndarray: Beta schedule array of shape (timesteps,)
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    @property
    def tmin(self) -> int:
        """Minimum timestep value."""
        return 1

    @property
    def tmax(self) -> int:
        """Maximum timestep value."""
        return self._diffusion_steps

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
        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)
        # Convert to indices (0-indexed)
        idx = (t - 1).long()

        # Ensure idx is on the same device as _alphas
        if idx.device != self._alphas.device:
            idx = idx.to(self._alphas.device)

        # Return values on the correct device
        return self._alphas[idx]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves sigma(t) for the given time indices.

        Args:
            t (torch.Tensor): Timesteps (batch_size,).

        Returns:
            torch.Tensor: Sigma values corresponding to timesteps.
        """
        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)
        # Convert to indices (0-indexed)
        idx = (t - 1).long()

        # Ensure idx is on the same device as _sigmas
        if idx.device != self._sigmas.device:
            idx = idx.to(self._sigmas.device)

        # Return values on the correct device
        return self._sigmas[idx]

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the forward diffusion process q(xt | x0). It gives you the
        noisy version xt from x0 and t, without looping through every step.

        Args:
            x0 (torch.Tensor): Original clean input (batch_size, [channels,] seq_len).
            t (torch.Tensor): Diffusion timesteps (batch_size,).
            eps (torch.Tensor): Gaussian noise with same shape as x0.

        Returns:
            torch.Tensor: Noisy sample xt.
        """
        # Get alpha and sigma values for the timesteps
        alpha_values = self.alpha(t)
        sigma_values = self.sigma(t)

        # Move alpha and sigma to the same device as x0
        alpha_values = alpha_values.to(x0.device)
        sigma_values = sigma_values.to(x0.device)

        # Reshape alpha_t and sigma_t according to the input shape
        if len(x0.shape) == 3:  # [batch_size, channels, seq_len]
            alpha_t = alpha_values.view(-1, 1, 1)
            sigma_t = sigma_values.view(-1, 1, 1)
        else:  # [batch_size, seq_len]
            alpha_t = alpha_values.view(-1, 1)
            sigma_t = sigma_values.view(-1, 1)

        xt = alpha_t * x0 + sigma_t * eps
        return xt

    def register_buffer(self, name, tensor):
        """
        Registers a tensor as a buffer (similar to PyTorch's nn.Module.register_buffer).
        This is a simple implementation for a non-nn.Module class.
        """
        setattr(self, name, tensor)

    def to(self, device):
        """
        Moves all tensors to the specified device.
        This mimics the behavior of nn.Module.to() for compatibility with PyTorch Lightning.
        """
        self._alphas = self._alphas.to(device)
        self._sigmas = self._sigmas.to(device)
        return self


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load config
    config = load_config("config.yaml")
    input_path = config.get("input_path")
    print("\nInitializing dataset...")

    # Initialize dataset
    dataset = SNPDataset(input_path)

    # Initialize dataloader
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # Get batch
    batch = next(iter(train_loader))  # Shape: [B, seq_len]
    print(f"Batch shape [B, seq_len]: {batch.shape}")

    # Prepare input (ensure [B, C, L] format for both models)
    batch = batch.unsqueeze(1).to(device)  # [B, 1, seq_len]
    print(f"Batch shape [B, C, seq_len]: {batch.shape}")

    # ------------------ Test DDPM
    print("\nTesting DDPM...")
    forward_diffusion = DDPM(diffusion_steps=1000, beta_start=0.0001, beta_end=0.02)

    # Move forward diffusion tensors to device
    forward_diffusion._alphas = forward_diffusion._alphas.to(device)
    forward_diffusion._sigmas = forward_diffusion._sigmas.to(device)

    # Test different timesteps
    timesteps = [0, 250, 500, 750, 999]

    # 1. Test with single sample for visualization
    x0_single = batch[0:1].to(device)  # Take first sample [1, seq_len]
    print(f"1. Single sample shape: {x0_single.shape}")

    # Create figure for visualization
    plt.figure(figsize=(15, 5))

    for i, t in enumerate(timesteps):
        # Sample noise
        eps = torch.randn_like(x0_single)
        t_tensor = torch.tensor([t], device=device)

        # Apply forward diffusion
        xt = forward_diffusion.sample(x0_single, t_tensor, eps)

        print(f"\nt_tensor.shape: {t_tensor.shape}")
        print(f"eps.shape: {eps.shape}")
        print(f"xt.shape: {xt.shape}")

        # Plot results
        plt.subplot(1, len(timesteps), i + 1)
        plt.plot(xt[0, 0].cpu().detach().numpy(), linewidth=1, color="blue", alpha=0.8)
        plt.title(f"t={t}")

    plt.tight_layout()
    plt.savefig("ddpm_timesteps.png")
    plt.close()

    # 2. Test with full batch and different timesteps
    print("\n2. Testing with full batch...")
    batch = batch.to(device)

    # Assign different timesteps to each batch element
    batch_size = batch.shape[0]
    # TODO: use a time sampler instead of linspace
    varied_timesteps = torch.linspace(0, 999, batch_size).long().to(device)
    print(f"Varied timesteps shape: {varied_timesteps.shape}")

    # Sample noise
    eps = torch.randn_like(batch)

    # Apply forward diffusion
    xt_batch = forward_diffusion.sample(batch, varied_timesteps, eps)
    print(f"Output batch shape: {xt_batch.shape}")

    # 3. Validate noise levels
    print("\n3. Validating noise levels:")
    t_start = torch.tensor([0], device=device)
    t_mid = torch.tensor([500], device=device)
    t_end = torch.tensor([999], device=device)

    # Use same noise for fair comparison
    same_noise = torch.randn_like(x0_single)

    # Get samples at different timesteps
    x_start = forward_diffusion.sample(x0_single, t_start, same_noise)
    x_mid = forward_diffusion.sample(x0_single, t_mid, same_noise)
    x_end = forward_diffusion.sample(x0_single, t_end, same_noise)

    # Calculate signal-to-noise ratio
    # For 3D tensors, calculate variance along the sequence dimension
    if len(x_start.shape) == 3:
        # Calculate variance along the sequence dimension (dim=2)
        snr_start = (
            x_start.var(dim=2).mean() / (x_start - x0_single).var(dim=2).mean()
        ).item()
        snr_mid = (
            x_mid.var(dim=2).mean() / (x_mid - x0_single).var(dim=2).mean()
        ).item()
        snr_end = (
            x_end.var(dim=2).mean() / (x_end - x0_single).var(dim=2).mean()
        ).item()
    else:
        # Original calculation for 2D tensors
        snr_start = (x_start.var() / (x_start - x0_single).var()).item()
        snr_mid = (x_mid.var() / (x_mid - x0_single).var()).item()
        snr_end = (x_end.var() / (x_end - x0_single).var()).item()

    print(f"SNR at t={forward_diffusion.tmin}: {snr_start:.4f}")
    print(f"SNR at t=500: {snr_mid:.4f}")
    print(f"SNR at t=999: {snr_end:.4f}")


if __name__ == "__main__":
    main()
