#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Global device variable for consistent handling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # Calculate beta schedule using cosine schedule
        betas_np = self._cosine_beta_schedule(self._diffusion_steps)

        # Calculate alphas and cumulative alphas
        alphas_np = 1.0 - betas_np
        alphas_bar_np = self._get_alphas_bar(betas_np)

        # Calculate sigmas
        sigmas_np = np.sqrt(1.0 - alphas_bar_np)

        # Convert to tensors and register as buffers
        self.register_buffer("_betas_t", torch.tensor(betas_np, dtype=torch.float32))
        self.register_buffer("_alphas_t", torch.tensor(alphas_np, dtype=torch.float32))
        self.register_buffer(
            "_alphas_bar_t", torch.tensor(alphas_bar_np, dtype=torch.float32)
        )
        self.register_buffer("_sigmas_t", torch.tensor(sigmas_np, dtype=torch.float32))

    # Removed redundant _register_buffers method as we now register buffers in __init__

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

    def _get_alphas_bar(self, betas: np.ndarray) -> np.ndarray:
        """Computes cumulative alpha values following the DDPM formula.

        Args:
            betas: Beta values for the noise schedule

        Returns:
            np.ndarray: Cumulative product of alphas with 1.0 prepended
        """
        # Calculate alphas = 1 - betas
        alphas = 1.0 - betas

        # Calculate cumulative product of alphas
        alphas_bar = np.cumprod(alphas)

        # Append 1 at the beginning for convenient indexing (t=0 has no noise)
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

        # Ensure idx is on the same device as alphas
        if idx.device != self._alphas_t.device:
            idx = idx.to(self._alphas_t.device)

        # Return values on the correct device
        return self._alphas_t[idx]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves sigma(t) for the given time indices.
        This is the standard deviation of the forward process noise.

        Args:
            t (torch.Tensor): Timesteps (batch_size,).

        Returns:
            torch.Tensor: Sigma values corresponding to timesteps.
        """
        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)
        # Convert to indices (0-indexed)
        idx = (t - 1).long()

        # Ensure idx is on the same device as sigmas
        if idx.device != self._sigmas_t.device:
            idx = idx.to(self._sigmas_t.device)

        # Return pre-computed sigma values on the correct device
        return self._sigmas_t[idx]

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
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)

        # Reshape alpha_t and sigma_t according to the input shape
        if len(x0.shape) == 3:  # [batch_size, channels, seq_len]
            alpha_t = alpha_t.view(-1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1)
        else:  # [batch_size, seq_len]
            alpha_t = alpha_t.view(-1, 1)
            sigma_t = sigma_t.view(-1, 1)

        # Both alpha_t and sigma_t are already on the correct device from alpha() and sigma()
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

        Args:
            device: The device to move tensors to

        Returns:
            self: Returns self for method chaining
        """
        self._betas_t = self._betas_t.to(device)
        self._alphas_t = self._alphas_t.to(device)
        self._alphas_bar_t = self._alphas_bar_t.to(device)
        self._sigmas_t = self._sigmas_t.to(device)
        return self

    @property
    def tmin(self) -> int:
        """Minimum timestep value."""
        return 1

    @property
    def tmax(self) -> int:
        """Maximum timestep value."""
        return self._diffusion_steps

    @property
    def betas(self):
        """Get beta values for noise schedule."""
        return self._betas_t

    @property
    def alphas(self):
        """Get alpha values (1 - beta)."""
        return self._alphas_t

    @property
    def alphas_bar(self):
        """Get cumulative product of alphas."""
        return self._alphas_bar_t

    @property
    def sigmas(self):
        """Get sigma values (sqrt(1 - alphas_bar))."""
        return self._sigmas_t


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    try:
        print(f"Using device: {DEVICE}")

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Load config
        config = load_config("../config.yaml")
        print("\nInitializing dataset...")

        # Initialize dataset
        dataset = SNPDataset(
            input_path=config.get("input_path"),
            seq_length=config.get("data").get("seq_length"),
        )

        # Initialize dataloader
        train_loader = DataLoader(
            dataset,
            batch_size=config.get("batch_size"),
            shuffle=True,
            num_workers=config.get("num_workers"),
            pin_memory=True,
        )

        # Get batch
        batch = next(iter(train_loader))  # Shape: [B, seq_len]
        batch = batch.unsqueeze(1).to(DEVICE)  # [B, 1, seq_len]

        # ------------------ Test DDPM
        print("\nTesting DDPM...")
        forward_diffusion = DDPM(
            diffusion_steps=config.get("diffusion").get("diffusion_steps"),
            beta_start=config.get("diffusion").get("beta_start"),
            beta_end=config.get("diffusion").get("beta_end"),
        )

        # Move forward diffusion tensors to device
        forward_diffusion.to(DEVICE)

        # Test different timesteps
        timesteps = [0, 250, 500, 750, 999]

        # 1. Test with single sample for visualization
        x0_single = batch[0:1]  # Take first sample [1, seq_len]

        # Create figure for visualization
        plt.figure(figsize=(15, 5))

        for i, t in enumerate(timesteps):
            # Sample noise
            eps = torch.randn_like(x0_single).to(DEVICE)
            t_tensor = torch.tensor([t], device=DEVICE)

            # Apply forward diffusion
            xt = forward_diffusion.sample(x0_single, t_tensor, eps)

            print(f"\nt_tensor.shape: {t_tensor.shape}")
            print(f"eps.shape: {eps.shape}")
            print(f"xt.shape: {xt.shape}")

            # Plot results - move to CPU for plotting if needed
            xt_cpu = xt.cpu() if DEVICE.type == "cuda" else xt
            plt.subplot(1, len(timesteps), i + 1)
            plt.plot(
                xt_cpu[0, 0].detach().numpy(), linewidth=1, color="blue", alpha=0.8
            )
            plt.title(f"t={t}")

        plt.tight_layout()
        plt.savefig("ddpm_timesteps.png")
        plt.close()

        # 2. Test with full batch and different timesteps
        print("\n2. Testing with full batch...")

        # Assign different timesteps to each batch element
        batch_size = batch.shape[0]
        # TODO: use a time sampler instead of linspace
        varied_timesteps = torch.linspace(0, 999, batch_size).long().to(DEVICE)
        print(f"Varied timesteps shape: {varied_timesteps.shape}")

        # Sample noise
        eps = torch.randn_like(batch).to(DEVICE)

        # Apply forward diffusion
        xt_batch = forward_diffusion.sample(batch, varied_timesteps, eps)
        print(f"Output batch shape: {xt_batch.shape}")

        # 3. Validate noise levels
        print("\n3. Validating noise levels:")
        t_start = torch.tensor([0], device=DEVICE)
        t_mid = torch.tensor([500], device=DEVICE)
        t_end = torch.tensor([999], device=DEVICE)

        # Use same noise for fair comparison
        same_noise = torch.randn_like(x0_single).to(DEVICE)

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

    except Exception as e:
        print(f"Error in main(): {e}")


if __name__ == "__main__":
    main()
