#!/usr/bin/env python
# coding: utf-8

"""Denoising Diffusion Probabilistic Models (DDPM) Implementation.

This module implements the DDPM framework as described in the paper:
'Denoising Diffusion Probabilistic Models' (Ho et al., 2020)
https://arxiv.org/abs/2006.11239

The implementation includes both cosine and linear noise schedules, with the cosine
schedule following the improvements from 'Improved Denoising Diffusion Probabilistic Models'
(Nichol & Dhariwal, 2021) https://arxiv.org/abs/2102.09672

Typical usage:
    ddpm = DDPM(diffusion_steps=1000, schedule_type='cosine')
    noisy_sample = ddpm.sample(x0, timestep, noise)
"""

from typing import Optional, Union, Tuple

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Global device variable for consistent GPU/CPU handling across the module
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPM:
    """

    Implements the forward diffusion process that gradually adds noise to data.

    This class implements the DDPM framework where at each timestep t, we add a controlled
    amount of Gaussian noise according to:
        q(xt|x0) = N(α(t) * x0, σ(t)^2 * I)

    The forward process transitions from clean data x0 to noisy data xt via:
        xt = α(t) * x0 + σ(t) * ε, where ε ~ N(0, I)

    Key Features:
        - Supports both cosine and linear beta schedules
        - Implements the improved cosine schedule from Nichol & Dhariwal, 2021
        - Handles proper indexing for α, α_bar, and σ values
        - GPU/CPU compatible through the device system

    Attributes:
        _diffusion_steps (int): Total number of diffusion steps T
        _beta_start (float): Starting value for β schedule
        _beta_end (float): Ending value for β schedule
        _schedule_type (str): Type of β schedule ('cosine' or 'linear')
        _betas_t (torch.Tensor): β values for each timestep
        _alphas_t (torch.Tensor): α values (1 - β)
        _alphas_bar_t (torch.Tensor): Cumulative product of α values
        _sigmas_t (torch.Tensor): Standard deviation of noise at each step
    """

    def __init__(
        self,
        diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        """
        Initialize the diffusion process with specified noise schedule parameters.

        Args:
            diffusion_steps (int, optional): Total number of diffusion timesteps T. Defaults to 1000.
            beta_start (float, optional): Starting value for β schedule. Defaults to 0.0001.
            beta_end (float, optional): Final value for β schedule. Defaults to 0.02.
            schedule_type (str, optional): Type of β schedule to use ('cosine' or 'linear'). Defaults to 'cosine'.

        Raises:
            ValueError: If schedule_type is not 'cosine' or 'linear'.
        """

        # Initialize parameters
        self._diffusion_steps = diffusion_steps
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._schedule_type = schedule_type

        # Select beta schedule
        if schedule_type == "linear":
            betas_np = self._linear_beta_schedule(
                self._diffusion_steps, self._beta_start, self._beta_end
            )
        elif schedule_type == "cosine":
            betas_np = self._cosine_beta_schedule(self._diffusion_steps)
        else:
            raise ValueError(
                f"Unknown schedule_type '{schedule_type}'. Use 'cosine' or 'linear'."
            )

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

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
        """Generate a cosine beta schedule as proposed in 'Improved DDPM' (arXive:2102.09672, 2021).

        This schedule reduces training time and improves sample quality by using a cosine
        function to control the noise schedule. The offset parameter s prevents
        the schedule from getting too small near t=0.

        Args:
            timesteps (int): Total number of diffusion timesteps T.
            s (float, optional): Offset parameter to prevent β from being too small near t=0.
                               Defaults to 0.008 as per the paper.

        Returns:
            np.ndarray: Array of β values with shape (timesteps,), clipped to [0, 0.999].
        """
        # Add 1 to timesteps to account for t=0
        steps = timesteps + 1

        # Generate linearly spaced values from 0 to steps
        x = np.linspace(0, steps, steps)

        # Compute alphas_cumprod using cosine function
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Compute betas from alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clip betas to [0, 0.999]
        return np.clip(betas, 0, 0.999)

    @staticmethod
    def _linear_beta_schedule(
        timesteps: int, beta_start: float, beta_end: float
    ) -> np.ndarray:
        """Generate a linear beta schedule from beta_start to beta_end.

        This is the original schedule proposed in the DDPM paper (Ho et al., 2020).
        It provides a simple linear interpolation between the start and end noise levels.

        Args:
            timesteps (int): Total number of diffusion timesteps T.
            beta_start (float): Initial β value for the schedule.
            beta_end (float): Final β value for the schedule.

        Returns:
            np.ndarray: Array of β values with shape (timesteps,).
        """
        return np.linspace(beta_start, beta_end, timesteps)

    def _get_alphas_bar(self, betas: np.ndarray) -> np.ndarray:
        """
        Computes cumulative alpha values following the DDPM formula.

        Key indexing convention:
        - We prepend a 1.0 at the start of alphas_bar so that alphas_bar[0] = 1.0 (t=0, no noise).
        - This means for timestep t, you access alphas_bar[t].
        - This matches the DDPM paper and avoids off-by-one errors for the t=0 case.
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

        Indexing scheme:
        - For timestep t, alpha is at index t-1 (since Python is 0-indexed and alphas_t has length diffusion_steps).
        - This matches the DDPM paper's convention: alpha_1, ..., alpha_T.
        - t should be in [1, diffusion_steps].
        """

        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)

        # For alpha, use t-1 (0-indexed)
        idx = (t - 1).long()

        # Assert that idx is within bounds
        assert torch.all(
            (idx >= 0) & (idx < self._alphas_t.shape[0])
        ), f"Alpha index out of bounds: idx={idx}, shape={self._alphas_t.shape}"

        # Ensure idx is on the same device as _alphas
        if idx.device != self._alphas_t.device:
            idx = idx.to(self._alphas_t.device)

        # Return values on the correct device
        return self._alphas_t[idx]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retrieves sigma(t) for the given time indices.
        This is the standard deviation of the forward process noise.

        Indexing scheme:
        - For timestep t, sigma is at index t (because _sigmas_t has a prepended 1.0 for t=0).
        - This matches the convention for alphas_bar and ensures t=0 is handled correctly.
        - t should be in [1, diffusion_steps].
        """

        # Ensure t is in the valid range
        t = torch.clamp(t, min=1, max=self._diffusion_steps)

        # For sigma, use t directly (since _sigmas_t has a prepended 1.0)
        idx = t.long()

        # Assert that idx is within bounds
        assert torch.all(
            (idx >= 1) & (idx < self._sigmas_t.shape[0])
        ), f"Sigma index out of bounds: idx={idx}, shape={self._sigmas_t.shape}"

        # Ensure idx is on the same device as _sigmas
        if idx.device != self._sigmas_t.device:
            idx = idx.to(self._sigmas_t.device)

        # Return values on the correct device
        return self._sigmas_t[idx]

    def sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
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

    def to(self, device: Union[str, torch.device]) -> "DDPM":
        """Move all internal tensors to the specified device.

        This method mimics the behavior of nn.Module.to() for compatibility with
        PyTorch Lightning and general PyTorch operations. It ensures all internal
        tensors (betas, alphas, etc.) are on the same device.

        Args:
            device (Union[str, torch.device]): Target device to move tensors to.
                Can be either a string (e.g., 'cuda:0', 'cpu') or torch.device.

        Returns:
            DDPM: Returns self for method chaining.
        """
        # Move all internal tensors to the target device
        self._betas_t = self._betas_t.to(device)
        self._alphas_t = self._alphas_t.to(device)
        self._alphas_bar_t = self._alphas_bar_t.to(device)
        self._sigmas_t = self._sigmas_t.to(device)
        return self

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
            int: Maximum timestep value (diffusion_steps).
        """
        return self._diffusion_steps

    @property
    def betas(self) -> torch.Tensor:
        """β values for the noise schedule.

        These control how much noise is added at each timestep.

        Returns:
            torch.Tensor: β values of shape (diffusion_steps,).
        """
        return self._betas_t

    @property
    def alphas(self) -> torch.Tensor:
        """α values for the noise schedule, where α_t = 1 - β_t.

        These represent how much of the original signal is preserved at each timestep.

        Returns:
            torch.Tensor: α values of shape (diffusion_steps,).
        """
        return self._alphas_t

    @property
    def alphas_bar(self) -> torch.Tensor:
        """Cumulative product of α values (ᾱ_t = Π_{s=1}^t α_s).

        These represent the total signal preservation up to timestep t.
        Note: Index 0 is 1.0 (no noise) for convenient indexing.

        Returns:
            torch.Tensor: ᾱ values of shape (diffusion_steps + 1,).
        """
        return self._alphas_bar_t

    @property
    def sigmas(self) -> torch.Tensor:
        """σ values for the noise schedule, where σ_t = √(1 - ᾱ_t).

        These represent the standard deviation of the noise at each timestep.
        Note: Index 0 is 0.0 (no noise) for convenient indexing.

        Returns:
            torch.Tensor: σ values of shape (diffusion_steps + 1,).
        """
        return self._sigmas_t


def load_config(config_path: str) -> dict:
    """Load DDPM configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary containing:
            - diffusion: DDPM parameters (timesteps, beta schedule)
            - time_sampler: Timestep sampling parameters
            - unet: UNet1D architecture settings
            - data: Data-specific parameters
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run tests for the DDPM implementation.

    This function performs several tests:
    1. Tests the forward diffusion process with different timesteps
    2. Tests batch processing with varied timesteps
    3. Validates noise levels through signal-to-noise ratio analysis

    Results are saved as plots and metrics are printed to stdout.
    """
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
