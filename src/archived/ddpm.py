"""
Diffusion Process implementation for Denoising Diffusion Probabilistic Models (DDPM).

This module provides a cleaner and more modular implementation of the diffusion process,
separating the noise scheduling, forward process, and reverse process into distinct
components while maintaining compatibility with the existing codebase.
"""

import math
import os
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract elements from a tensor at specified indices.

    Args:
        a: Tensor to extract values from.
        t: Indices to extract.
        x_shape: Shape of the target tensor for broadcasting.

    Returns:
        Tensor with values from `a` at indices `t` with shape matching `x_shape`.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class NoiseSchedule:
    """Manages the noise schedule for the diffusion process.

    Handles the computation of all noise schedule-related parameters (α, ᾱ, σ)
    and provides methods for sampling from the forward and reverse processes.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        """Initialize the noise schedule.

        Args:
            num_timesteps: Total number of diffusion timesteps (T).
            beta_start: Starting value of beta for noise schedule.
            beta_end: Final value of beta for noise schedule.
            schedule_type: Type of noise schedule ('cosine' or 'linear').
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type.lower()

        # Initialize the noise schedule
        self._init_noise_schedule()

    def _init_noise_schedule(self) -> None:
        """Initialize the noise schedule parameters."""
        if self.schedule_type == "linear":
            betas = self._linear_beta_schedule()
        elif self.schedule_type == "cosine":
            betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Convert to tensors
        self.betas = torch.tensor(betas, dtype=torch.float32)

        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _linear_beta_schedule(self) -> np.ndarray:
        """Linear beta schedule."""
        return np.linspace(
            self.beta_start, self.beta_end, self.num_timesteps, dtype=np.float32
        )

    def _cosine_beta_schedule(self, s: float = 0.008) -> np.ndarray:
        """Cosine beta schedule as proposed in Improved DDPM."""
        steps = self.num_timesteps + 1
        x = np.linspace(0, self.num_timesteps, steps, dtype=np.float32)
        alphas_cumprod = (
            np.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0, 0.999)

    def to(self, device: Union[str, torch.device]) -> "NoiseSchedule":
        """Move all tensors to the specified device."""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


class DDPM:
    """Denoising Diffusion Probabilistic Model (DDPM).

    This class implements both the forward process (q(x_t | x_0)) that gradually adds noise to data
    and the reverse process (p(x_{t-1} | x_t)) that learns to denoise data.
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        model: nn.Module,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """Initialize the diffusion process.

        Args:
            noise_schedule: Configured noise schedule.
            model: Model that predicts the noise (typically a UNet).
            device: Device to run computations on.
        """
        self.noise_schedule = noise_schedule.to(device)
        self.model = model.to(device)
        self.device = device

        # Pre-compute values for the reverse process
        self._init_reverse_process()

    def _init_reverse_process(self) -> None:
        """Initialize parameters needed for the reverse process."""
        # For q(x_{t-1} | x_t, x_0)
        alphas_cumprod_prev = F.pad(
            self.noise_schedule.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # Posterior variance
        self.posterior_variance = (
            self.noise_schedule.betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - self.noise_schedule.alphas_cumprod)
        )

        # Clamp to avoid numerical issues
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)

        # Coefficients for the posterior mean
        self.posterior_mean_coef1 = (
            self.noise_schedule.betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - self.noise_schedule.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(self.noise_schedule.alphas)
            / (1.0 - self.noise_schedule.alphas_cumprod)
        )

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0).

        Args:
            x0: Initial data point (batch_size, *shape).
            t: Timestep for each sample in the batch (batch_size,).
            noise: Optional pre-sampled noise.

        Returns:
            Noisy sample x_t.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = extract(
            self.noise_schedule.sqrt_alphas_cumprod, t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t).

        Args:
            xt: The noisy sample at timestep t.
            t: The current timestep.

        Returns:
            Sample x_{t-1}.
        """
        # Predict the noise component
        pred_noise = self.model(xt, t)

        # Calculate the posterior mean and variance
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, xt.shape)
            * self._predict_x0_from_noise(xt, t, pred_noise)
            + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )

        # Sample from the posterior
        noise = torch.randn_like(xt) if any(t > 0) else torch.zeros_like(xt)
        posterior_variance_t = extract(self.posterior_variance, t, xt.shape)

        return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    def _predict_x0_from_noise(
        self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip_alphas_cumprod_t = extract(
            1.0 / torch.sqrt(self.noise_schedule.alphas_cumprod), t, xt.shape
        )
        sqrt_recipm1_alphas_cumprod_t = extract(
            torch.sqrt(1.0 / self.noise_schedule.alphas_cumprod - 1.0), t, xt.shape
        )

        return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * noise

    def training_loss(
        self, x0: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the training loss (MSE between predicted and true noise).

        Args:
            x0: The original data sample.
            t: Timesteps. If None, sample uniformly.

        Returns:
            Loss value.
        """
        if t is None:
            t = torch.randint(
                0, self.noise_schedule.num_timesteps, (x0.shape[0],), device=self.device
            )

        # Sample noise and create noisy sample
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        # Predict the noise
        pred_noise = self.model(xt, t)

        # MSE loss
        return F.mse_loss(pred_noise, noise)

    def sample(
        self,
        num_samples: int,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate.
            shape: Shape of each sample (excluding batch dimension).
            device: Device to generate samples on.

        Returns:
            Generated samples.
        """
        device = device or self.device
        x = torch.randn((num_samples, *shape), device=device)

        for t in reversed(range(self.noise_schedule.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)

        return x


# Factory function for backward compatibility
def create_ddpm(config: Dict[str, Any], model: nn.Module) -> DDPM:
    """Create a DDPM instance from a config dictionary."""
    noise_schedule = NoiseSchedule(
        num_timesteps=config["diffusion_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_type=config.get("schedule_type", "cosine"),
    )

    return DDPM(
        noise_schedule=noise_schedule,
        model=model,
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Demonstrate the DDPM class with MLP as the noise prediction model."""
    print("=== Testing DDPM with MLP ===\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load config from file
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
    config = load_config(config_path)

    # Initialize dataset
    print("Loading dataset...")
    from .dataset import SNPDataset

    dataset = SNPDataset(
        input_path=config["input_path"], seq_length=config["data"]["seq_length"]
    )

    # Create dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )

    # Get a batch of real data
    x0 = next(iter(dataloader))
    if isinstance(x0, (list, tuple)):
        x0 = x0[0]  # Unpack if dataloader returns (data, target)

    # Ensure data is in correct shape [batch, channels, seq_len]
    if x0.dim() == 2:
        x0 = x0.unsqueeze(1)  # Add channel dimension if missing

    print(f"\nLoaded data shape: {x0.shape}")
    print(f"Data range: [{x0.min().item():.4f}, {x0.max().item():.4f}]")

    # Update config with actual sequence length
    config["seq_length"] = x0.shape[2]
    config["channels"] = x0.shape[1]

    # Import and initialize MLP model
    from .mlp import MLP

    model = MLP(
        embedding_dim=config["unet"]["embedding_dim"],
        seq_length=config["seq_length"],
        channels=config["channels"],
        dim_mults=config["unet"]["dim_mults"],
        with_time_emb=config["unet"]["with_time_emb"],
        with_pos_emb=config["unet"]["with_pos_emb"],
    )

    # Initialize DDPM with diffusion parameters from config
    ddpm_config = {
        "diffusion_steps": config["diffusion"]["diffusion_steps"],
        "beta_start": config["diffusion"]["beta_start"],
        "beta_end": config["diffusion"]["beta_end"],
        "schedule_type": config["diffusion"]["schedule_type"],
        "seq_length": config["seq_length"],
        "channels": config["channels"],
        "embedding_dim": config["unet"]["embedding_dim"],
    }

    ddpm = create_ddpm(ddpm_config, model)

    # Test forward process (q_sample)
    print("\n1. Testing forward process...")
    batch_size = min(4, x0.shape[0])  # Use smaller batch size for testing
    t = torch.randint(0, ddpm_config["diffusion_steps"], (batch_size,))
    x0_batch = x0[:batch_size]
    xt = ddpm.q_sample(x0_batch, t)
    print(f"  - Input shape: {x0_batch.shape}")
    print(f"  - Output shape: {xt.shape}")
    print(
        f"  - Input mean: {x0_batch.mean().item():.4f}, Output mean: {xt.mean().item():.4f}"
    )

    # Test reverse process (p_sample)
    print("\n2. Testing reverse process...")
    xt_prev = ddpm.p_sample(xt, t)
    print(f"  - Output shape: {xt_prev.shape}")
    print(f"  - Output mean: {xt_prev.mean().item():.4f}")

    # Test training loss
    print("\n3. Testing training loss...")
    loss = ddpm.training_loss(x0_batch)
    print(f"  - Training loss: {loss.item():.6f}")

    # Test sampling
    print("\n4. Testing sampling...")
    samples = ddpm.sample(
        num_samples=2, shape=(config["channels"], config["seq_length"])
    )
    print(f"  - Generated samples shape: {samples.shape}")
    print(
        f"  - Samples mean: {samples.mean().item():.4f}, std: {samples.std().item():.4f}"
    )

    print("\n=== DDPM Test completed successfully ===")


if __name__ == "__main__":
    main()
