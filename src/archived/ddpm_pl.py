"""
PyTorch Lightning implementation of DDPM using NetworkBase.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F

from .mlp import MLP
from .network_base import NetworkBase


class DDPM(NetworkBase):
    """Denoising Diffusion Probabilistic Model with PyTorch Lightning integration."""

    def __init__(self, config: Dict):
        """Initialize the DDPM model.

        Args:
            config: Configuration dictionary containing model and training parameters.
        """
        # Initialize base class with config
        super().__init__(config)

        # Store config
        self.config = config

        # Initialize MLP model
        self.model = MLP(
            embedding_dim=config["unet"]["embedding_dim"],
            seq_length=config["data"]["seq_length"],
            channels=1,  # For SNP data
            dim_mults=config["unet"]["dim_mults"],
            with_time_emb=config["unet"]["with_time_emb"],
            with_pos_emb=config["unet"]["with_pos_emb"],
        )

        # Initialize noise schedule
        self.num_timesteps = config["diffusion"].get("timesteps", 1000)
        self.beta_start = config["diffusion"].get("beta_start", 0.0001)
        self.beta_end = config["diffusion"].get("beta_end", 0.02)
        self.schedule_type = config["diffusion"].get("schedule_type", "cosine")

        # Pre-compute values for the forward and reverse processes
        self._init_noise_schedule()

    def _init_noise_schedule(self):
        """Initialize the noise schedule parameters."""
        # Linear schedule for beta
        if self.schedule_type == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        # Cosine schedule as in Improved DDPM
        elif self.schedule_type == "cosine":
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5)
                ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Register as buffer to move with model
        self.register_buffer("betas", betas)

        # Pre-compute values for q(x_t | x_0)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register all buffers
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-compute values for q(x_{t-1} | x_t, x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # Clamp to avoid numerical issues
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas_cumprod)
            / (1.0 - alphas_cumprod),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
            t: Timestep tensor of shape [batch_size]

        Returns:
            torch.Tensor: Predicted noise of same shape as x
        """
        return self.model(x, t)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch of data.

        Args:
            batch: Input batch of shape [batch_size, channels, seq_len]

        Returns:
            torch.Tensor: Loss value
        """
        # Sample random timesteps for each item in the batch
        batch_size = batch.size(0)
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=batch.device
        ).long()

        # Sample noise and add to the input
        noise = torch.randn_like(batch)
        x_t = self.q_sample(batch, t, noise=noise)

        # Predict the noise
        predicted_noise = self(x_t, t)

        # Compute MSE loss
        return F.mse_loss(predicted_noise, noise)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0).

        Args:
            x0: Initial data point [batch_size, channels, seq_len]
            t: Timestep for each sample in the batch [batch_size]
            noise: Optional pre-sampled noise

        Returns:
            torch.Tensor: Noisy sample x_t
        """
        # Ensure t is within valid range
        t = t.clamp(0, self.num_timesteps - 1)

        # Get the corresponding alpha_cumprod for each t
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1
        )

        # Sample from q(x_t | x_0)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t).

        Args:
            x: The noisy sample at timestep t [batch_size, channels, seq_len]
            t: The current timestep

        Returns:
            torch.Tensor: Sample x_{t-1}
        """
        batch_size = x.size(0)
        device = x.device

        # Create tensor of timesteps
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict the noise
        predicted_noise = self(x, t_tensor)

        # Calculate the mean and variance of the reverse process
        model_mean, model_variance = self.q_posterior_mean_variance(
            x, t_tensor, predicted_noise
        )

        # Sample from the posterior
        if t > 0:
            noise = torch.randn_like(x)
            sample = model_mean + torch.sqrt(model_variance) * noise
        else:
            sample = model_mean

        return sample

    def q_posterior_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and variance of the reverse process posterior q(x_{t-1} | x_t, x_0)."""
        # Get the predicted x_0
        x0_pred = self._predict_x0_from_noise(x, t, noise)

        # Compute the mean of the posterior
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x0_pred
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x
        )

        # Get the variance
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1)

        return posterior_mean, posterior_variance

    def _predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and the predicted noise."""
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    @torch.no_grad()
    def sample(
        self, num_samples: int, seq_length: int, device: Union[str, torch.device] = None
    ) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate
            seq_length: Length of each sequence
            device: Device to generate samples on

        Returns:
            torch.Tensor: Generated samples of shape [num_samples, 1, seq_length]
        """
        if device is None:
            device = next(self.parameters()).device

        # Start with random noise
        x = torch.randn((num_samples, 1, seq_length), device=device)

        # Sample from p(x_{t-1} | x_t) for t = T, ..., 1
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)

        return x

    @torch.no_grad()
    def denoise_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of data.

        Args:
            batch: Input batch of shape [batch_size, channels, seq_len]

        Returns:
            torch.Tensor: Denoised batch
        """
        # Add noise to the batch
        t = torch.full((batch.size(0),), self.num_timesteps - 1, device=batch.device)
        noise = torch.randn_like(batch)
        noisy_batch = self.q_sample(batch, t, noise)

        # Denoise the batch
        return self.sample_from_noisy(noisy_batch)

    @torch.no_grad()
    def sample_from_noisy(self, x_t: torch.Tensor) -> torch.Tensor:
        """Sample from the model starting from a noisy input.

        Args:
            x_t: Noisy input of shape [batch_size, channels, seq_len]

        Returns:
            torch.Tensor: Denoised sample
        """
        # Sample from p(x_{t-1} | x_t) for t = T, ..., 1
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t)

        return x_t

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate

        Returns:
            torch.Tensor: Generated samples
        """
        # Use the sequence length from config
        seq_length = self.config["data"]["seq_length"]
        return self.sample(num_samples, seq_length)
