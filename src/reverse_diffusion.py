#!/usr/bin/env python
# coding: utf-8

"""
Reverse Diffusion Process for Denoising Diffusion Probabilistic Models (DDPM).

This module implements the REVERSE (denoising/sampling) process of the DDPM framework as described in:
    'Denoising Diffusion Probabilistic Models' (Ho et al., 2020)
    https://arxiv.org/abs/2006.11239

Typical usage:
    reverse_diff = ReverseDiffusion(forward_diffusion, noise_predictor, data_shape)
    samples = reverse_diff.generate_samples(num_samples=10)

Note: This class only implements the reverse (denoising) process p(x_{t-1}|x_t), NOT the full DDPM model.
"""

import torch

from .utils import bcast_right, prepare_batch_shape, set_seed, tensor_to_device


class ReverseDiffusion:
    """
    Implements the reverse diffusion process (denoising/sampling) for DDPM.

    Args:
        forward_diffusion: Instance of ForwardDiffusion (provides noise schedule, etc.)
        noise_predictor: The model (e.g., UNet1D or MLP) for noise prediction.
        data_shape: Shape of the data (channels, seq_length)
    """

    def __init__(
        self,
        forward_diffusion,
        noise_predictor,
        data_shape,
        denoise_step=10,
        discretize=False,
    ):
        self.forward_diffusion = forward_diffusion
        self.noise_predictor = noise_predictor
        self.data_shape = data_shape
        self.denoise_step = denoise_step
        self.discretize = discretize

    # FIXME: Extract function allows to extract appropriate t index for a batch of indices.
    def extract(a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    # FIXME: Reverse diffusion: https://huggingface.co/blog/annotated-diffusion
    def p_sample(self, x_t, t, t_index):
        betas_t = extract(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # FIXME: Algorithm 2 (including returning all images)
    def p_sample_loop(model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, timesteps)),
            desc="sampling loop time step",
            total=timesteps,
        ):
            img = p_sample(
                model, img, torch.full((b,), i, device=device, dtype=torch.long), i
            )
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(model, image_size, batch_size=16, channels=3):
        return p_sample_loop(
            model, shape=(batch_size, channels, image_size, image_size)
        )

    def reverse_diffusion_step(self, x_t, t):
        """
        Single reverse diffusion step to estimate x_{t-1} given x_t and t.
        Implements Algorithm 2 from the DDPM paper (Ho et al., 2020):
            p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), β_t I)
            μ_θ(x_t, t) = (1/sqrt(α_t)) * [x_t - (β_t / sqrt(1 - ᾱ_t)) * ε_θ(x_t, t)]
        Args:
            x_t: Noisy sample at timestep t, shape [B, C, seq_len] (must be on correct device)
            t: Current timestep (tensor of shape [B] or scalar int, 1-based)
        Returns:
            x_{t-1}: Sample at timestep t-1, same shape as x_t
        """
        # Ensure tensors are on the correct device
        device = x_t.device
        t = tensor_to_device(t, device)

        # Ensure x_t has the correct shape [B, C, L]
        x_t = prepare_batch_shape(x_t)

        # Get diffusion parameters for timestep t
        beta_t = tensor_to_device(self.forward_diffusion.betas[t - 1], device)  # β_t
        alpha_t = tensor_to_device(self.forward_diffusion.alpha(t), device)  # α_t
        alpha_bar_t = tensor_to_device(
            self.forward_diffusion.alphas_bar[t], device
        )  # ᾱ_t
        alpha_bar_prev = tensor_to_device(
            self.forward_diffusion.alphas_bar[t - 1], device
        )  # ᾱ_{t-1}

        # Broadcast parameters to match x_t's dimensions
        ndim = x_t.ndim
        beta_t = bcast_right(beta_t, ndim)
        alpha_t = bcast_right(alpha_t, ndim)
        alpha_bar_t = bcast_right(alpha_bar_t, ndim)
        alpha_bar_prev = bcast_right(alpha_bar_prev, ndim)

        # Numerical stability constant
        eps = 1e-7

        # Predict noise using the noise prediction model (ε_θ(x_t, t))
        eps_theta = self.noise_predictor(x_t, t)

        # FIXME: Add t=1, t>1 cases

        # Compute mean for p(x_{t-1}|x_t) as in Eq. 7
        inv_sqrt_alpha_t = torch.rsqrt(alpha_t + eps)  # 1/sqrt(α_t)
        mean = inv_sqrt_alpha_t * (
            x_t
            - (beta_t / torch.sqrt(1.0 - alpha_bar_t + eps)) * eps_theta  # μ_θ(x_t, t)
        )
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        # Variance is β_t (Eq. 7)
        std = torch.sqrt(torch.clamp(beta_t, min=1e-6))  # sqrt(β_t)

        # Sample from N(mean, std^2 * I)
        z = torch.randn_like(x_t, device=device)  # Sample from N(0, I)
        x_prev = mean + std * z  # x_{t-1} ~ N(mean, β_t I)

        return x_prev

    def _reverse_diffusion_process(self, x, denoise_step=10, discretize=False):
        """
        Internal method to run the reverse diffusion process on a given tensor.

        Args:
            x: Input tensor to denoise (already on the correct device)
            denoise_step: Number of timesteps to skip in reverse diffusion
            discretize: If True, discretize output to SNP values

        Returns:
            Denoised tensor of the same shape as x
        """
        # Get diffusion process limits
        tmax = self.forward_diffusion.tmax
        tmin = self.forward_diffusion.tmin

        # Iterate over timesteps in reverse (Algorithm 2 from Ho et al., 2020)
        for t in reversed(range(tmin, tmax + 1, denoise_step)):
            t_tensor = tensor_to_device(
                torch.full((x.size(0),), t, dtype=torch.long), x.device
            )
            x = self.reverse_diffusion_step(x, t_tensor)

        # Post-processing for SNP data
        x = torch.clamp(x, 0, 0.5)
        if discretize:
            x = torch.round(x * 4) / 4
            x = torch.clamp(x, 0, 0.5)

        return x

    def generate_samples(
        self, num_samples=10, denoise_step=None, discretize=None, seed=42, device=None
    ):
        """
        Generate new samples from random noise using the reverse diffusion process.

        This method implements Algorithm 2 from Ho et al., 2020:
            For t = T,...,1:
                x_{t-1} ~ p_θ(x_{t-1}|x_t)
            where p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), β_t I)

        Args:
            num_samples: Number of samples to generate
            denoise_step: Number of timesteps to skip in reverse diffusion
            discretize: If True, discretize output to SNP values
            seed: Random seed for reproducibility
            device: torch.device

        Returns:
            Generated samples of shape [num_samples, C, seq_len]
        """
        with torch.no_grad():
            # Set seed for reproducibility
            set_seed(seed)

            # Determine device to use
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create batch shape
            batch_shape = (num_samples,) + self.data_shape

            # Start from pure noise
            x = tensor_to_device(torch.randn(batch_shape), device)

            # Use instance defaults if not provided
            if denoise_step is None:
                denoise_step = self.denoise_step
            if discretize is None:
                discretize = self.discretize
            # Run the reverse diffusion process
            return self._reverse_diffusion_process(x, denoise_step, discretize)

    def denoise_sample(
        self, batch, denoise_step=None, discretize=None, seed=42, device=None
    ):
        """
        Denoise an existing batch using the reverse diffusion process.

        This method takes an existing batch (which could be noisy data or even
        clean data that you want to process through the model) and applies the
        reverse diffusion process to it.

        Args:
            batch: Input batch to denoise, shape [B, C, seq_len]
            denoise_step: Number of timesteps to skip in reverse diffusion
            discretize: If True, discretize output to SNP values
            seed: Random seed for reproducibility
            device: torch.device

        Returns:
            Denoised output of shape [B, C, seq_len]
        """
        with torch.no_grad():
            # Set seed for reproducibility
            set_seed(seed)

            # Determine device to use
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Ensure batch has the correct shape and is on the right device
            batch = tensor_to_device(prepare_batch_shape(batch), device)

            # Start from pure noise with the same shape as the input batch
            x = torch.randn_like(batch)

            # Use instance defaults if not provided
            if denoise_step is None:
                denoise_step = self.denoise_step
            if discretize is None:
                discretize = self.discretize
            # Run the reverse diffusion process
            return self._reverse_diffusion_process(x, denoise_step, discretize)
