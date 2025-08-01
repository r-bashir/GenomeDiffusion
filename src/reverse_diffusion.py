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

"""
Summary: Diagnosing Rising MSE in Reverse Diffusion

- This file includes experiments to diagnose why MSE(x_t_minus_1, x0) increases during reverse diffusion.
- Key findings:
    * Schedule and indexing are correct (confirmed by ε_theta = 0 test).
    * A linear model is too simple to denoise, so MSE increases with each step.
    * With ε_theta = 0 and z = 0 (see FIXME lines), the process is stationary and MSE stays constant.
- Recommendation: Use a more expressive, trained model to observe MSE decrease during denoising.

See FIXME comments for deterministic/noise-free test code.
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

    def reverse_diffusion_step_Ho(self, x_t, t, return_all=False):
        """
        Single reverse diffusion step to estimate x_{t-1} given x_t and t.
        Implements Algorithm 2 from the DDPM paper (Ho et al., 2020):

        For t > 1:
            p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2 I)
            μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
            σ_t^2 = β_t
            z ~ N(0, I)

        For t = 1:
            p_θ(x_0|x_1) = μ_θ(x_1, 1)  # No noise added (z = 0)
            μ_θ(x_1, 1) = (1/√α_1) * (x_1 - (β_1/√(1-ᾱ_1)) * ε_θ(x_1, 1))

        Args:
            x_t: Noisy sample at timestep t, shape [B, C, seq_len] (must be on correct device)
            t: Current timestep (tensor of shape [B] or scalar int, 1-based)
        Returns:
            x_{t-1}: Sample at timestep t-1, same shape as x_t
        """
        # Ensure tensors are on the correct device
        device = x_t.device

        # Convert integer timestep to tensor if needed
        if isinstance(t, int):
            t = torch.tensor([t], device=device)
        else:
            t = tensor_to_device(t, device)

        # Ensure x_t has the correct shape [B, C, L]
        # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, ε ~ N(0, I) comes from ForwardDiffusion
        x_t = prepare_batch_shape(x_t)

        # Get diffusion parameters for timestep t, either use properties with manual indexing
        # beta_t = tensor_to_device(self.forward_diffusion.betas[t - 1], device)  # β_t
        # alpha_t = tensor_to_device(self.forward_diffusion.alphas[t - 1], device)  # α_t
        # alpha_bar_t = tensor_to_device(self.forward_diffusion.alphas_bar[t], device)  # ᾱ_t

        # or use instance methods that handle indexing internally
        beta_t = tensor_to_device(self.forward_diffusion.beta(t), device)  # β_t
        alpha_t = tensor_to_device(self.forward_diffusion.alpha(t), device)  # α_t
        alpha_bar_t = tensor_to_device(
            self.forward_diffusion.alpha_bar(t), device
        )  # ᾱ_t

        # Broadcast parameters to match x_t's dimensions
        ndim = x_t.ndim
        beta_t = bcast_right(beta_t, ndim)
        alpha_t = bcast_right(alpha_t, ndim)
        alpha_bar_t = bcast_right(alpha_bar_t, ndim)

        # Numerical stability constant
        eps = 1e-8

        # Predict noise using the noise prediction model
        # ε_θ(x_t, t): Model's prediction of the noise added at timestep t
        epsilon_theta = self.noise_predictor(x_t, t)

        # === DDPM Reverse Step Equations ===
        # Compute mean for p(x_{t-1}|x_t) as in Algorithm 2
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))

        # inv_sqrt_alpha_t = 1/√α_t
        inv_sqrt_alpha_t = torch.rsqrt(alpha_t + eps)

        # coef = β_t/√(1-ᾱ_t)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t + eps)

        # scaled_pred_noise = β_t/√(1-ᾱ_t) * ε_θ(x_t, t)
        scaled_epsilon_theta = coef * epsilon_theta  # coef * ε_θ(x_t, t)

        # === Debugging: Print Important Variables per Timestep ===
        # print(
        #     f"t: {t.item()}, β_t: {beta_t.flatten()[0].item():.10f}, α_t: {alpha_t.flatten()[0].item():.10f}, ᾱ_t: {alpha_bar_t.flatten()[0].item():.10f}, 1/√α_t: {inv_sqrt_alpha_t.flatten()[0].item():.10f}, β_t/√(1-ᾱ_t): {coef.flatten()[0].item():.10f}"
        # )

        # Compute mean for p(x_{t-1}|x_t) as in DDPM Algorithm 2:
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
        # mean = inv_sqrt_alpha_t * (x_t - scaled_epsilon_theta)

        # TODO: Compute mean for p(x_{t-1}|x_t) slightly different from Algorithm 2
        # μ_θ(x_t, t) = x_t - ε_θ(x_t, t)

        # FIXME: Test ε_θ(x_t, t) = 0, i.e. μ_θ(x_t, t) = x_t
        # epsilon_theta = torch.zeros_like(x_t)

        mean = x_t - epsilon_theta
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        # abs_diff = torch.abs(epsilon_theta)
        # mean_abs_diff = abs_diff.mean().item()
        # max_abs_diff = abs_diff.max().item()
        # min_abs_diff = abs_diff.min().item()
        # print(f"t: {t.item()}, Mean |ε_theta|: {mean_abs_diff:.6f}, Max |ε_theta|: {max_abs_diff:.6f}, Min |ε_theta|: {min_abs_diff:.6f}")

        # For t=1, z=0 (no noise); for t>1, z ~ N(0,I) as per Algorithm 2
        z = torch.zeros_like(x_t, device=device)
        if (t > 1).all():  # Only add noise if all timesteps are > 1
            z = torch.randn_like(x_t, device=device)  # z ~ N(0, I)

        # FIXME: No noise added, along with epsilon_theta = 0 (DDIM)
        # z = torch.zeros_like(x_t, device=device)

        # Sample from N(mean, σ_t^2 * I)
        # x_{t-1} = μ_θ(x_t, t) + σ_t * z, where σ_t = √β_t
        sigma_t = torch.sqrt(beta_t)  # σ_t
        x_t_minus_1 = mean + sigma_t * z

        # === Debugging: Print & Return Important Variables per Timestep ===
        if return_all:

            print(
                f"t: {t.item()}, β_t: {beta_t.flatten()[0].item():.10f}, α_t: {alpha_t.flatten()[0].item():.10f}, ᾱ_t: {alpha_bar_t.flatten()[0].item():.10f}, 1/√α_t: {inv_sqrt_alpha_t.flatten()[0].item():.10f}, β_t/√(1-ᾱ_t): {coef.flatten()[0].item():.10f}, σ_t: {sigma_t.flatten()[0].item():.10f}"
            )
            return {
                "epsilon_theta": epsilon_theta,  # ε_θ(x_t, t): Model's predicted noise
                "coef": coef,  # β_t/√(1-ᾱ_t): Scaling coefficient for predicted noise
                "scaled_epsilon_theta": scaled_epsilon_theta,  # coeff * ε_θ(x_t, t): Scaled predicted noise
                "inv_sqrt_alpha_t": inv_sqrt_alpha_t,  # 1/√α_t: factor for diagnostics
                "mean": mean,  # μ_θ(x_t, t): Denoised mean before adding noise
                "x_t_minus_1": x_t_minus_1,  # x_{t-1}: Denoised sample after adding noise
                "alpha_t": alpha_t,  # α_t: Alpha for this step
                "alpha_bar_t": alpha_bar_t,  # ᾱ_t: Cumulative product of alphas up to t
                "beta_t": beta_t,  # β_t: Noise schedule value for this step
                "sigma_t": sigma_t,  # σ_t: Noise std added at this step (√β_t)
            }
        return x_t_minus_1

    def reverse_diffusion_step(self, x_t, t, return_all=False):
        """
        Single reverse diffusion step to estimate x_{t-1} given x_t and t.
        Implements improved DDPM from Nichol & Dhariwal (2021).

        Key improvements over Ho et al. (2020):
        1. Uses the true posterior variance β̃_t instead of fixed β_t:
           - Ho et al. used σ_t^2 = β_t for simplicity
           - Nichol & Dhariwal derive the correct variance β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
           - This better matches the true posterior q(x_{t-1}|x_t,x_0)

        2. Explicit x_0 prediction:
           - Ho et al. used noise prediction ε_θ to compute mean
           - Nichol & Dhariwal show this is equivalent to predicting x_0
           - Use x_0 prediction in mean calculation for better interpretability

        For t > 1:
            # Eq. 10: True posterior variance
            β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t

            # New to Nicole & Dhariwal 2021
            x_0 = x_t - ε_θ(x_t, t)
            # Eq. 11: Mean using x_0 prediction
            μ̃_t(x_t, x_0) = (√ᾱ_{t-1}*β_t)/(1-ᾱ_t) * x_0 + (√α_t*(1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t

            # Eq. 12: True posterior distribution
            q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)

            # Sample using reparameterization:
            z ~ N(0, I), σ_t = √β̃_t
            x_{t-1} = μ̃_t(x_t,x_0) + σ_t * z

        For t = 1:
            β̃_1 = 0, z = 0
            x_0 = x_1 - ε_θ(x_1, 1)
            x_{t-1} = μ̃_1(x_1, x_0)

        The key improvement over reverse_diffusion_step() is using the improved
        variance β̃_t (Eq. 10) and mean μ̃_t (Eq. 11) which better match the
        true posterior q(x_{t-1}|x_t,x_0).

        Args:
            x_t: Noisy sample at timestep t, shape [B, C, seq_len] (must be on correct device)
            t: Current timestep (tensor of shape [B] or scalar int, 1-based)
        Returns:
            x_{t-1}: Sample at timestep t-1, same shape as x_t
        """
        # Ensure tensors are on the correct device
        device = x_t.device

        # Convert integer timestep to tensor if needed
        if isinstance(t, int):
            t = torch.tensor([t], device=device)
        else:
            t = tensor_to_device(t, device)

        # Ensure x_t has the correct shape [B, C, L]
        # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, comes from ForwardDiffusion
        x_t = prepare_batch_shape(x_t)

        # Get diffusion parameters for timestep t
        # β_t, α_t, ᾱ_t (beta, alpha, and cumulative alpha)
        beta_t = tensor_to_device(self.forward_diffusion.beta(t), device)  # β_t
        alpha_t = tensor_to_device(self.forward_diffusion.alpha(t), device)  # α_t
        alpha_bar_t = tensor_to_device(
            self.forward_diffusion.alpha_bar(t), device
        )  # ᾱ_t

        # Broadcast parameters to match x_t's dimensions
        ndim = x_t.ndim
        beta_t = bcast_right(beta_t, ndim)
        alpha_t = bcast_right(alpha_t, ndim)
        alpha_bar_t = bcast_right(alpha_bar_t, ndim)

        # === Step 1: Predict x_0 using noise predictor ===
        epsilon_theta = self.noise_predictor(x_t, t)
        x_0 = x_t - epsilon_theta

        # === Step 2: Get ᾱ_{t-1} for both mean and variance ===
        t_minus_1 = t - 1 if not (t == 1).all() else torch.zeros_like(t)
        alpha_bar_prev = tensor_to_device(
            self.forward_diffusion.alpha_bar(t_minus_1), device
        )

        # === Step 3: Compute β̃_t using Eq. 10 ===
        # β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        beta_tilde = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t
        if (t == 1).all():
            beta_tilde = torch.zeros_like(beta_t)

        # === Step 4: Compute μ̃_t using Eq. 11 ===
        # μ̃_t(x_t, x_0) = (√ᾱ_{t-1}*β_t)/(1-ᾱ_t) * x_0 + (√α_t*(1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
        coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)
        mean = coef_x0 * x_0 + coef_xt * x_t
        # mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        # === Step 5: Sample from posterior using Eq. 12 ===
        # q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I)
        # z ~ N(0, I), β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        z = torch.randn_like(x_t) if not (t == 1).all() else torch.zeros_like(x_t)

        # x_{t-1} = μ̃_t(x_t,x_0) + σ_t * z, where σ_t = √β̃_t
        sigma_t = torch.sqrt(beta_tilde)  # σ_t
        x_t_minus_1 = mean + sigma_t * z

        # === Debugging: Print Important Variables per Timestep ===
        if return_all:
            print(
                f"t: {t.item()}, β_t: {beta_t.flatten()[0].item():.10f}, α_t: {alpha_t.flatten()[0].item():.10f}, ᾱ_t: {alpha_bar_t.flatten()[0].item():.10f}, β~_t: {beta_tilde.flatten()[0].item():.10f}, σ_t: {sigma_t.flatten()[0].item():.10f}"
            )
            return {
                "epsilon_theta": epsilon_theta,  # ε_θ(x_t, t): Model's predicted noise
                "x0": x_0,  # x_0: predicted signal
                "xt": x_t,  # x_t: noisy signal
                "coef_x0": coef_x0,  # (√ᾱ_{t-1}*β_t)/(1-ᾱ_t): Coefficient for x_0
                "coef_xt": coef_xt,  # (√α_t*(1-ᾱ_{t-1}))/(1-ᾱ_t): Coefficient for x_t
                "mean": mean,  # μ̃_t(x_t, x_0): Denoised mean before adding noise
                "beta_tilde": beta_tilde,  # β̃_t: Posterior variance
                "x_t_minus_1": x_t_minus_1,  # x_{t-1}: Denoised sample after adding noise
                "alpha_t": alpha_t,  # α_t: Alpha for this step
                "alpha_bar_t": alpha_bar_t,  # ᾱ_t: Cumulative product of alphas up to t
                "beta_t": beta_t,  # β_t: Noise schedule value for this step
                "sigma_t": sigma_t,  # √β̃_t: Noise std added at this step
            }
        return x_t_minus_1

    def _reverse_diffusion_process(self, x, denoise_step=1, discretize=False):
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
        # for t in range(tmax, 0, -1):
        for t in reversed(range(tmin, tmax + 1, denoise_step)):
            x = self.reverse_diffusion_step(x, t)

        # Post-processing for SNP data
        if discretize:
            x = torch.round(x * 4) / 4
            x = torch.clamp(x, 0, 0.5)

        return x

    def generate_samples(
        self, num_samples=10, denoise_step=1, discretize=False, seed=42, device=None
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
        self, batch, denoise_step=1, discretize=False, seed=42, device=None
    ):
        """
        Denoise an input batch using the reverse diffusion process.

        This method starts the reverse diffusion process from the provided batch
        (which should be noisy or real data). It does not generate new noise, but
        instead denoises the actual input batch by iteratively applying the reverse
        diffusion steps. This is useful for denoising specific data samples, e.g.,
        for evaluation or restoration tasks.

        Args:
            batch: Input batch to denoise. Shape: [B, C, seq_len]. This can be noisy or real data.
            denoise_step: Number of timesteps to skip in reverse diffusion. If None, uses the instance default.
            discretize: If True, discretize output to SNP values. If None, uses the instance default.
            seed: Random seed for reproducibility (not used in this function, kept for API compatibility).
            device: Device to run the computation on. If None, uses CUDA if available, else CPU.

        Returns:
            Denoised batch of the same shape as input.
        """

        # Set seed for reproducibility
        set_seed(seed)

        # Determine device to use
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure batch has the correct shape and is on the right device
        x = tensor_to_device(prepare_batch_shape(batch), device)

        # Use instance defaults if not provided
        if denoise_step is None:
            denoise_step = self.denoise_step
        if discretize is None:
            discretize = self.discretize

        # Run the reverse diffusion process starting from the input batch
        return self._reverse_diffusion_process(x, denoise_step, discretize)
