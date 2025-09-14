#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cosine Beta Schedule
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    Implements the cosine beta schedule as introduced in:
    Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021).

    Arguments:
    - timesteps: Total number of diffusion steps (T).
    - s: Small offset to prevent alpha_bar from being too small near t=0 (default: 0.008).

    Returns:
    - betas: An array of length `timesteps`, each beta_t ∈ (0, 1)
    """

    # Create t values in range [0, 1] with (T + 1) points
    # This represents normalized time: t / T
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps) / timesteps

    # Compute the cumulative product of alphas (ᾱₜ) using cosine schedule
    # Equation: ᾱₜ = cos²(((t + s)/(1 + s)) * (π / 2))
    # This starts near 1 and smoothly decreases toward 0 as t increases
    alphas_bar = np.cos(((t + s) / (1 + s)) * np.pi / 2) ** 2

    # Normalize so that ᾱ₀ = 1 exactly (important for stability)
    alphas_bar = alphas_bar / alphas_bar[0]

    # Compute βₜ from ᾱₜ
    # Since ᾱₜ = Π_{s=1}^{t} α_s, we can derive:
    #     α_t = ᾱ_t / ᾱ_{t-1}
    #     β_t = 1 - α_t = 1 - (ᾱ_t / ᾱ_{t-1})
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])

    # Clip the values to be within [1e-8, 0.999] for numerical stability
    betas = np.clip(betas, a_min=1e-8, a_max=0.999)

    return betas


# Linear Beta Schedule
def linear_beta_schedule(
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
