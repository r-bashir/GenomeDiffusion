#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def weighted_mse_loss(predicted_noise, true_noise, t):
    """
    Computes weighted MSE loss, giving higher weight to early timesteps.

    Args:
        predicted_noise: Model's noise prediction
        true_noise: Actual noise that was added
        t: Timestep tensor

    Returns:
        Weighted MSE loss
    """
    t = t.float()
    weights = 1.0 / (t + 1)  # +1 to avoid division by zero
    weights = weights / weights.mean()  # Normalize to maintain scale
    mse = (predicted_noise - true_noise) ** 2
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # Mean over spatial dimensions
    weighted_loss = (weights * mse).mean()
    return weighted_loss.item()


def test_diffusion_at_timestep(
    model, x0, timestep, plot=True, save_plot=True, output_dir="diffusion_plots"
):
    """Test the diffusion process at a specific timestep.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at
        plot: Whether to generate plots
        save_plot: Whether to save plots to files
        output_dir: Directory to save plots (only used if save_plot=True)

    Returns:
        Dictionary containing all computed values and metrics
    """
    model.eval()

    with torch.no_grad():
        # Create timestep tensor
        batch_size = x0.shape[0]
        t_tensor = torch.full(
            (batch_size,), timestep, device=x0.device, dtype=torch.long
        )

        # Sample random noise
        noise = torch.randn_like(x0)

        # Forward diffusion
        xt = model.ddpm.sample(x0, t_tensor, noise)

        # Get model's noise prediction
        predicted_noise = model.predict_added_noise(xt, t_tensor)

        # Reverse diffusion step
        x_t_minus_1 = model.reverse_diffusion(xt, t_tensor)

        # Calculate metrics
        mse = F.mse_loss(predicted_noise, noise).item()
        weighted_mse = weighted_mse_loss(predicted_noise, noise, t_tensor)
        x0_diff = F.mse_loss(x_t_minus_1, x0).item()

        # Print debug information
        print(f"\n=== Timestep {timestep} ===")
        print(f"- Input shape: {x0.shape}")
        print(f"- Noise shape: {noise.shape}")
        print(f"- Predicted noise shape: {predicted_noise.shape}")
        print(f"- MSE (noise vs predicted): {mse:.6f}")
        print(f"- Weighted MSE: {weighted_mse:.6f}")
        print(f"- Reconstruction MSE (x0 vs x_t-1): {x0_diff:.6f}")

        # Plot if requested
        if plot:
            save_dir = output_dir if save_plot else None
            plot_diffusion_process(
                x0, noise, xt, predicted_noise, x_t_minus_1, timestep, save_dir=save_dir
            )

    # Return all computed values
    return {
        "x0": x0,
        "noise": noise,
        "xt": xt,
        "predicted_noise": predicted_noise,
        "x_t_minus_1": x_t_minus_1,
        "metrics": {
            "mse": mse,
            "weighted_mse": weighted_mse,
            "x0_diff": x0_diff,
        },
        "timestep": timestep,
    }


def to_numpy(x):
    """Convert tensor to numpy array, handling batch and channel dimensions."""
    return x[0, 0].detach().cpu().numpy()


def plot_diffusion_process(
    x0, noise, x_t, predicted_noise, x_t_minus_1, timestep, save_dir=None
):
    """Plot all diffusion process steps in separate figures.

    Args:
        x0: Original input tensor [B, C, seq_len]
        noise: Added noise tensor
        x_t: Noisy input tensor at timestep t
        predicted_noise: Model's noise prediction
        x_t_minus_1: Denoised output tensor
        timestep: Current timestep
        save_dir: Directory to save the plots (if None, show plots)
    """
    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Generate all plots
    plot_signal_comparison(x0, x_t, timestep, save_dir)
    plot_noise_comparison(noise, predicted_noise, timestep, save_dir)
    plot_denoising_comparison(x0, x_t_minus_1, timestep, save_dir)


def plot_signal_comparison(x0, x_t, timestep, save_dir=None):
    """Plot original vs noisy signal.

    Args:
        x0: Original input tensor [B, C, seq_len]
        x_t: Noisy input tensor at timestep t
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    x0_np = to_numpy(x0)
    x_t_np = to_numpy(x_t)

    plt.figure(figsize=(12, 4))
    plt.title(f"Original vs Noisy Signal (t={timestep})")
    plt.plot(x0_np[:100], label="Original x0", alpha=0.8)
    plt.plot(x_t_np[:100], label=f"Noisy x_t", alpha=0.8)
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/signal_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def plot_noise_comparison(noise, predicted_noise, timestep, save_dir=None):
    """Plot true vs predicted noise.

    Args:
        noise: True noise tensor
        predicted_noise: Model's noise prediction
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    noise_np = to_numpy(noise)
    pred_noise_np = to_numpy(predicted_noise)

    plt.figure(figsize=(12, 4))
    plt.title(f"True vs Predicted Noise (t={timestep})")
    plt.plot(noise_np[:100], label="True Noise", alpha=0.8)
    plt.plot(pred_noise_np[:100], label="Predicted Noise", alpha=0.8)
    plt.legend()
    plt.ylabel("Noise")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/noise_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def plot_denoising_comparison(x0, x_t_minus_1, timestep, save_dir=None):
    """Plot original vs denoised signal.

    Args:
        x0: Original input tensor
        x_t_minus_1: Denoised output tensor
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    x0_np = to_numpy(x0)
    x_t_minus_1_np = to_numpy(x_t_minus_1)

    plt.figure(figsize=(12, 4))
    plt.title(f"Original vs Denoised Signal (t={timestep})")
    plt.plot(x0_np[:100], label="Original x0", alpha=0.8)
    plt.plot(x_t_minus_1_np[:100], label="Denoised x_(t-1)", alpha=0.8)
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/denoised_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def display_diffusion_parameters(model, timestep):
    """Display diffusion process parameters for a specific timestep.

    Args:
        model: The diffusion model
        timestep: Timestep to display parameters for
    """
    # Get parameters directly
    alpha_t = model.ddpm._alphas_t[timestep - 1].item()
    alpha_bar_t = model.ddpm._alphas_bar_t[timestep].item()
    sigma_t = model.ddpm._sigmas_t[timestep].item()
    beta_t = 1.0 - alpha_t

    # Print formatted parameters
    print(f"\n--- Diffusion Parameters at t={timestep} ---")
    print(f"β_{timestep} = {beta_t:.6f}")
    print(f"α_{timestep} = {alpha_t:.6f}")
    print(f"α̅_{timestep} = {alpha_bar_t:.6f}")
    print(f"σ_{timestep} = {sigma_t:.6f}")
    print(f"√α_{timestep} = {alpha_t**0.5:.6f}")
    print(f"√(1-α̅_{timestep}) = {(1-alpha_bar_t)**0.5:.6f}")
    print(
        f"Forward diffusion equation: x_{timestep} = {alpha_t**0.5:.3f}·x₀ + {sigma_t:.3f}·ε"
    )

    return alpha_t, alpha_bar_t, sigma_t, beta_t


def print_diffusion_results(
    x0, noise, xt, predicted_noise, x_t_minus_1, metrics, timestep
):
    """Print the results of a diffusion step.

    Args:
        x0, noise, xt, predicted_noise, x_t_minus_1: Tensors from diffusion process
        metrics: Dictionary of computed metrics
        timestep: Current timestep
    """
    # Print signal analysis
    print(f"\nSignal Analysis:")
    print(f"1. Original x0 (first 10):\n{x0[0,0,:10].cpu().numpy()}")
    print(f"2. Added noise (first 10):\n{noise[0,0,:10].cpu().numpy()}")
    print(f"3. Noisy xt (first 10):\n{xt[0,0,:10].cpu().numpy()}")
    print(f"4. Predicted noise (first 10):\n{predicted_noise[0,0,:10].cpu().numpy()}")
    print(f"5. Denoised output (first 10):\n{x_t_minus_1[0,0,:10].cpu().numpy()}")

    # Print metrics
    print("\nMetrics:")
    print(f"- Average predicted noise magnitude: {metrics['pred_noise_magnitude']:.6f}")
    print(f"- True vs Predicted noise MSE: {metrics['noise_mse']:.6f}")
    print(f"- Original vs Denoised MSE: {metrics['x0_diff']:.6f}")
