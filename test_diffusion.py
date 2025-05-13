#!/usr/bin/env python
# coding: utf-8

"""
Test script for diffusion model parameters and behavior.

This script analyzes the diffusion process parameters at different timesteps
and visualizes how data transforms during forward and reverse diffusion.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import math
import yaml
from torch.utils.data import DataLoader

from diffusion import DiffusionModel, SNPDataset

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_diffusion_process(
    x0, noise, x_t, predicted_noise, x_t_minus_1, timestep, save_path=None
):
    """Plot the diffusion process steps.

    Args:
        x0: Original input [B, C, seq_len]
        noise: Added noise
        x_t: Noisy input at timestep t
        predicted_noise: Model's noise prediction
        x_t_minus_1: Denoised output
        timestep: Current timestep
        save_path: Optional path to save the plot
    """

    # Convert tensors to numpy arrays
    def to_numpy(x):
        return x[0, 0].detach().cpu().numpy()

    x0 = to_numpy(x0)
    noise = to_numpy(noise)
    x_t = to_numpy(x_t)
    predicted_noise = to_numpy(predicted_noise)
    x_t_minus_1 = to_numpy(x_t_minus_1)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Diffusion Process at t={timestep}")

    # Plot original and noisy signals
    axes[0].plot(x0[:100], label="Original x0", alpha=0.8)
    axes[0].plot(x_t[:100], label=f"Noisy x_t", alpha=0.8)
    axes[0].legend()
    axes[0].set_ylabel("Signal")
    axes[0].grid(True)

    # Plot true and predicted noise
    axes[1].plot(noise[:100], label="True Noise", alpha=0.8)
    axes[1].plot(predicted_noise[:100], label="Predicted Noise", alpha=0.8)
    axes[1].legend()
    axes[1].set_ylabel("Noise")
    axes[1].grid(True)

    # Plot original and denoised signals
    axes[2].plot(x0[:100], label="Original x0", alpha=0.8)
    axes[2].plot(x_t_minus_1[:100], label="Denoised x_(t-1)", alpha=0.8)
    axes[2].legend()
    axes[2].set_ylabel("Signal")
    axes[2].set_xlabel("Position")
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


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


def run_diffusion_step(model, x0, timestep):
    """Run a single step of forward and reverse diffusion.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at

    Returns:
        tuple: (x0, noise, xt, predicted_noise, x_t_minus_1, metrics)
    """
    with torch.no_grad():
        # Create timestep tensor
        t = torch.full((x0.shape[0],), timestep, dtype=torch.long, device=x0.device)

        # Sample random noise
        noise = torch.randn_like(x0)

        # 1. Forward diffusion: Add noise to input
        xt = model.ddpm.sample(x0, t, noise)

        # 2. Model prediction (should be ≈ 0 for zero-initialized model)
        predicted_noise = model.predict_added_noise(xt, t)

        # 3. Reverse process using predicted noise
        x_t_minus_1 = model.reverse_denoising(xt, t)

        # Compute metrics
        metrics = {
            "noise_mse": F.mse_loss(predicted_noise, noise).item(),
            "pred_noise_magnitude": torch.mean(torch.abs(predicted_noise)).item(),
            "x0_diff": F.mse_loss(x_t_minus_1, x0).item(),
        }

        return x0, noise, xt, predicted_noise, x_t_minus_1, metrics


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
    print(f"1. Original x0 (first 30):\n{x0[0,0,:30].cpu().numpy()}")
    print(f"2. Added noise (first 30):\n{noise[0,0,:30].cpu().numpy()}")
    print(f"3. Noisy xt (first 30):\n{xt[0,0,:30].cpu().numpy()}")
    print(f"4. Predicted noise (first 30):\n{predicted_noise[0,0,:30].cpu().numpy()}")
    print(f"5. Denoised output (first 30):\n{x_t_minus_1[0,0,:30].cpu().numpy()}")

    # Print metrics
    print("\nMetrics:")
    print(f"- Average predicted noise magnitude: {metrics['pred_noise_magnitude']:.6f}")
    print(f"- True vs Predicted noise MSE: {metrics['noise_mse']:.6f}")
    print(f"- Original vs Denoised MSE: {metrics['x0_diff']:.6f}")


def test_diffusion_at_timestep(
    model, x0, timestep, plot=True, save_plot=True
):  # plot/save_plot args retained for compatibility, always True in main()
    """Test the diffusion process at a specific timestep.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at
        plot: Whether to plot the results
        save_plot: Whether to save the plot to a file (if plot=True)
    """
    # 1. Display parameters
    display_diffusion_parameters(model, timestep)

    # 2. Run diffusion
    results = run_diffusion_step(model, x0, timestep)
    x0, noise, xt, predicted_noise, x_t_minus_1, metrics = results

    # 3. Print results
    print_diffusion_results(
        x0, noise, xt, predicted_noise, x_t_minus_1, metrics, timestep
    )

    # 4. Plot if requested
    if plot:
        save_path = f"diffusion_t{timestep}.png" if save_plot else None
        plot_diffusion_process(
            x0, noise, xt, predicted_noise, x_t_minus_1, timestep, save_path=save_path
        )


def load_config(config_path):
    """Load configuration from yaml file."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main function to test diffusion model parameters and behavior."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load configuration
    config = load_config("config.yaml")

    # Set diffusion steps
    print("\nTesting Diffusion Process")
    print("======================")
    print(f"Diffusion steps: {config['diffusion']['diffusion_steps']}")
    print(f"Schedule type: {config['diffusion']['schedule_type']}")
    print(f"Device: {device}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = SNPDataset(
        config.get("input_path"), seq_length=config.get("data").get("seq_length")
    )

    # Create data loader and get a batch
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    x0 = next(iter(loader)).unsqueeze(1).to(device)  # shape=[B, C, seq_len]

    # Initialize model
    print("Initializing model...")
    model = DiffusionModel(hparams=config)
    model.eval()

    # Define timesteps to test
    # Early (1, 2), middle (250, 500), and late (750, 999, 1000) timesteps
    timesteps_to_test = [1, 2, 250, 500, 750, 999, 1000]
    print(f"\nAnalyzing diffusion process at timesteps: {timesteps_to_test}")

    # Always plot and save for all timesteps
    print("\nPlotting and saving results for all timesteps...")
    for t in timesteps_to_test:
        test_diffusion_at_timestep(model, x0, timestep=t, plot=True, save_plot=True)


if __name__ == "__main__":
    main()
