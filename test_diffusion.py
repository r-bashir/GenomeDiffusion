#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import math
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


def test_zero_noise_prediction(
    model: DiffusionModel, x0: torch.Tensor, timestep: int = 2, plot=True
):
    """Test the behavior of zero-noise prediction model.

    Since the model is initialized to predict zero noise, we expect:
    1. predicted_noise ≈ 0
    2. x_t to be noisy version of x0
    3. x_t_minus_1 to be closer to the mean of the forward process

    Args:
        model: The diffusion model (initialized with zero weights)
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at
        plot: Whether to plot the results
    """
    with torch.no_grad():
        # 1. Forward process: Add noise to input
        noise = torch.randn_like(x0)
        t = torch.full((x0.shape[0],), timestep, dtype=torch.long, device=x0.device)
        x_t = model.ddpm.sample(x0, t, noise)

        # 2. Model prediction (should be ≈ 0)
        predicted_noise = model.predict_added_noise(x_t, t)

        # 3. Reverse process using predicted noise
        x_t_minus_1 = model.reverse_denoising(x_t, t)

        # Compute metrics
        noise_mse = F.mse_loss(predicted_noise, noise)
        pred_noise_magnitude = torch.mean(torch.abs(predicted_noise)).item()
        x0_diff = F.mse_loss(x_t_minus_1, x0)

        # Print analysis
        # Get noise schedule parameters for display
        alpha_t = model.ddpm.alpha(t)
        # Calculate sigma directly to ensure it matches the formula
        # For timestep t, we need alphas_bar at index t
        alpha_bar_t = model.ddpm._alphas_bar_t[timestep].to(t.device)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)

        print(f"\nZero Noise Model Analysis at t={timestep}:")
        print(f"Noise Schedule Parameters:")
        print(f"- α_t: {alpha_t[0].item():.6f}")
        print(f"- σ_t: {sigma_t.item():.6f}")
        print(f"\nSignal Analysis:")
        print(f"1. Original x0 (first 10):\n{x0[0,0,:10].cpu().numpy()}")
        print(f"2. Added noise (first 10):\n{noise[0,0,:10].cpu().numpy()}")
        print(f"3. Noisy x_t (first 10):\n{x_t[0,0,:10].cpu().numpy()}")
        print(
            f"4. Predicted noise (should be ≈ 0):\n{predicted_noise[0,0,:10].cpu().numpy()}"
        )
        print(f"5. Denoised output (first 10):\n{x_t_minus_1[0,0,:10].cpu().numpy()}")
        print("\nMetrics:")
        print(
            f"- Average predicted noise magnitude: {pred_noise_magnitude:.6f} (should be ≈ 0)"
        )
        print(f"- True vs Predicted noise MSE: {noise_mse.item():.6f}")
        print(f"- Original vs Denoised MSE: {x0_diff.item():.6f}")

        # Plot results
        if plot:
            plot_diffusion_process(
                x0,
                noise,
                x_t,
                predicted_noise,
                x_t_minus_1,
                timestep,
                save_path=f"zero_noise_t{timestep}.png",
            )


def load_config(config_path):
    """Load configuration from yaml file."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Setup
    torch.manual_seed(42)
    config = load_config("config.yaml")

    # Modify config for minimum diffusion steps and proper dimensions
    config["diffusion"]["diffusion_steps"] = 2  # This means t ∈ [1,2]
    print("\nTesting minimum-step diffusion (t ∈ [1,2])")
    print("Expected behavior:")
    print("1. t=1: First noise addition step")
    print("2. t=2: Maximum noise step")
    print("3. Predicted noise should be ≈ 0 (zero weights)")

    # Load data
    dataset = SNPDataset(
        config.get("input_path"), seq_length=config.get("data").get("seq_length")
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    x0 = next(iter(loader)).unsqueeze(1).to(device)  # shape=[B, C, seq_len]

    # Initialize model with 2-step diffusion
    model = DiffusionModel(hparams=config)
    model.eval()

    print("\nTesting both possible timesteps:")
    # Test both possible timesteps
    for t in [1, 2]:  # Only valid timesteps for 2-step diffusion
        print(f"\n--- Timestep {t} ---")
        test_zero_noise_prediction(model, x0, timestep=t, plot=True)

        # Show beta and alpha values for this step
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        # Use methods instead of properties
        alpha_t = model.ddpm.alpha(t_tensor).item()

        # Calculate sigma directly from alphas_bar
        # For t=1, we need sqrt(1-alpha_bar_1)
        # For t=2, we need sqrt(1-alpha_bar_2)
        alpha_bar_t = model.ddpm._alphas_bar_t[t].item()
        sigma_t = math.sqrt(1.0 - alpha_bar_t)

        beta_t = 1.0 - alpha_t  # Beta = 1 - alpha
        print(f"\nDiffusion parameters at t={t}:")
        print(f"β_{t} = {beta_t:.6f}")
        print(f"α_{t} = {alpha_t:.6f}")
        print(f"σ_{t} = {sigma_t:.6f}  # Should equal √(1-α_{t})")
        print(f"√α_{t} = {alpha_t**0.5:.6f}")
        print(f"√(1-α_{t}) = {(1-alpha_t)**0.5:.6f}")
        print(f"x_{t} = {alpha_t**0.5:.3f}·x₀ + {sigma_t:.3f}·ε")


if __name__ == "__main__":
    main()
