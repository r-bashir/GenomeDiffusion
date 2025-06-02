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
        xt = model.forward_diffusion.sample(x0, t_tensor, noise)

        # Get model's noise prediction
        predicted_noise = model.predict_added_noise(xt, t_tensor)

        # Reverse diffusion step
        x_t_minus_1 = model.reverse_diffusion.reverse_diffusion_step(xt, t_tensor)

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


def visualize_diffusion_process(
    model, batch, timesteps=[100, 500, 900], output_dir=None
):
    """Visualize the forward and reverse diffusion process at different timesteps.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
    """
    print("\nVisualizing diffusion process at different timesteps...")
    model.eval()
    device = next(model.parameters()).device

    # Ensure batch has the right shape and is on the correct device
    batch = batch.to(device)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)  # Add channel dimension if needed

    # Use only the first sample for visualization
    x0 = batch[0:1]  # Shape: [1, C, seq_len]

    # Create a figure with 5 columns (original, noise, noisy, pred_noise, denoised)
    # and one row per timestep
    fig, axes = plt.subplots(len(timesteps), 5, figsize=(20, 4 * len(timesteps)))

    # Handle the case of a single timestep
    if len(timesteps) == 1:
        axes = axes.reshape(1, -1)  # Reshape to 2D array with 1 row

    # Set column titles
    axes[0, 0].set_title("Original Sample")
    axes[0, 1].set_title("Added Noise")
    axes[0, 2].set_title("Noisy Sample")
    axes[0, 3].set_title("Predicted Noise")
    axes[0, 4].set_title("Denoised Sample")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # 1. Original sample (x0)
            # Already have x0

            # 2. Generate noise
            eps = torch.randn_like(x0)

            # 3. Forward diffusion: Add noise to get x_t
            x_t = model.forward_diffusion.sample(x0, t_tensor, eps)

            # 4. Predict noise
            pred_eps = model.predict_added_noise(x_t, t_tensor)

            # 5. Reverse diffusion: Denoise x_t to get x_0_pred
            # For a single step denoising, we'll use the reverse_diffusion_step method
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)

            # Prepare data for visualization - reshape to 2D
            # For 1D data, we'll reshape to [1, seq_len] for imshow
            x0_vis = x0.cpu().squeeze().numpy().reshape(1, -1)
            eps_vis = eps.cpu().squeeze().numpy().reshape(1, -1)
            x_t_vis = x_t.cpu().squeeze().numpy().reshape(1, -1)
            pred_eps_vis = pred_eps.cpu().squeeze().numpy().reshape(1, -1)
            x_0_pred_vis = x_0_pred.cpu().squeeze().numpy().reshape(1, -1)

            # Plot results
            # 1. Original
            im0 = axes[i, 0].imshow(x0_vis, aspect="auto", cmap="viridis")
            axes[i, 0].set_ylabel(f"t={t}")
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            # 2. Noise
            im1 = axes[i, 1].imshow(eps_vis, aspect="auto", cmap="viridis")
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

            # 3. Noisy
            im2 = axes[i, 2].imshow(x_t_vis, aspect="auto", cmap="viridis")
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])

            # 4. Predicted Noise
            im3 = axes[i, 3].imshow(pred_eps_vis, aspect="auto", cmap="viridis")
            axes[i, 3].set_xticks([])
            axes[i, 3].set_yticks([])

            # 5. Denoised
            im4 = axes[i, 4].imshow(x_0_pred_vis, aspect="auto", cmap="viridis")
            axes[i, 4].set_xticks([])
            axes[i, 4].set_yticks([])

            # Add text with stats
            axes[i, 0].text(
                0.5,
                -0.15,
                f"Mean: {x0.mean():.3f}\nStd: {x0.std():.3f}",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 1].text(
                0.5,
                -0.15,
                f"Mean: {eps.mean():.3f}\nStd: {eps.std():.3f}",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 2].text(
                0.5,
                -0.15,
                f"Mean: {x_t.mean():.3f}\nStd: {x_t.std():.3f}",
                transform=axes[i, 2].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 3].text(
                0.5,
                -0.15,
                f"Mean: {pred_eps.mean():.3f}\nStd: {pred_eps.std():.3f}",
                transform=axes[i, 3].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 4].text(
                0.5,
                -0.15,
                f"Mean: {x_0_pred.mean():.3f}\nStd: {x_0_pred.std():.3f}",
                transform=axes[i, 4].transAxes,
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    if output_dir:
        plt.savefig(
            output_dir / "diffusion_process_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    return fig


def visualize_diffusion_process_lineplot(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the forward and reverse diffusion process at different timesteps using line plots.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization (to avoid overcrowded plots)
    """
    print("\nVisualizing diffusion process using line plots...")
    model.eval()
    device = next(model.parameters()).device

    # Ensure batch has the right shape and is on the correct device
    batch = batch.to(device)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)  # Add channel dimension if needed

    # Use only the first sample for visualization
    x0 = batch[0:1]  # Shape: [1, C, seq_len]

    # Get the sequence length
    seq_len = x0.shape[-1]

    # If sequence is too long, sample points for visualization
    if seq_len > sample_points:
        indices = np.linspace(0, seq_len - 1, sample_points, dtype=int)
    else:
        indices = np.arange(seq_len)

    # Create a figure with 5 columns (original, noise, noisy, pred_noise, denoised)
    # and one row per timestep
    fig, axes = plt.subplots(len(timesteps), 5, figsize=(20, 4 * len(timesteps)))

    # Handle the case of a single timestep
    if len(timesteps) == 1:
        axes = axes.reshape(1, -1)  # Reshape to 2D array with 1 row

    # Set column titles
    axes[0, 0].set_title("Original Sample")
    axes[0, 1].set_title("Added Noise")
    axes[0, 2].set_title("Noisy Sample")
    axes[0, 3].set_title("Predicted Noise")
    axes[0, 4].set_title("Denoised Sample")

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # 1. Original sample (x0)
            # Already have x0

            # 2. Generate noise
            eps = torch.randn_like(x0)

            # 3. Forward diffusion: Add noise to get x_t
            x_t = model.forward_diffusion.sample(x0, t_tensor, eps)

            # 4. Predict noise
            pred_eps = model.predict_added_noise(x_t, t_tensor)

            # 5. Reverse diffusion: Denoise x_t to get x_0_pred
            # For a single step denoising, we'll use the reverse_diffusion_step method
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)

            # Prepare data for visualization - convert to numpy and sample points
            x0_vis = x0.cpu().squeeze().numpy()[indices]
            eps_vis = eps.cpu().squeeze().numpy()[indices]
            x_t_vis = x_t.cpu().squeeze().numpy()[indices]
            pred_eps_vis = pred_eps.cpu().squeeze().numpy()[indices]
            x_0_pred_vis = x_0_pred.cpu().squeeze().numpy()[indices]

            # Create x-axis for plotting
            x_axis = np.arange(len(indices))

            # Plot results
            # 1. Original
            axes[i, 0].plot(x_axis, x0_vis, "b-", linewidth=1)
            axes[i, 0].set_ylabel(f"t={t}")
            axes[i, 0].set_ylim(-0.1, 0.6)  # Set consistent y-axis limits for SNP data

            # 2. Noise
            axes[i, 1].plot(x_axis, eps_vis, "r-", linewidth=1)
            axes[i, 1].set_ylim(-3, 3)  # Typical range for noise

            # 3. Noisy
            axes[i, 2].plot(x_axis, x_t_vis, "g-", linewidth=1)
            axes[i, 2].set_ylim(-3, 3)  # Allow wider range for noisy data

            # 4. Predicted Noise
            axes[i, 3].plot(x_axis, pred_eps_vis, "m-", linewidth=1)
            axes[i, 3].set_ylim(-3, 3)  # Typical range for noise

            # 5. Denoised
            axes[i, 4].plot(x_axis, x_0_pred_vis, "c-", linewidth=1)
            axes[i, 4].set_ylim(-0.1, 0.6)  # Set consistent y-axis limits for SNP data

            # Add text with stats
            axes[i, 0].text(
                0.5,
                -0.15,
                f"Mean: {x0.mean():.3f}\nStd: {x0.std():.3f}",
                transform=axes[i, 0].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 1].text(
                0.5,
                -0.15,
                f"Mean: {eps.mean():.3f}\nStd: {eps.std():.3f}",
                transform=axes[i, 1].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 2].text(
                0.5,
                -0.15,
                f"Mean: {x_t.mean():.3f}\nStd: {x_t.std():.3f}",
                transform=axes[i, 2].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 3].text(
                0.5,
                -0.15,
                f"Mean: {pred_eps.mean():.3f}\nStd: {pred_eps.std():.3f}",
                transform=axes[i, 3].transAxes,
                ha="center",
                fontsize=8,
            )
            axes[i, 4].text(
                0.5,
                -0.15,
                f"Mean: {x_0_pred.mean():.3f}\nStd: {x_0_pred.std():.3f}",
                transform=axes[i, 4].transAxes,
                ha="center",
                fontsize=8,
            )

            # Hide x-axis ticks except for the last row
            if i < len(timesteps) - 1:
                for j in range(5):
                    axes[i, j].set_xticks([])

    plt.tight_layout()
    if output_dir:
        plt.savefig(
            output_dir / "diffusion_process_lineplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    return fig


def display_diffusion_parameters(model, timestep):
    """Display diffusion process parameters for a specific timestep.

    Args:
        model: The diffusion model
        timestep: Timestep to display parameters for
    """
    # Get parameters directly
    alpha_t = model.forward_diffusion._alphas_t[timestep - 1].item()
    alpha_bar_t = model.forward_diffusion._alphas_bar_t[timestep].item()
    sigma_t = model.forward_diffusion._sigmas_t[timestep].item()
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
