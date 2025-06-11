#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src import DiffusionModel

# Type aliases for better readability
DiffusionAnalysisResult = Dict[str, Any]
DiffusionAnalysisResults = Dict[int, DiffusionAnalysisResult]


def weighted_mse_loss(predicted_noise, true_noise, t, model=None):
    """
    Computes weighted MSE loss, giving higher weight to early timesteps.
    Scales the predicted noise by 1/sigma_t to match the true noise distribution N(0,1).

    Args:
        predicted_noise: Model's noise prediction
        true_noise: Actual noise that was added (sampled from N(0,1))
        t: Timestep tensor
        model: The diffusion model (needed to get sigma values)

    Returns:
        Weighted MSE loss
    """
    t = t.float()
    weights = 1.0 / (t + 1)  # +1 to avoid division by zero
    weights = weights / weights.mean()  # Normalize to maintain scale

    # Scale predicted noise by 1/sigma_t if model is provided
    if model is not None:
        # Get sigma values for each timestep in the batch
        sigmas = torch.tensor(
            [model.forward_diffusion._sigmas_t[int(t_i)].item() for t_i in t],
            device=predicted_noise.device,
        )
        # Reshape for broadcasting
        sigmas = sigmas.view(-1, 1, 1)
        # Scale predicted noise by 1/sigma_t
        scaled_pred_noise = predicted_noise / sigmas
        mse = (scaled_pred_noise - true_noise) ** 2
    else:
        # Fallback to original calculation if model not provided
        mse = (predicted_noise - true_noise) ** 2

    mse = mse.view(mse.size(0), -1).mean(dim=1)  # Mean over spatial dimensions
    weighted_loss = (weights * mse).mean()
    return weighted_loss.item()


# ==================== Core Reverse Process Analysis ====================
def run_reverse_process(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int = 10,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
) -> DiffusionAnalysisResults:
    """Run diffusion process analysis at specified timesteps.

    This function automates running the diffusion process analysis across multiple timesteps,
    collecting results and metrics for each timestep. It helps analyze how well the model
    performs denoising at different stages of the diffusion process.

    Args:
        model: The diffusion model to analyze
        x0: Clean input tensor of shape [batch_size, channels, seq_length]
        num_samples: Number of samples to analyze per timestep
        timesteps: List of timesteps to analyze. If None, uses a subset of timesteps from tmin to tmax
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping timesteps to their analysis results
    """
    if timesteps is None:
        # Use a subset of timesteps from tmin to tmax
        tmin = model.forward_diffusion.tmin
        tmax = model.forward_diffusion.tmax

        # For diffusion analysis, we typically want fewer timesteps than noise analysis
        # to avoid generating too many plots
        step_size = max(1, (tmax - tmin) // 10)  # Aim for about 10 timesteps
        timesteps = list(range(tmin, tmax + 1, step_size))

        # Always include tmin and tmax
        if tmin not in timesteps:
            timesteps.insert(0, tmin)
        if tmax not in timesteps:
            timesteps.append(tmax)

    # Dictionary to store results for each timestep
    results: DiffusionAnalysisResults = {}

    if verbose:
        print("\n" + "=" * 70)
        print(f" STARTING DIFFUSION ANALYSIS (timesteps: {len(timesteps)}) ")
        print("=" * 70)

    # Run diffusion analysis for each timestep
    for t in timesteps:
        if verbose:
            print(f"\nAnalyzing diffusion process at timestep {t}...")

        # Run reverse process at this timestep
        results[t] = run_reverse_process_at_timestep(
            model, x0, num_samples, t, verbose=verbose
        )

        if verbose:
            print("\n" + "-" * 50 + "\n")

    return results


def run_reverse_process_at_timestep(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int,
    timestep: int,
    verbose: bool = True,
) -> DiffusionAnalysisResult:
    """Test the diffusion process at a specific timestep.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary containing analysis results including:
        - timestep: The analyzed timestep
        - x0: Original clean input
        - xt: Noisy input at timestep t
        - noise: Actual noise added
        - predicted_noise: Model's predicted noise
        - x_t_minus_1: Denoised result from reverse diffusion step
        - metrics: Dictionary of computed statistics
    """
    model.eval()

    with torch.no_grad():
        x0_samples = x0[:num_samples].to(x0.device)
        batch_size = x0_samples.shape[0]
        t_tensor = torch.full(
            (batch_size,), timestep, device=x0_samples.device, dtype=torch.long
        )

        # Sample random noise
        noise = torch.randn_like(x0_samples)

        # Forward diffusion
        xt = model.forward_diffusion.sample(x0_samples, t_tensor, noise)

        # Get model's noise prediction
        predicted_noise = model.predict_added_noise(xt, t_tensor)

        # Reverse diffusion step
        x_t_minus_1 = model.reverse_diffusion.reverse_diffusion_step(xt, t_tensor)

        # Get sigma_t for the current timestep
        sigma_t = model.forward_diffusion._sigmas_t[timestep].item()

        # Scale predicted noise by 1/sigma_t to match true noise distribution
        scaled_pred_noise = predicted_noise / sigma_t

        # Calculate metrics with scaled predicted noise
        noise_mse = F.mse_loss(scaled_pred_noise, noise).item()
        weighted_mse = weighted_mse_loss(predicted_noise, noise, t_tensor, model)
        x0_diff = F.mse_loss(x_t_minus_1, x0_samples).item()
        noise_magnitude = torch.mean(torch.abs(noise)).item()
        pred_noise_magnitude = torch.mean(torch.abs(predicted_noise)).item()
        signal_to_noise = (xt.norm() / (predicted_noise.norm() + 1e-8)).item()

        # Compile metrics
        metrics = {
            "noise_mse": noise_mse,
            "weighted_mse": weighted_mse,
            "x0_diff": x0_diff,
            "sigma_t": sigma_t,
            "noise_magnitude": noise_magnitude,
            "pred_noise_magnitude": pred_noise_magnitude,
            "signal_to_noise": signal_to_noise,
        }

        if verbose:
            print_reverse_statistics(timestep, metrics)

    # Return all computed values
    return {
        "timestep": timestep,
        "x0": x0_samples,
        "xt": xt,
        "noise": noise,
        "predicted_noise": predicted_noise,
        "scaled_pred_noise": scaled_pred_noise,
        "x_t_minus_1": x_t_minus_1,
        "metrics": metrics,
    }


# ==================== Utility Functions ====================
def print_reverse_statistics(timestep: int, metrics: Dict[str, float]) -> None:
    """Print formatted diffusion statistics.

    Args:
        timestep: Current timestep
        metrics: Dictionary of computed statistics
    """
    print(f"\n[Diffusion Statistics at t={timestep}]")
    print(f"- Sigma_t: {metrics['sigma_t']:.6f}")
    print(f"- Noise MSE: {metrics['noise_mse']:.6f}")
    print(f"- Weighted MSE: {metrics['weighted_mse']:.6f}")
    print(f"- Reconstruction MSE (x0 vs x_t-1): {metrics['x0_diff']:.6f}")
    print(f"- Signal-to-Noise Ratio: {metrics['signal_to_noise']:.6f}")
    print(f"- True Noise Magnitude: {metrics['noise_magnitude']:.6f}")
    print(f"- Predicted Noise Magnitude: {metrics['pred_noise_magnitude']:.6f}")


def save_reverse_analysis(
    results: DiffusionAnalysisResults,
    output_dir: Path,
    filename: str = "diffusion_analysis_results.csv",
) -> None:
    """Save diffusion analysis results to a CSV file.

    Args:
        results: Dictionary mapping timesteps to analysis results
        output_dir: Directory to save the CSV file
        filename: Name of the output CSV file
    """
    from typing import Any, Dict, List

    import pandas as pd

    # Prepare data for DataFrame
    rows: List[Dict[str, Any]] = []
    for t, result in results.items():
        row = {"timestep": t}
        row.update(result["metrics"])
        rows.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)

    print(f"Saved diffusion analysis results to {csv_path}")


# ==================== Visualization Functions ====================
def plot_diffusion_results(
    result: DiffusionAnalysisResult, save_dir: Optional[str] = None
) -> None:
    """Plot diffusion process results for a single timestep.

    This function creates visualizations for the diffusion process at a specific timestep
    using the results from diffusion_at_timestep.

    Args:
        result: Results from diffusion_at_timestep
        save_dir: Directory to save plots (if None, plots are displayed but not saved)
    """
    # Extract data from results
    timestep = result["timestep"]
    x0 = result["x0"]
    xt = result["xt"]
    noise = result["noise"]
    predicted_noise = result["predicted_noise"]
    x_t_minus_1 = result["x_t_minus_1"]
    metrics = result["metrics"]

    # Create output directory if saving plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # Plot diffusion process
    plot_diffusion_process(
        x0=x0,
        noise=noise,
        x_t=xt,
        predicted_noise=predicted_noise,
        x_t_minus_1=x_t_minus_1,
        timestep=timestep,
        save_dir=save_dir,
        sigma_t=metrics["sigma_t"],
    )

    # Plot superimposed comparison
    plot_combined_comparison(
        x0=x0, x_t=xt, x_t_minus_1=x_t_minus_1, timestep=timestep, save_dir=save_dir
    )


def plot_diffusion_metrics(
    results: DiffusionAnalysisResults, save_dir: Optional[str] = None
) -> None:
    """Plot diffusion metrics across multiple timesteps.

    Args:
        results: Dictionary mapping timesteps to diffusion analysis results
        save_dir: Directory to save plots (if None, plots are displayed but not saved)
    """
    # Extract timesteps and metrics
    timesteps = sorted(results.keys())
    noise_mse = [results[t]["metrics"]["noise_mse"] for t in timesteps]
    weighted_mse = [results[t]["metrics"]["weighted_mse"] for t in timesteps]
    x0_diff = [results[t]["metrics"]["x0_diff"] for t in timesteps]
    signal_to_noise = [results[t]["metrics"]["signal_to_noise"] for t in timesteps]

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Diffusion Metrics Across Timesteps", fontsize=16)

    # Plot noise MSE
    axs[0, 0].plot(timesteps, noise_mse, "o-", color="blue")
    axs[0, 0].set_title("Noise Prediction MSE")
    axs[0, 0].set_xlabel("Timestep")
    axs[0, 0].set_ylabel("MSE")
    axs[0, 0].grid(True)

    # Plot weighted MSE
    axs[0, 1].plot(timesteps, weighted_mse, "o-", color="green")
    axs[0, 1].set_title("Weighted MSE")
    axs[0, 1].set_xlabel("Timestep")
    axs[0, 1].set_ylabel("Weighted MSE")
    axs[0, 1].grid(True)

    # Plot reconstruction error
    axs[1, 0].plot(timesteps, x0_diff, "o-", color="red")
    axs[1, 0].set_title("Reconstruction Error (x0 vs x_t-1)")
    axs[1, 0].set_xlabel("Timestep")
    axs[1, 0].set_ylabel("MSE")
    axs[1, 0].grid(True)

    # Plot signal-to-noise ratio
    axs[1, 1].plot(timesteps, signal_to_noise, "o-", color="purple")
    axs[1, 1].set_title("Signal-to-Noise Ratio")
    axs[1, 1].set_xlabel("Timestep")
    axs[1, 1].set_ylabel("SNR")
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

    # Save or show plot
    if save_dir:
        save_path = Path(save_dir) / "diffusion_metrics_summary.png"
        plt.savefig(save_path)
        print(f"Saved metrics summary plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def to_numpy(x):
    """Convert tensor to numpy array, handling batch and channel dimensions."""
    return x[0, 0].detach().cpu().numpy()


def plot_combined_comparison(x0, x_t, x_t_minus_1, timestep, save_dir=None):
    """Plot original, noisy, and denoised signals superimposed with consistent vertical scale.

    This visualization makes it easier to see how noise diminishes in the final timesteps.

    Args:
        x0: Original input tensor
        x_t: Noisy input tensor at timestep t
        x_t_minus_1: Denoised output tensor
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    x0_np = to_numpy(x0)
    x_t_np = to_numpy(x_t)
    x_t_minus_1_np = to_numpy(x_t_minus_1)

    # Calculate global min/max for consistent y-axis scale
    all_data = np.concatenate([x0_np[:100], x_t_np[:100], x_t_minus_1_np[:100]])
    y_min, y_max = np.min(all_data), np.max(all_data)
    # Add a small margin
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    plt.figure(figsize=(12, 4))
    plt.title(f"Diffusion Process Comparison (t={timestep})")
    plt.plot(x0_np[:100], label="Original x0", alpha=0.8, color="blue")
    plt.plot(x_t_np[:100], label=f"Noisy x_t", alpha=0.5, color="red", linestyle=":")
    plt.plot(x_t_minus_1_np[:100], label="Denoised x_(t-1)", alpha=0.8, color="green")
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    # Add noise reduction annotation
    noise_reduction = np.mean(np.abs(x_t_np - x0_np)) - np.mean(
        np.abs(x_t_minus_1_np - x0_np)
    )
    plt.annotate(
        f"Noise reduction: {noise_reduction:.4f}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    if save_dir:
        plt.savefig(f"{save_dir}/combined_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def plot_diffusion_process(
    x0, noise, x_t, predicted_noise, x_t_minus_1, timestep, save_dir=None, sigma_t=None
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
        sigma_t: Sigma value at timestep t (for scaling)
    """
    # Create save directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Generate all plots
    plot_signal_comparison(x0, x_t, timestep, save_dir)
    plot_noise_comparison(noise, predicted_noise, timestep, save_dir, sigma_t)
    plot_denoising_comparison(x0, x_t_minus_1, timestep, save_dir)
    plot_combined_comparison(x0, x_t, x_t_minus_1, timestep, save_dir)


def plot_signal_comparison(x0, x_t, timestep, save_dir=None):
    """Plot original vs noisy signal with consistent vertical scale.

    Args:
        x0: Original input tensor [B, C, seq_len]
        x_t: Noisy input tensor at timestep t
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    x0_np = to_numpy(x0)
    x_t_np = to_numpy(x_t)

    # Calculate global min/max for consistent y-axis scale
    all_data = np.concatenate([x0_np[:100], x_t_np[:100]])
    y_min, y_max = np.min(all_data), np.max(all_data)
    # Add a small margin
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    plt.figure(figsize=(12, 4))
    plt.title(f"Original vs Noisy Signal (t={timestep})")
    plt.plot(x0_np[:100], label="Original x0", alpha=0.8)
    plt.plot(x_t_np[:100], label=f"Noisy x_t", alpha=0.8)
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/signal_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def plot_noise_comparison(
    noise, predicted_noise, timestep, save_dir=None, sigma_t=None
):
    """Plot true vs predicted noise.

    Args:
        noise: True noise tensor
        predicted_noise: Model's noise prediction
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
        sigma_t: Sigma value at timestep t (for scaling)
    """
    noise_np = to_numpy(noise)
    pred_noise_np = to_numpy(predicted_noise)

    # Scale predicted noise if sigma_t is provided
    if sigma_t is not None:
        scaled_pred_noise_np = pred_noise_np / sigma_t
    else:
        scaled_pred_noise_np = None

    plt.figure(figsize=(12, 4))
    plt.title(f"True vs Predicted Noise (t={timestep})")
    plt.plot(noise_np[:100], label="True Noise N(0,1)", alpha=0.8)
    plt.plot(pred_noise_np[:100], label="Predicted Noise", alpha=0.8)

    if scaled_pred_noise_np is not None:
        plt.plot(
            scaled_pred_noise_np[:100],
            label="Scaled Predicted Noise (1/σ)",
            alpha=0.8,
            linestyle="--",
        )

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
    """Plot original vs denoised signal with consistent vertical scale.

    Args:
        x0: Original input tensor
        x_t_minus_1: Denoised output tensor
        timestep: Current timestep
        save_dir: Directory to save the plot (if None, show plot)
    """
    x0_np = to_numpy(x0)
    x_t_minus_1_np = to_numpy(x_t_minus_1)

    # Calculate global min/max for consistent y-axis scale
    all_data = np.concatenate([x0_np[:100], x_t_minus_1_np[:100]])
    y_min, y_max = np.min(all_data), np.max(all_data)
    # Add a small margin
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    plt.figure(figsize=(12, 4))
    plt.title(f"Original vs Denoised Signal (t={timestep})")
    plt.plot(x0_np[:100], label="Original x0", alpha=0.8)
    plt.plot(x_t_minus_1_np[:100], label="Denoised x_(t-1)", alpha=0.8)
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.legend()
    plt.ylabel("Signal")
    plt.xlabel("Position")
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/denoised_t{timestep:04d}.png")
        plt.close()
    else:
        plt.show()


def visualize_diffusion_process_heatmap(
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

            # Get sigma_t for scaling
            sigma_t = model.forward_diffusion.sigma(t_tensor).item()

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

            # Calculate global min/max for consistent y-axis scale for signal plots
            all_signal_data = np.concatenate([x0_vis, x_t_vis, x_0_pred_vis])
            y_min_signal, y_max_signal = np.min(all_signal_data), np.max(
                all_signal_data
            )
            # Add a small margin
            y_range_signal = y_max_signal - y_min_signal
            y_min_signal -= 0.1 * y_range_signal
            y_max_signal += 0.1 * y_range_signal

            # Calculate global min/max for consistent y-axis scale for noise plots
            all_noise_data = np.concatenate([eps_vis, pred_eps_vis])
            y_min_noise, y_max_noise = np.min(all_noise_data), np.max(all_noise_data)
            # Add a small margin
            y_range_noise = y_max_noise - y_min_noise
            y_min_noise -= 0.1 * y_range_noise
            y_max_noise += 0.1 * y_range_noise

            # Plot results
            # 1. Original sig
            axes[i, 0].plot(x_axis, x0_vis, "b-", linewidth=1)
            axes[i, 0].set_ylabel(f"t={t}")
            axes[i, 0].set_ylim(y_min_signal, y_max_signal)  # Consistent y-axis

            # 2. Added Noise
            axes[i, 1].plot(x_axis, eps_vis, "r-", linewidth=1)
            axes[i, 1].set_ylim(-4, 4)  # Consistent y-axis for noise

            # 3. Noisy Signal
            axes[i, 2].plot(x_axis, x_t_vis, "g-", linewidth=1)
            axes[i, 2].set_ylim(y_min_signal, y_max_signal)  # Consistent y-axis

            # 4. Predicted Noise
            axes[i, 3].plot(x_axis, pred_eps_vis, "k-", linewidth=1, label="Predicted")

            # Also plot scaled predicted noise
            scaled_pred_eps_vis = pred_eps_vis / sigma_t
            axes[i, 3].plot(
                x_axis,
                scaled_pred_eps_vis,
                "m-",
                linewidth=1,
                alpha=0.7,
                label="Scaled (1/σ)",
            )
            # axes[i, 3].set_ylim(-4, 4)  # Consistent y-axis for noise
            axes[i, 3].legend(fontsize=6)

            # 5. Denoised
            axes[i, 4].plot(x_axis, x_0_pred_vis, "c-", linewidth=1)
            axes[i, 4].set_ylim(y_min_signal, y_max_signal)  # Consistent y-axis

            # Calculate noise reduction metrics for statistics (not displayed in this plot)
            noise_magnitude = np.mean(np.abs(x_t_vis - x0_vis))
            denoised_error = np.mean(np.abs(x_0_pred_vis - x0_vis))
            noise_reduction = noise_magnitude - denoised_error
            noise_reduction_percent = (
                (noise_reduction / noise_magnitude) * 100 if noise_magnitude > 0 else 0
            )

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
                f"Mean: {x_t.mean():.3f}\nStd: {x_t.std():.3f}\nσ_t: {sigma_t:.3f}",
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
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_dir / "diffusion_process_lineplot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    return fig


def visualize_superimposed_comparison(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the superimposed comparison of original, noisy, and denoised signals at different timesteps.

    This function creates a dedicated visualization that superimposes the original, noisy, and denoised signals
    in the same plot for each timestep, making it easier to see how well the model denoises the data.

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization (to avoid overcrowded plots)
    """
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

    # Create a figure with one column per timestep
    fig, axes = plt.subplots(1, len(timesteps), figsize=(5 * len(timesteps), 5))

    # Handle the case of a single timestep
    if len(timesteps) == 1:
        axes = np.array([axes])  # Convert to array for consistent indexing

    # Set figure title
    fig.suptitle(
        "Superimposed Comparison of Original, Noisy, and Denoised Signals", fontsize=14
    )

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # Create timestep tensor
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

            # Generate noise
            eps = torch.randn_like(x0)

            # Forward diffusion: Add noise to get x_t
            x_t = model.forward_diffusion.sample(x0, t_tensor, eps)

            # Get sigma_t for scaling
            sigma_t = model.forward_diffusion.sigma(t_tensor).item()

            # Reverse diffusion: Denoise x_t to get x_0_pred
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)

            # Prepare data for visualization
            x0_vis = x0.cpu().squeeze().numpy()[indices]
            x_t_vis = x_t.cpu().squeeze().numpy()[indices]
            x_0_pred_vis = x_0_pred.cpu().squeeze().numpy()[indices]

            # Create x-axis for plotting
            x_axis = np.arange(len(indices))

            # Calculate global min/max for consistent y-axis scale
            all_signal_data = np.concatenate([x0_vis, x_t_vis, x_0_pred_vis])
            y_min, y_max = np.min(all_signal_data), np.max(all_signal_data)
            # Add a small margin
            y_range = y_max - y_min
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range

            # Plot superimposed comparison
            axes[i].plot(x_axis, x0_vis, "b-", linewidth=1.5, label="Original")
            axes[i].plot(
                x_axis, x_t_vis, "r:", linewidth=1, alpha=0.6, label=f"Noisy (t={t})"
            )
            axes[i].plot(x_axis, x_0_pred_vis, "g-", linewidth=1.5, label="Denoised")
            axes[i].set_ylim(y_min, y_max)  # Consistent y-axis
            axes[i].legend(fontsize=8)
            axes[i].set_title(f"Timestep {t}")

            # Calculate noise reduction metrics
            noise_magnitude = np.mean(np.abs(x_t_vis - x0_vis))
            denoised_error = np.mean(np.abs(x_0_pred_vis - x0_vis))
            noise_reduction = noise_magnitude - denoised_error
            noise_reduction_percent = (
                (noise_reduction / noise_magnitude) * 100 if noise_magnitude > 0 else 0
            )

            # Calculate loss using model's loss_per_timesteps method for the specific timestep
            with torch.enable_grad():  # Temporarily enable gradients for loss computation
                # Use the same noise that was used for visualization
                loss = model.loss_per_timesteps(
                    x0, eps, torch.tensor([t], device=device)
                )[0].item()

            # Add annotation showing noise reduction and loss
            axes[i].annotate(
                f"Noise reduction: {noise_reduction:.4f} ({noise_reduction_percent:.1f}%)\nLoss at t={t}: {loss:.6f}\n\u03c3_t: {sigma_t:.3f}",
                xy=(0.5, 0.02),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=8,
                ha="center",
            )

    plt.tight_layout()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_dir / "superimposed_comparison.png", dpi=300, bbox_inches="tight"
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
