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
ReverseDiffusionResult = Dict[str, Any]
ReverseDiffusionResults = Dict[int, ReverseDiffusionResult]
TimestepDict = Dict[str, List[int]]


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
def generate_timesteps(tmin: int, tmax: int) -> TimestepDict:
    """
    Generate different sets of timesteps for analysis.

    Args:
        tmin: Minimum timestep (usually 1)
        tmax: Maximum timestep (usually 1000)

    Returns:
        Dictionary with different timestep sets
    """

    # Generate linearly spaced timesteps
    linear_steps = np.linspace(tmin, tmax, num=20, dtype=int).tolist()

    # Generate logarithmically spaced timesteps for better coverage
    log_steps = np.unique(
        np.logspace(np.log10(tmin), np.log10(tmax), num=20, dtype=int)
    ).tolist()

    # Enhanced boundary timesteps with more points at beginning, middle and end
    # Include more steps at the beginning to see the progression from pure signal
    early_steps = [
        tmin,
        tmin + 1,
        tmin + 2,
        tmin + 3,
        tmin + 10,
        tmin + 20,
        tmin + 50,
        tmin + 100,
        tmin + 200,
    ]

    # Include steps in the middle to see the transition
    middle_steps = [tmax // 4, tmax // 3, tmax // 2, 2 * tmax // 3, 3 * tmax // 4]

    # Include more steps at the end to see the progression to pure noise
    late_steps = [
        tmax - 200,
        tmax - 100,
        tmax - 50,
        tmax - 20,
        tmax - 10,
        tmax - 3,
        tmax - 2,
        tmax - 1,
        tmax,
    ]

    # Combine all steps and remove duplicates
    boundary_steps = sorted(list(set(early_steps + middle_steps + late_steps)))

    return {"log": log_steps, "linear": linear_steps, "boundary": boundary_steps}


def run_reverse_process(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int = 10,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
) -> ReverseDiffusionResults:
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
    results: ReverseDiffusionResults = {}

    if verbose:
        print("\n" + "=" * 70)
        print(f" STARTING DIFFUSION ANALYSIS (timesteps: {len(timesteps)}) ")
        print("=" * 70)

    # Run diffusion analysis for each timestep
    for t in timesteps:
        if verbose:
            print(f"\nAnalyzing diffusion process at timestep {t}...")

        # Run reverse process at this timestep
        results[t] = run_reverse_process_at_timestep(model, x0, num_samples, t)

        if verbose:
            print("\n" + "-" * 50 + "\n")

    return results


def run_reverse_process_at_timestep(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int,
    timestep: int,
) -> ReverseDiffusionResults:
    """Test the diffusion process at a specific timestep.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        timestep: Timestep to test at

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

        # Get sigma_t for the current timestep using the proper method
        sigma_t = model.forward_diffusion.sigma(t_tensor)[0].item()

        # Scale predicted noise by 1/sigma_t to match true noise distribution
        # This ensures consistency with how noise is scaled in the forward process
        scaled_pred_noise = predicted_noise / sigma_t

        # Calculate metrics with scaled predicted noise
        noise_mse = F.mse_loss(scaled_pred_noise, noise).item()
        weighted_mse = weighted_mse_loss(predicted_noise, noise, t_tensor, model)
        x0_diff = F.mse_loss(x_t_minus_1, x0_samples).item()
        noise_magnitude = torch.mean(torch.abs(noise)).item()
        pred_noise_magnitude = torch.mean(torch.abs(predicted_noise)).item()

        # Calculate SNR using the same formula as in forward diffusion
        # SNR = |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
        alpha_bar_t = model.forward_diffusion.alpha_bar(t_tensor)[0].item()
        signal_power = torch.mean(
            (torch.sqrt(torch.tensor(alpha_bar_t)) * x0_samples) ** 2
        ).item()
        noise_power = torch.mean(
            (torch.sqrt(torch.tensor(1 - alpha_bar_t)) * noise) ** 2
        ).item()
        signal_to_noise = signal_power / (noise_power + 1e-8)

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
def print_reverse_statistics(
    results: ReverseDiffusionResults, timesteps: Optional[List[int]] = None
) -> None:
    """Print formatted diffusion statistics.

    Args:
        timestep: Current timestep
        metrics: Dictionary of computed statistics
    """
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to those available in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    for t in timesteps:
        r = results[t]
        print(f"\nTimestep t = {t}:")
        print(f"- Sigma_t: {r['metrics']['sigma_t']:.8f}")
        print(f"- Noise MSE: {r['metrics']['noise_mse']:.8f}")
        print(f"- Weighted MSE: {r['metrics']['weighted_mse']:.8f}")
        print(f"- Reconstruction MSE (x0 vs x_t-1): {r['metrics']['x0_diff']:.8f}")
        print(f"- Signal-to-Noise Ratio: {r['metrics']['signal_to_noise']:.8f}")
        print(f"- True Noise Magnitude: {r['metrics']['noise_magnitude']:.8f}")
        print(
            f"- Predicted Noise Magnitude: {r['metrics']['pred_noise_magnitude']:.8f}"
        )


# ==================== Evolution of Denoising ====================
def calculate_diffusion_data(model, batch, timesteps=[100, 500, 900], device=None):
    """Calculate diffusion process data for visualization.

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize (1-indexed, as expected by model)
        device: Device to run calculations on

    Returns:
        Dictionary containing calculated data for each timestep:
        {
            timestep: {
                'x0': Original sample,
                'eps': Original noise,
                'x_t': Noisy sample,
                'pred_eps': Predicted noise,
                'x_0_pred': Denoised sample,
                'sigma_t': Sigma at timestep t,
                'noise_reduction': Noise reduction metric,
                'noise_reduction_percent': Noise reduction percentage,
                'loss': Loss at timestep t
            }
        }
    """
    if device is None:
        device = next(model.parameters()).device

    # Move batch to device and ensure correct shape
    x0 = batch.to(device)
    if x0.dim() == 2:
        x0 = x0.unsqueeze(1)  # Add channel dimension if needed

    # Use only the first sample for visualization if it's a batch
    if x0.size(0) > 1:
        x0 = x0[0:1]  # Shape: [1, C, seq_len]

    results = {}
    model.eval()  # Ensure model is in eval mode

    with torch.no_grad():
        for t in timesteps:
            # Important: t is 1-indexed (1 to 1000)
            # For beta and alpha parameters, use t-1 (0 to 999)
            # For alpha_bar and sigma parameters, use t directly
            t_tensor = torch.tensor(
                [t], device=device
            )  # Keep 1-indexed for parameter access
            t_idx_tensor = torch.tensor(
                [t - 1], device=device
            )  # 0-indexed for model operations

            # Expand tensors to match batch size if needed
            if x0.size(0) > 1:
                t_tensor = t_tensor.expand(x0.size(0))
                t_idx_tensor = t_idx_tensor.expand(x0.size(0))

            # Generate random noise
            eps = torch.randn_like(x0, device=device)

            # Forward process: add noise to create x_t
            # Use t_idx_tensor (0-indexed) for model operations
            x_t = model.forward_diffusion.sample(x0, t_idx_tensor, eps)

            # Get sigma_t for this timestep
            # Use t_tensor (1-indexed) for parameter access
            sigma_t = model.forward_diffusion.sigma(t_tensor).item()

            # Predict noise and calculate denoised sample
            # Use t_idx_tensor (0-indexed) for model operations
            pred_eps = model.predict_added_noise(x_t, t_idx_tensor)
            x_0_pred = model.reverse_diffusion.reverse_diffusion_step(x_t, t_idx_tensor)

            # Calculate noise reduction metrics
            initial_noise = torch.mean(torch.abs(x_t - x0)).item()
            final_noise = torch.mean(torch.abs(x_0_pred - x0)).item()
            noise_reduction = initial_noise - final_noise
            noise_reduction_percent = (
                (noise_reduction / initial_noise * 100) if initial_noise > 0 else 0
            )

            # Calculate loss
            # Use t_idx_tensor (0-indexed) for model operations
            loss = model.loss_per_timesteps(x0, eps, t_idx_tensor)[0].item()

            # Store results
            results[t] = {
                "x0": x0,
                "eps": eps,
                "x_t": x_t,
                "pred_eps": pred_eps,
                "x_0_pred": x_0_pred,
                "sigma_t": sigma_t,
                "noise_reduction": noise_reduction,
                "noise_reduction_percent": noise_reduction_percent,
                "loss": loss,
            }

    return results


def visualize_diffusion_process_heatmap(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the forward and reverse diffusion process at different timesteps using heatmaps.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise (both raw and scaled)
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize (1-indexed, as expected by model)
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization
    """
    # Create figure
    n_rows = len(timesteps)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Set column titles
    titles = [
        "Original x₀",
        "Added Noise ε",
        "Noisy x_t",
        "Predicted Noise",
        "Denoised x_{t-1}",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)

    # Plot data for each timestep
    for i, t in enumerate(timesteps):
        # Set seed for reproducible results across different visualization functions
        torch.manual_seed(42 + t)
        # Run reverse process at this timestep
        result = run_reverse_process_at_timestep(
            model, batch, num_samples=1, timestep=t
        )

        # Extract data
        x0_vis = to_numpy(result["x0"])[:sample_points]
        eps_vis = to_numpy(result["noise"])[:sample_points]
        x_t_vis = to_numpy(result["xt"])[:sample_points]
        pred_eps_vis = to_numpy(result["predicted_noise"])[:sample_points]
        scaled_pred_eps_vis = to_numpy(result["scaled_pred_noise"])[:sample_points]
        x_t_minus_1_vis = to_numpy(result["x_t_minus_1"])[:sample_points]

        # Plot heatmaps
        axes[i, 0].imshow(x0_vis.reshape(1, -1), aspect="auto", cmap="viridis")
        axes[i, 1].imshow(eps_vis.reshape(1, -1), aspect="auto", cmap="viridis")
        axes[i, 2].imshow(x_t_vis.reshape(1, -1), aspect="auto", cmap="viridis")

        # Combined predicted noise heatmap (show both raw and scaled)
        combined_pred = np.vstack(
            [pred_eps_vis.reshape(1, -1), scaled_pred_eps_vis.reshape(1, -1)]
        )
        axes[i, 3].imshow(combined_pred, aspect="auto", cmap="viridis")
        axes[i, 3].set_yticks([0, 1])
        axes[i, 3].set_yticklabels(["Raw", "Scaled"], fontsize=8)

        axes[i, 4].imshow(x_t_minus_1_vis.reshape(1, -1), aspect="auto", cmap="viridis")

        # Add statistics
        for j, (ax, arr) in enumerate(
            zip(axes[i, :4], [x0_vis, eps_vis, x_t_vis, pred_eps_vis])
        ):
            ax.text(
                0.5,
                -0.15,
                f"Mean: {arr.mean():.3f}\nStd: {arr.std():.3f}",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
            if j == 0:
                ax.set_ylabel(f"t={t}")

        # Add statistics for denoised sample
        axes[i, 4].text(
            0.5,
            -0.15,
            f"Mean: {x_t_minus_1_vis.mean():.3f}\nStd: {x_t_minus_1_vis.std():.3f}",
            transform=axes[i, 4].transAxes,
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diffusion_heatmap.png")
        plt.close()
    else:
        plt.show()


def visualize_diffusion_process_lineplot(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the forward and reverse diffusion process at different timesteps using line plots.

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise (both raw and scaled)
    5. Denoised sample (reverse diffusion)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize (1-indexed, as expected by model)
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization
    """
    # Create figure
    n_rows = len(timesteps)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Set column titles
    titles = [
        "Original x₀",
        "Added Noise ε",
        "Noisy x_t",
        "Predicted Noise",
        "Denoised x_{t-1}",
    ]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)

    # Plot data for each timestep
    for i, t in enumerate(timesteps):
        # Set seed for reproducible results across different visualization functions
        torch.manual_seed(42 + t)
        # Run reverse process at this timestep
        result = run_reverse_process_at_timestep(
            model, batch, num_samples=1, timestep=t
        )

        # Extract data
        x0_vis = to_numpy(result["x0"])[:sample_points]
        eps_vis = to_numpy(result["noise"])[:sample_points]
        x_t_vis = to_numpy(result["xt"])[:sample_points]
        pred_eps_vis = to_numpy(result["predicted_noise"])[:sample_points]
        scaled_pred_eps_vis = to_numpy(result["scaled_pred_noise"])[:sample_points]
        x_t_minus_1_vis = to_numpy(result["x_t_minus_1"])[:sample_points]

        x_axis = np.arange(len(x0_vis))

        # 1. Original
        axes[i, 0].plot(x_axis, x0_vis, "b-", linewidth=1)
        axes[i, 0].set_ylim(-3, 3)

        # 2. Added Noise
        axes[i, 1].plot(x_axis, eps_vis, "r-", linewidth=1)
        axes[i, 1].set_ylim(-3, 3)

        # 3. Noisy Signal
        axes[i, 2].plot(x_axis, x_t_vis, "g-", linewidth=1)
        axes[i, 2].set_ylim(-3, 3)

        # 4. Predicted Noise (both raw and scaled)
        axes[i, 3].plot(x_axis, pred_eps_vis, "k-", linewidth=1, label="Raw", alpha=0.8)
        axes[i, 3].plot(
            x_axis, scaled_pred_eps_vis, "m-", linewidth=1, label="Scaled", alpha=0.8
        )
        if i == 0:
            axes[i, 3].legend(fontsize=8)
        axes[i, 3].set_ylim(-3, 3)

        # 5. Denoised Sample
        axes[i, 4].plot(x_axis, x_t_minus_1_vis, "c-", linewidth=1)
        # Special case for t=1000: set wider y-axis limits for denoised sample
        if t == 1000:
            axes[i, 4].set_ylim(-80, 80)
        # axes[i, 4].set_ylim(-3, 3)

        # Add timestep label
        axes[i, 0].set_ylabel(f"t={t}")

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diffusion_lineplot.png")
        plt.close()
    else:
        plt.show()


def visualize_diffusion_process_superimposed(
    model, batch, timesteps=[100, 500, 900], output_dir=None, sample_points=200
):
    """Visualize the superimposed comparison of original, noisy, and denoised signals at different timesteps.

    For each timestep (as columns), shows superimposed plots of:
    - Original sample (blue)
    - Noisy sample (red)
    - Denoised sample (green)

    Args:
        model: The diffusion model
        batch: Batch of real data samples
        timesteps: List of timesteps to visualize (1-indexed, as expected by model)
        output_dir: Directory to save visualizations
        sample_points: Number of points to sample for visualization
    """
    # Create figure - timesteps as columns
    n_cols = len(timesteps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Plot data for each timestep
    for i, t in enumerate(timesteps):
        # Set seed for reproducible results across different visualization functions
        torch.manual_seed(42 + t)
        # Run reverse process at this timestep
        result = run_reverse_process_at_timestep(
            model, batch, num_samples=1, timestep=t
        )

        # Extract data
        x0_vis = to_numpy(result["x0"])[:sample_points]
        x_t_vis = to_numpy(result["xt"])[:sample_points]
        x_t_minus_1_vis = to_numpy(result["x_t_minus_1"])[:sample_points]

        x_axis = np.arange(len(x0_vis))

        # Superimposed plot: Original, Noisy, Denoised
        axes[i].plot(x_axis, x0_vis, "b-", linewidth=2, label="Original", alpha=0.8)
        axes[i].plot(x_axis, x_t_vis, "r-", linewidth=2, label="Noisy", alpha=0.7)
        axes[i].plot(
            x_axis, x_t_minus_1_vis, "g-", linewidth=2, label="Denoised", alpha=0.7
        )

        axes[i].set_title(f"t={t}")
        axes[i].legend(fontsize=10)
        # Special case for t=1000: set wider y-axis limits for denoised sample
        if t == 1000:
            axes[i].set_ylim(-80, 80)
        # axes[i].set_ylim(-3, 3)
        axes[i].grid(True, alpha=0.3)

        # Calculate and add metrics annotation
        initial_noise = torch.mean(torch.abs(result["xt"] - result["x0"])).item()
        final_noise = torch.mean(torch.abs(result["x_t_minus_1"] - result["x0"])).item()
        noise_reduction = initial_noise - final_noise
        noise_reduction_percent = (
            (noise_reduction / initial_noise * 100) if initial_noise > 0 else 0
        )

        axes[i].annotate(
            f"Noise reduction:\n{noise_reduction:.3f} ({noise_reduction_percent:.1f}%)",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            fontsize=8,
            ha="left",
            va="top",
        )

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diffusion_superimposed.png")
        plt.close()
    else:
        plt.show()


# ==================== Visualization Functions ====================
def plot_diffusion_results(
    result: ReverseDiffusionResult, save_dir: Optional[str] = None
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
    results: ReverseDiffusionResults, save_dir: Optional[str] = None
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


def display_diffusion_parameters(model, timestep):
    """Display diffusion process parameters for a specific timestep.

    Args:
        model: The diffusion model
        timestep: Timestep to display parameters for
    """
    # Get parameters using proper methods
    t_tensor = torch.tensor([timestep], device=model.device)
    beta_t = model.forward_diffusion.beta(t_tensor).item()
    alpha_t = model.forward_diffusion.alpha(t_tensor).item()
    alpha_bar_t = model.forward_diffusion.alpha_bar(t_tensor).item()
    sigma_t = model.forward_diffusion.sigma(t_tensor).item()

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
