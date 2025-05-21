#!/usr/bin/env python
# coding: utf-8

"""Utility functions for analyzing SNP-specific behavior in diffusion models.

This module provides functions to track and analyze noise patterns at specific SNP positions
during the forward and reverse diffusion processes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

# Type aliases
ArrayLike = Union[np.ndarray, List[float], torch.Tensor]
RunResult = Dict[str, Any]
VarianceStats = Dict[str, Any]


def analyze_single_snp(
    model: Any, x0: Tensor, snp_index: int, num_runs: int, output_dir: Path
) -> Dict[str, Any]:
    """Analyze noise at a specific SNP position across multiple runs.

    Args:
        model: The diffusion model
        x0: Clean input data [B, C, L]
        snp_index: Index of the SNP to analyze
        num_runs: Number of runs to perform
        output_dir: Directory to save outputs

    Returns:
        Dictionary with analysis results
    """

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Perform multiple runs
    all_runs = []
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        run_result = track_single_run_at_snp(model, x0, snp_index)
        all_runs.append(run_result)

    # Calculate variance statistics
    variance_stats = calculate_variance_across_runs(all_runs)

    # Plot variance statistics if we have data
    if variance_stats:
        plot_variance_statistics(
            variance_stats, save_path=output_dir / "variance_stats.png"
        )

    return {
        "all_runs": all_runs,
        "variance_stats": variance_stats,
    }


def track_single_run_at_snp(model: Any, x0: Tensor, snp_index: int) -> Dict[str, Any]:
    """Track noise at a fixed SNP position across all timesteps for a single run.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        snp_index: Index of the SNP to monitor

    Returns:
        Dictionary containing timesteps, true_noises, and pred_noises for a single run
    """
    model.eval()
    timesteps = list(range(1, model.ddpm.tmax + 1))
    true_noises = []
    pred_noises = []

    with torch.no_grad():
        for t in timesteps:
            # Create timestep tensor
            batch_size = x0.shape[0]
            t_tensor = torch.full((batch_size,), t, device=x0.device, dtype=torch.long)

            # Sample random noise
            noise = torch.randn_like(x0)

            # Forward diffusion
            xt = model.ddpm.sample(x0, t_tensor, noise)

            # Get model's noise prediction
            predicted_noise = model.predict_added_noise(xt, t_tensor)

            # Store noise at the specified SNP position
            true_noises.append(noise[0, 0, snp_index].item())
            pred_noises.append(predicted_noise[0, 0, snp_index].item())

            # --- Debug: Print standard and weighted MSE at this timestep ---
            # mse = ((predicted_noise - noise) ** 2).mean().item()
            # weighted_mse = weighted_mse_loss(predicted_noise, noise, t_tensor)
            # print(f"Timestep {t:4d}: Standard MSE={mse:.6f} | Weighted MSE={weighted_mse:.6f}")

    return {
        "timesteps": timesteps,
        "true_noises": true_noises,
        "pred_noises": pred_noises,
        "snp_index": snp_index,
    }


def calculate_variance_across_runs(
    run_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Calculate variance statistics across multiple runs.

    Args:
        run_results: List of dictionaries, each containing results from track_single_run_at_snp

    Returns:
        Dictionary containing variance statistics
    """
    if not run_results or len(run_results) == 1:
        return None

    # Stack all runs
    all_true = np.array([r["true_noises"] for r in run_results])
    all_pred = np.array([r["pred_noises"] for r in run_results])

    # Calculate statistics
    true_var = np.var(all_true, axis=0)
    pred_var = np.var(all_pred, axis=0)
    mse = np.mean((all_true - all_pred) ** 2, axis=0)

    return {
        "timesteps": run_results[0]["timesteps"],
        "true_variance": true_var,
        "pred_variance": pred_var,
        "mse": mse,
        "snp_index": run_results[0]["snp_index"],
        "num_runs": len(run_results),
    }


def plot_variance_statistics(
    variance_stats: Optional[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot variance statistics across runs.

    Args:
        variance_stats: Output from calculate_variance_across_runs
        save_path: Optional path to save the plot
    """
    if variance_stats is None:
        print("No variance statistics to plot (need multiple runs).")
        return

    plt.figure(figsize=(12, 8))

    # Plot variances
    plt.subplot(2, 1, 1)
    plt.plot(
        variance_stats["timesteps"],
        variance_stats["true_variance"],
        "b-",
        label="True Noise Variance",
    )
    plt.plot(
        variance_stats["timesteps"],
        variance_stats["pred_variance"],
        "r--",
        label="Predicted Noise Variance",
    )
    plt.title(
        f'Noise Variance at SNP {variance_stats["snp_index"]} Across {variance_stats["num_runs"]} Runs'
    )
    plt.xlabel("Timestep")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot MSE
    plt.subplot(2, 1, 2)
    plt.plot(variance_stats["timesteps"], variance_stats["mse"], "g-", label="MSE")
    plt.title("Mean Squared Error Between True and Predicted Noise")
    plt.xlabel("Timestep")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_noise_evolution(
    timesteps: List[int],
    true_noises: ArrayLike,
    pred_noises: ArrayLike,
    snp_index: int,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot the evolution of noise and its prediction at a fixed SNP position.

    Args:
        timesteps: List of timesteps
        true_noises: List of true noise values at the SNP position
        pred_noises: List of predicted noise values at the SNP position
        snp_index: Index of the SNP being monitored
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(15, 12))

    # Convert to numpy arrays for easier manipulation
    true_noises = np.array(true_noises)
    pred_noises = np.array(pred_noises)

    # Calculate cumulative noise (running sum of absolute noise)
    cumul_true = np.cumsum(np.abs(true_noises))
    cumul_pred = np.cumsum(np.abs(pred_noises))

    # 1. Raw Noise Comparison
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(timesteps, true_noises, "b-", label="True Noise", alpha=0.7)
    ax1.plot(timesteps, pred_noises, "r--", label="Predicted Noise", alpha=0.7)
    ax1.set_title(f"Noise at SNP {snp_index} Across Timesteps")
    ax1.set_ylabel("Noise Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Absolute Noise Magnitude
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(timesteps, np.abs(true_noises), "b-", label="|True Noise|", alpha=0.7)
    ax2.plot(
        timesteps, np.abs(pred_noises), "r--", label="|Predicted Noise|", alpha=0.7
    )

    # Add moving average (window=5% of total timesteps)
    window_size = max(1, len(timesteps) // 20)
    if window_size > 1:
        true_ma = np.convolve(
            np.abs(true_noises), np.ones(window_size) / window_size, mode="same"
        )
        pred_ma = np.convolve(
            np.abs(pred_noises), np.ones(window_size) / window_size, mode="same"
        )
        ax2.plot(timesteps, true_ma, "b-", linewidth=2, label="True Noise (MA)")
        ax2.plot(timesteps, pred_ma, "r--", linewidth=2, label="Predicted Noise (MA)")

    ax2.set_title("Absolute Noise Magnitude with Moving Average")
    ax2.set_ylabel("Absolute Noise")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative Noise
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(timesteps, cumul_true, "b-", label="Cumulative |True Noise|")
    ax3.plot(timesteps, cumul_pred, "r--", label="Cumulative |Predicted Noise|")
    ax3.set_title("Cumulative Absolute Noise")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Cumulative Noise")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add a vertical line at t=1 and t=T for reference
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=1, color="g", linestyle=":", alpha=0.5, label="t=1 (Min Noise)")
        # ax.axvline(x=len(timesteps), color='r', linestyle=':', alpha=0.5, label=f't={len(timesteps)} (Max Noise)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()
