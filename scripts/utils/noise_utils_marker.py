#!/usr/bin/env python
# coding: utf-8

"""Utility functions for analyzing SNP-specific behavior in diffusion models.

This module provides functions to track and analyze noise patterns at specific SNP positions
during the forward and reverse diffusion processes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from src import DiffusionModel

# Type aliases
ArrayLike = Union[np.ndarray, List[float], torch.Tensor]
RunResult = Dict[str, Any]
VarianceStats = Dict[str, Any]

# Analyze a single sample's noise trajectory over all timesteps
def analyze_marker_noise_trajectory(
    model: DiffusionModel,
    x0: Tensor,
    sample_idx: int = 0,
    marker_index: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Track true and predicted noise for a single sample's position across timesteps.
    
    Args:
        model: The diffusion model
        x0: Input tensor of shape [batch, 1, seq_len]
        sample_idx: Index of the sample to analyze
        marker_index: Specific marker to analyze. If None, aggregates over all markers
        timesteps: List of timesteps to analyze. If None, uses all timesteps
        
    Returns:
        Dictionary containing 'true_noise' and 'pred_noise' trajectories
    """
    model.eval()

    if timesteps is None:
        timesteps = list(range(1, getattr(model.forward_diffusion, "tmax", 1000) + 1))
    
    true_trajectory = []
    pred_trajectory = []

    with torch.no_grad():
        for t in timesteps:
            t_tensor = torch.tensor([t], device=x0.device, dtype=torch.long)
            x0_sample = x0[sample_idx:sample_idx + 1].to(x0.device)
            eps = torch.randn_like(x0_sample)
            xt = model.forward_diffusion.sample(x0_sample, t_tensor, eps)
            pred_noise = model.predict_added_noise(xt, t_tensor)
            
            if marker_index is None:
                # Aggregate over all positions (mean across sequence length)
                true_trajectory.append(eps[0, 0].mean().cpu().numpy())
                pred_trajectory.append(pred_noise[0, 0].mean().cpu().numpy())
            else:
                # Get specific marker
                true_trajectory.append(eps[0, 0, marker_index].item())
                pred_trajectory.append(pred_noise[0, 0, marker_index].item())
    return {
        "timesteps": np.array(timesteps),
        "true_noise": np.array(true_trajectory),
        "pred_noise": np.array(pred_trajectory),
    }


def plot_marker_noise_trajectory(
    traj: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    marker_index: Optional[int] = None,
):
    """Plot the noise trajectory for a single sample/channel/position across timesteps."""
    timesteps = traj["timesteps"]
    true_noise = traj["true_noise"]
    pred_noise = traj["pred_noise"]
    
    fig, ax = plt.subplots(figsize=(12, 6))

    if true_noise.ndim == 2:  # All positions
        ax.plot(timesteps, true_noise.mean(axis=-1), label="True Noise (mean)")
        ax.plot(timesteps, pred_noise.mean(axis=-1), label="Pred Noise (mean)")
    else:
        ax.plot(timesteps, true_noise, label="True Noise")
        ax.plot(timesteps, pred_noise, label="Pred Noise")
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Noise Value")

    title = f"Sample {sample_idx}"
    if marker_index is not None:
        title += f", Marker {marker_index}"
    
    ax.set_title(f"Noise Trajectory: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "marker_noise_trajectory.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def track_single_run_at_marker(model: DiffusionModel, x0: Tensor, marker_index: int) -> Dict[str, Any]:
    """Track noise at a fixed marker position across all timesteps for a single run.

    Args:
        model: The diffusion model
        x0: Input data [B, C, seq_len]
        marker_index: Index of the marker to monitor

    Returns:
        Dictionary containing timesteps, true_noises, and pred_noises for a single run
    """
    model.eval()
    timesteps = list(range(1, model.forward_diffusion.tmax + 1))
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
            xt = model.forward_diffusion.sample(x0, t_tensor, noise)

            # Get model's noise prediction
            predicted_noise = model.predict_added_noise(xt, t_tensor)

            # Store noise at the specified marker position
            true_noises.append(noise[0, 0, marker_index].item())
            pred_noises.append(predicted_noise[0, 0, marker_index].item())

            # --- Debug: Print standard and weighted MSE at this timestep ---
            # mse = ((predicted_noise - noise) ** 2).mean().item()
            # weighted_mse = weighted_mse_loss(predicted_noise, noise, t_tensor)
            # print(f"Timestep {t:4d}: Standard MSE={mse:.6f} | Weighted MSE={weighted_mse:.6f}")

    return {
        "timesteps": timesteps,
        "true_noises": true_noises,
        "pred_noises": pred_noises,
        "marker_index": marker_index,
    }


def plot_noise_evolution(
    timesteps: List[int],
    true_noises: ArrayLike,
    pred_noises: ArrayLike,
    marker_index: int,
    output_dir: Path,
) -> None:
    """Plot the evolution of noise and its prediction at a fixed marker position.

    Args:
        timesteps: List of timesteps
        true_noises: List of true noise values at the marker position
        pred_noises: List of predicted noise values at the marker position
        marker_index: Index of the marker being monitored
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
    ax1.set_title(f"Noise at Marker {marker_index} Across Timesteps")
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

    plot_path = output_dir / f"marker_{marker_index}_noise_evolution.png"
    fig.tight_layout()
    fig.savefig(plot_path,dpi=300, bbox_inches="tight")
    plt.close()
