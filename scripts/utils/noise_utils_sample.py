#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Sample-level noise analysis utilities for diffusion models.
Includes functions for per-sample, per-channel, or per-feature noise analysis and visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src import DiffusionModel

# Type aliases
NoiseAnalysisResult = Dict[str, Any]


# ==================== Sample Noise Analysis ====================


# Analyze a single sample's noise trajectory over all timesteps
def analyze_sample_noise_trajectory(
    model: DiffusionModel,
    x0: Tensor,
    sample_idx: int = 0,
    position: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Track true and predicted noise for a single sample's position across timesteps.

    Args:
        model: The diffusion model
        x0: Input tensor of shape [batch, 1, seq_len]
        sample_idx: Index of the sample to analyze
        position: Specific position to analyze. If None, aggregates over all positions
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
            x0_sample = x0[sample_idx : sample_idx + 1].to(x0.device)
            eps = torch.randn_like(x0_sample)
            xt = model.forward_diffusion.sample(x0_sample, t_tensor, eps)
            pred_noise = model.predict_added_noise(xt, t_tensor)

            if position is None:
                # Aggregate over all positions (mean across sequence length)
                true_trajectory.append(eps[0, 0].mean().cpu().numpy())
                pred_trajectory.append(pred_noise[0, 0].mean().cpu().numpy())
            else:
                # Get specific position
                true_trajectory.append(eps[0, 0, position].item())
                pred_trajectory.append(pred_noise[0, 0, position].item())
    return {
        "timesteps": np.array(timesteps),
        "true_noise": np.array(true_trajectory),
        "pred_noise": np.array(pred_trajectory),
    }


def plot_sample_noise_trajectory(
    traj: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: Path,
    position: Optional[int] = None,
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
    if position is not None:
        title += f", Position {position}"

    ax.set_title(f"Noise Trajectory: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "sample_noise_trajectory.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


# Analyze per-position/channel error for a single sample at a timestep
def analyze_sample_error_by_position(
    model: DiffusionModel,
    x0: Tensor,
    sample_idx: int = 0,
    timestep: int = 1000,
) -> Dict[str, np.ndarray]:
    """Compute absolute and squared error for each position in a sample at a given timestep.

    Args:
        model: The diffusion model
        x0: Input tensor of shape [batch, 1, seq_len]
        sample_idx: Index of the sample to analyze
        timestep: Timestep at which to compute the errors

    Returns:
        Dictionary containing:
        - 'position_errors': Absolute errors for each position
        - 'squared_errors': Squared errors for each position
        - 'positions': Position indices
    """
    model.eval()
    with torch.no_grad():
        x0_sample = x0[sample_idx : sample_idx + 1].to(x0.device)
        t_tensor = torch.tensor([timestep], device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0_sample)
        xt = model.forward_diffusion.sample(x0_sample, t_tensor, eps)
        pred_noise = model.predict_added_noise(xt, t_tensor)

        # Compute errors for each position (channel is always 0 for 1D SNP data)
        errors = (pred_noise[0, 0] - eps[0, 0]).abs()  # [seq_len]
        squared_errors = errors**2

        return {
            "position_errors": errors.cpu().numpy(),
            "squared_errors": squared_errors.cpu().numpy(),
            "positions": np.arange(len(errors)),
        }


def plot_sample_error_by_position(
    errors: Dict[str, np.ndarray],
    output_dir: Path,
    sample_idx: int = 0,
    timestep: int = 1000,
):
    """Plot error by position for a single sample at a specific timestep.

    Args:
        errors: Dictionary containing 'position_errors', 'squared_errors', and 'positions'
        save_path: Optional path to save the plot
        sample_idx: Index of the sample being analyzed
        timestep: Timestep at which the errors were computed
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot absolute errors
    ax.plot(errors["positions"], errors["position_errors"], alpha=0.7)
    ax.set_title(
        f"Absolute/Squared Error by Position\nSample {sample_idx}, t={timestep}"
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Absolute Error")
    ax.grid(True, alpha=0.3)

    # Plot squared errors
    ax2 = ax.twinx()
    ax2.plot(errors["positions"], errors["squared_errors"], alpha=0.7, color="orange")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Squared Error")
    ax2.grid(True, alpha=0.3)

    plot_path = output_dir / "sample_error_by_position.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
