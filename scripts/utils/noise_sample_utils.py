"""
Sample-level noise analysis utilities for diffusion models.
Includes functions for per-sample, per-channel, or per-feature noise analysis and visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

# Type aliases
NoiseAnalysisResult = Dict[str, Any]


# ==================== Sample Noise Analysis ====================
def noise_prediction_at_timestep(
    model: Any,
    x0: Tensor,
    timestep: int,
    num_samples: int = 3,
    num_positions: int = 5,
    verbose: bool = True,
) -> NoiseAnalysisResult:
    """Analyze noise prediction at a specific timestep for individual samples."""
    model.eval()
    with torch.no_grad():
        x0_samples = x0[:num_samples].to(x0.device)
        batch_size = x0_samples.size(0)
        t_tensor = torch.full(
            (batch_size,), timestep, device=x0.device, dtype=torch.long
        )
        true_noise = torch.randn_like(x0_samples)
        xt = model.forward_diffusion.sample(x0_samples, t_tensor, true_noise)
        pred_noise = model.predict_added_noise(xt, t_tensor)
        mse_loss = (
            model.loss_per_timesteps(x0_samples, true_noise, t_tensor).mean().item()
        )
        abs_errors = (pred_noise - true_noise).abs()
        stats = {
            "true_noise_mean": true_noise.mean().item(),
            "true_noise_std": true_noise.std().item(),
            "pred_noise_mean": pred_noise.mean().item(),
            "pred_noise_std": pred_noise.std().item(),
            "mse": mse_loss,
            "mae": abs_errors.mean().item(),
            "max_ae": abs_errors.max().item(),
            "signal_to_noise": (xt.norm() / (pred_noise.norm() + 1e-8)).item(),
        }
        if verbose:
            _print_noise_statistics(timestep, stats)
    return {
        "timestep": timestep,
        "x0": x0_samples,
        "xt": xt,
        "true_noise": true_noise,
        "pred_noise": pred_noise,
        "stats": stats,
    }


def _print_noise_statistics(timestep: int, stats: Dict[str, float]) -> None:
    print(f"\n[Noise Analysis @ t={timestep}]")
    print(
        f"True noise   : μ = {stats['true_noise_mean']:8.4f} ± {stats['true_noise_std']:8.4f}"
    )
    print(
        f"Pred noise  : μ = {stats['pred_noise_mean']:8.4f} ± {stats['pred_noise_std']:8.4f}"
    )
    print("\n[Prediction Errors]")
    print(f"MSE           : {stats['mse']:.6f}")
    print(f"MAE           : {stats['mae']:.6f}")
    print(f"Max AE        : {stats['max_ae']:.6f}")
    print(f"Signal/Noise  : {stats['signal_to_noise']:.6f}")


# --- NEW: Analyze a single sample's noise trajectory over all timesteps ---
def analyze_sample_noise_trajectory(
    model: Any,
    x0: Tensor,
    sample_idx: int = 0,
    channel: int = 0,
    position: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Track true and predicted noise for a single sample (and channel/position) across timesteps."""
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
                # Aggregate over all positions
                true_trajectory.append(eps[0, channel].cpu().numpy())
                pred_trajectory.append(pred_noise[0, channel].cpu().numpy())
            else:
                true_trajectory.append(eps[0, channel, position].item())
                pred_trajectory.append(pred_noise[0, channel, position].item())
    return {
        "timesteps": np.array(timesteps),
        "true_noise": np.array(true_trajectory),
        "pred_noise": np.array(pred_trajectory),
    }


def plot_sample_noise_trajectory(
    traj: Dict[str, np.ndarray],
    sample_idx: int = 0,
    channel: int = 0,
    position: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot the noise trajectory for a single sample/channel/position across timesteps."""
    timesteps = traj["timesteps"]
    true_noise = traj["true_noise"]
    pred_noise = traj["pred_noise"]
    plt.figure(figsize=(12, 6))
    if true_noise.ndim == 2:  # All positions
        plt.plot(timesteps, true_noise.mean(axis=-1), label="True Noise (mean)")
        plt.plot(timesteps, pred_noise.mean(axis=-1), label="Pred Noise (mean)")
    else:
        plt.plot(timesteps, true_noise, label="True Noise")
        plt.plot(timesteps, pred_noise, label="Pred Noise")
    plt.xlabel("Timestep")
    plt.ylabel("Noise Value")
    title = f"Sample {sample_idx}, Channel {channel}"
    if position is not None:
        title += f", Position {position}"
    plt.title(f"Noise Trajectory: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# --- NEW: Analyze per-position/channel error for a single sample at a timestep ---
def analyze_sample_error_by_position(
    model: Any,
    x0: Tensor,
    sample_idx: int = 0,
    timestep: int = 1000,
    channel: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute abs and squared error for each position in a sample/channel at a given timestep."""
    model.eval()
    with torch.no_grad():
        x0_sample = x0[sample_idx : sample_idx + 1].to(x0.device)
        t_tensor = torch.tensor([timestep], device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0_sample)
        xt = model.forward_diffusion.sample(x0_sample, t_tensor, eps)
        pred_noise = model.predict_added_noise(xt, t_tensor)
        abs_error = (pred_noise[0, channel] - eps[0, channel]).abs().cpu().numpy()
        sq_error = (pred_noise[0, channel] - eps[0, channel]).pow(2).cpu().numpy()
    return {"abs_error": abs_error, "sq_error": sq_error}


def plot_sample_error_by_position(
    errors: Dict[str, np.ndarray],
    channel: int = 0,
    timestep: int = 1000,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot error (abs/squared) across positions for a single sample and channel at a timestep."""
    abs_error = errors["abs_error"]
    sq_error = errors["sq_error"]
    plt.figure(figsize=(12, 5))
    plt.plot(abs_error, label="Absolute Error")
    plt.plot(sq_error, label="Squared Error")
    plt.xlabel("Position")
    plt.ylabel("Error")
    plt.title(f"Sample Error by Position (Channel {channel}, Timestep {timestep})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()
