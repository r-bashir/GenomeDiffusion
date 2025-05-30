"""
Batch-level noise analysis utilities for diffusion models.
Includes functions for running and visualizing aggregate noise analysis across batches.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src import DiffusionModel

# Type aliases
NoiseAnalysisResult = Dict[str, Any]
NoiseAnalysisResults = Dict[int, NoiseAnalysisResult]


# ==================== Core Noise Analysis ====================
def run_noise_analysis(
    model: Any,
    x0: Tensor,
    num_samples: int = 3,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> NoiseAnalysisResults:
    """Run detailed noise analysis at specified timesteps."""
    if timesteps is None:
        timesteps = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results: NoiseAnalysisResults = {}
    if verbose:
        print("\n" + "=" * 70)
        print(" STARTING DETAILED NOISE ANALYSIS ")
        print("=" * 70)
    for t in timesteps:
        results[t] = noise_prediction_at_timestep(
            model=model,
            x0=x0[:num_samples],
            timestep=t,
            num_samples=num_samples,
            verbose=verbose,
        )
        if verbose:
            print("\n" + "-" * 50 + "\n")
    return results


def noise_prediction_at_timestep(
    model: DiffusionModel,
    x0: Tensor,
    timestep: int,
    num_samples: int = 3,
    num_positions: int = 5,
    verbose: bool = True,
) -> NoiseAnalysisResult:
    """Analyze noise prediction at a specific timestep.

    Args:
        model: The diffusion model
        x0: Clean input tensor of shape [batch_size, channels, seq_length]
        timestep: Timestep to analyze
        num_samples: Number of samples to analyze
        num_positions: Number of positions to show in detailed output
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary containing analysis results including:
        - timestep: The analyzed timestep
        - x0: Original clean input
        - xt: Noisy input at timestep t
        - true_noise: Actual noise added
        - pred_noise: Model's predicted noise
        - stats: Dictionary of computed statistics
    """
    model.eval()

    with torch.no_grad():
        # Prepare inputs [num_samples, C, L]
        x0_samples = x0[:num_samples].to(x0.device)
        batch_size = x0_samples.size(0)
        t_tensor = torch.full(
            (batch_size,), timestep, device=x0.device, dtype=torch.long
        )

        # Sample noise and apply forward diffusion
        true_noise = torch.randn_like(x0_samples)
        xt = model.forward_diffusion.sample(x0_samples, t_tensor, true_noise)
        pred_noise = model.predict_added_noise(xt, t_tensor)

        # Use model's own loss function for this timestep
        mse_loss = (
            model.loss_per_timesteps(x0_samples, true_noise, t_tensor).mean().item()
        )

        # Calculate errors and statistics
        abs_errors = (pred_noise - true_noise).abs()
        mse_errors = (pred_noise - true_noise).pow(2)

        stats = {
            "true_noise_mean": true_noise.mean().item(),
            "true_noise_std": true_noise.std().item(),
            "pred_noise_mean": pred_noise.mean().item(),
            "pred_noise_std": pred_noise.std().item(),
            "mse": mse_loss,  # Use model's loss
            "mae": abs_errors.mean().item(),
            "max_ae": abs_errors.max().item(),
            "signal_to_noise": (xt.norm() / (pred_noise.norm() + 1e-8)).item(),
        }

        if verbose:
            _print_noise_statistics(timestep, stats)

        # Print true_noise and pred_noise for t=1000 for debugging sign flip
        if timestep == 1000:
            print("\nDetailed noise vectors at t=1000 (showing first sample):")
            print("true_noise[0]:", true_noise[0, 0, :10].cpu().numpy())
            print("pred_noise[0]:", pred_noise[0, 0, :10].cpu().numpy())
            # Compute and print correlation
            tn = true_noise[0].flatten().cpu().numpy()
            pn = pred_noise[0].flatten().cpu().numpy()
            corr = np.corrcoef(tn, pn)[0, 1]
            print(f"Pearson correlation (t=1000, first sample): {corr:.4f}")

    return {
        "timestep": timestep,
        "x0": x0_samples,
        "xt": xt,
        "true_noise": true_noise,
        "pred_noise": pred_noise,
        "stats": stats,
    }


# ==================== Utility Functions ====================
def _print_noise_statistics(timestep: int, stats: Dict[str, float]) -> None:
    """Print formatted noise statistics.

    Args:
        timestep: Current timestep
        stats: Dictionary of computed statistics
    """
    print(f"\n[Noise Statistics at t={timestep}]")
    print(
        f"True noise  : μ = {stats['true_noise_mean']:8.4f} ± {stats['true_noise_std']:8.4f}"
    )
    print(
        f"Pred noise  : μ = {stats['pred_noise_mean']:8.4f} ± {stats['pred_noise_std']:8.4f}"
    )
    print("\n[Prediction Errors]")
    print(f"MSE           : {stats['mse']:.6f}")
    print(f"MAE           : {stats['mae']:.6f}")
    print(f"Max AE        : {stats['max_ae']:.6f}")
    print(f"Signal/Noise  : {stats['signal_to_noise']:.2f}")


def save_noise_analysis(
    results: NoiseAnalysisResults,
    output_dir: Path,
    filename: str = "noise_analysis.csv",
):
    import pandas as pd

    records = []
    for t, res in results.items():
        row = {"timestep": t}
        row.update(res["stats"])
        records.append(row)
    df = pd.DataFrame(records)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False, float_format="%.6f")


# ==================== Plotting Functions ====================
def plot_loss_vs_timestep(
    results: NoiseAnalysisResults, output_dir: Path, figsize: Tuple[int, int] = (12, 6)
) -> None:
    import os

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    timesteps = sorted(results.keys())
    mse_losses = [results[t]["stats"]["mse"] for t in timesteps]
    mae_losses = [results[t]["stats"]["mae"] for t in timesteps]
    fig, ax1 = plt.subplots(figsize=figsize)
    color = "tab:blue"
    ax1.set_xlabel("Timestep (t)")
    ax1.set_ylabel("MSE Loss", color=color)
    line1 = ax1.semilogy(timesteps, mse_losses, "o-", color=color, label="MSE")
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("MAE Loss", color=color)
    line2 = ax2.semilogy(timesteps, mae_losses, "s--", color=color, label="MAE")
    ax2.tick_params(axis="y", labelcolor=color)
    if max(timesteps) >= 1000:
        ax1.axvline(x=1000, color="gray", linestyle=":", alpha=0.7, label="t=1000")
    plt.title("Noise Prediction Loss vs Timestep (Log Scale)")
    ax1.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if max(timesteps) >= 1000:
        lines.append(plt.Line2D([0], [0], color="gray", linestyle=":", alpha=0.7))
        labels.append("t=1000")
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plot_path = output_dir / "loss_vs_timestep.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_noise_scales(results: NoiseAnalysisResults, output_dir: Path):
    timesteps = sorted(results.keys())
    true_scales = [results[t]["true_noise"].std().item() for t in timesteps]
    pred_scales = [results[t]["pred_noise"].std().item() for t in timesteps]
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, true_scales, "o-", label="True Noise Scale")
    plt.plot(timesteps, pred_scales, "s-", label="Predicted Noise Scale")
    plt.axhline(1.0, color="gray", linestyle="--", label="Unit Normal")
    plt.xlabel("Timestep")
    plt.ylabel("Noise Scale (Std Dev)")
    plt.title("Noise Scale vs Timestep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "noise_scales.png", dpi=150)
    plt.close()
