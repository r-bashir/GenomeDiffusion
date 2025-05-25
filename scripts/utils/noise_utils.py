#!/usr/bin/env python
# coding: utf-8
"""Noise Analysis Utilities for Diffusion Models

This module provides tools for analyzing the noise prediction performance of diffusion models
across different timesteps. It helps diagnose issues in the diffusion process and understand
how well the model predicts noise at different stages of the forward diffusion process.

Key Functions:
    - run_noise_analysis: Main function to run noise analysis across specified timesteps
    - analyze_noise_prediction: Analyze noise prediction at a single timestep
    - plot_noise_analysis_results: Generate visualizations of the analysis
    - plot_loss_vs_timestep: Plot MSE and MAE loss as a function of timestep
    - save_noise_analysis: Save analysis results to CSV

Expected Behavior:
1. For each timestep, the code will:
   - Add noise to clean samples using the forward process
   - Predict the noise using the model
   - Compare predicted vs actual noise using MSE and MAE metrics
   - Calculate statistics about the noise distributions

2. The analysis will generate:
   - A summary plot showing noise statistics and error metrics
   - A dedicated loss vs timestep plot
   - A CSV file with detailed metrics

Common Issues to Watch For:
- High error at specific timesteps (especially t=1000)
- Mismatch between true and predicted noise statistics
- Non-monotonic behavior in error metrics
- Sudden jumps or drops in signal-to-noise ratio
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import norm
from torch import Tensor

from src import DiffusionModel

# Type aliases for better readability
NoiseAnalysisResult = Dict[str, Any]
NoiseAnalysisResults = Dict[int, NoiseAnalysisResult]


def run_noise_analysis(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int = 3,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> NoiseAnalysisResults:
    """Run detailed noise analysis at specified timesteps.

    Args:
        model: The diffusion model to analyze
        x0: Clean input tensor of shape [batch_size, channels, seq_length]
        num_samples: Number of samples to analyze per timestep
        timesteps: List of timesteps to analyze. Defaults to [1, 10, 100, 500, 1000]
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping timesteps to their analysis results
    """
    if timesteps is None:
        timesteps = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    results: NoiseAnalysisResults = {}

    if verbose:
        print("\n" + "=" * 70)
        print(" STARTING DETAILED NOISE ANALYSIS ")
        print("=" * 70)

    for t in timesteps:
        results[t] = analyze_noise_prediction_at_timestep(
            model=model,
            x0=x0[:num_samples],
            timestep=t,
            num_samples=num_samples,
            verbose=verbose,
        )
        if verbose:
            print("\n" + "-" * 50 + "\n")

    return results


def analyze_noise_prediction_at_timestep(
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
        pred_noise = model.reverse_diffusion.reverse_diffusion_step(xt, t_tensor)

        # Calculate errors and statistics
        abs_errors = (pred_noise - true_noise).abs()
        mse_errors = (pred_noise - true_noise).pow(2)

        stats = {
            "true_noise_mean": true_noise.mean().item(),
            "true_noise_std": true_noise.std().item(),
            "pred_noise_mean": pred_noise.mean().item(),
            "pred_noise_std": pred_noise.std().item(),
            "mse": mse_errors.mean().item(),
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
    print(f"\n[Prediction Errors]")
    print(f"MSE           : {stats['mse']:.6f}")
    print(f"MAE           : {stats['mae']:.6f}")
    print(f"Max AE        : {stats['max_ae']:.6f}")
    print(f"Signal/Noise  : {stats['signal_to_noise']:.2f}")


def plot_noise_analysis_results(
    results: NoiseAnalysisResults, output_dir: Path, figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Plot noise analysis results across timesteps.

    This function creates a summary plot with three subplots:
    1. Noise Statistics: Shows true and predicted noise means and standard deviations
    2. Error Metrics: Displays MSE, MAE, and Max AE on a log scale
    3. Signal to Noise Ratio: Shows how the SNR changes across timesteps

    Args:
        results: Dictionary mapping timesteps to analysis results
        output_dir: Directory to save the plot
        figsize: Figure size as (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    timesteps = sorted(results.keys())

    # Create a 2x3 grid of subplots (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1] * 1.5))

    # Plot 1: Noise Statistics
    means = [results[t]["stats"]["true_noise_mean"] for t in timesteps]
    stds = [results[t]["stats"]["true_noise_std"] for t in timesteps]
    pred_means = [results[t]["stats"]["pred_noise_mean"] for t in timesteps]
    pred_stds = [results[t]["stats"]["pred_noise_std"] for t in timesteps]

    axes[0].plot(timesteps, means, "o-", label="True Noise Mean")
    axes[0].plot(timesteps, stds, "s-", label="True Noise Std")
    axes[0].plot(timesteps, pred_means, "o--", label="Pred Noise Mean")
    axes[0].plot(timesteps, pred_stds, "s--", label="Pred Noise Std")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Noise Statistics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Error Metrics
    mse = [results[t]["stats"]["mse"] for t in timesteps]
    mae = [results[t]["stats"]["mae"] for t in timesteps]
    max_ae = [results[t]["stats"]["max_ae"] for t in timesteps]

    axes[1].semilogy(timesteps, mse, "o-", label="MSE")
    axes[1].semilogy(timesteps, mae, "s-", label="MAE")
    axes[1].semilogy(timesteps, max_ae, "^-", label="Max AE")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error Metrics (log scale)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Signal to Noise Ratio
    snr = [results[t]["stats"]["signal_to_noise"] for t in timesteps]
    axes[2].plot(timesteps, snr, "o-")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("SNR (xt / pred_noise)")
    axes[2].set_title("Signal to Noise Ratio")
    axes[2].grid(True, alpha=0.3)

    # Add a vertical line at t=1000 if it's in the range
    if max(timesteps) >= 1000:
        for ax in axes:
            ax.axvline(x=1000, color="gray", linestyle=":", alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = output_dir / "noise_analysis_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_vs_timestep(
    results: NoiseAnalysisResults, output_dir: Path, figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot the loss as a function of timestep.

    Args:
        results: Dictionary containing noise analysis results for different timesteps
        output_dir: Directory to save the plot
        figsize: Figure size (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract timesteps and sort them
    timesteps = sorted(results.keys())

    # Extract metrics
    mse_losses = [results[t]["stats"]["mse"] for t in timesteps]
    mae_losses = [results[t]["stats"]["mae"] for t in timesteps]

    # Create figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot MSE on primary y-axis
    color = "tab:blue"
    ax1.set_xlabel("Timestep (t)")
    ax1.set_ylabel("MSE Loss", color=color)
    line1 = ax1.semilogy(timesteps, mse_losses, "o-", color=color, label="MSE")
    ax1.tick_params(axis="y", labelcolor=color)

    # Create secondary y-axis for MAE
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("MAE Loss", color=color)
    line2 = ax2.semilogy(timesteps, mae_losses, "s--", color=color, label="MAE")
    ax2.tick_params(axis="y", labelcolor=color)

    # Add a vertical line at t=1000 if it's in the range
    if max(timesteps) >= 1000:
        ax1.axvline(x=1000, color="gray", linestyle=":", alpha=0.7, label="t=1000")

    # Add a title and grid
    plt.title("Noise Prediction Loss vs Timestep (Log Scale)")
    ax1.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if max(timesteps) >= 1000:
        lines.append(plt.Line2D([0], [0], color="gray", linestyle=":", alpha=0.7))
        labels.append("t=1000")
    ax1.legend(lines, labels, loc="upper right")

    # Adjust layout and save
    plt.tight_layout()
    plot_path = output_dir / "loss_vs_timestep.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_noise_analysis(
    results: NoiseAnalysisResults,
    output_dir: Path,
    filename: str = "noise_analysis_results.csv",
) -> None:
    """Save noise analysis results to a CSV file.

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
        row.update(result["stats"])
        rows.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False, float_format="%.6f")


def plot_noise_distributions(results: NoiseAnalysisResults, output_dir: Path):
    """Compare true vs predicted noise distributions at different timesteps"""
    timesteps = sorted(results.keys())
    n_timesteps = len(timesteps)

    fig, axes = plt.subplots(n_timesteps, 2, figsize=(12, 3 * n_timesteps))

    for i, t in enumerate(timesteps):
        # Flatten the noise tensors
        true_noise = results[t]["true_noise"].flatten().cpu().numpy()
        pred_noise = results[t]["pred_noise"].flatten().cpu().numpy()

        # Plot histograms
        axes[i, 0].hist(true_noise, bins=100, alpha=0.7, label="True", density=True)
        axes[i, 1].hist(
            pred_noise,
            bins=100,
            alpha=0.7,
            color="red",
            label="Predicted",
            density=True,
        )

        # Add Gaussian fit
        x = np.linspace(
            min(true_noise.min(), pred_noise.min()),
            max(true_noise.max(), pred_noise.max()),
            100,
        )
        axes[i, 0].plot(x, norm.pdf(x, 0, 1), "k--", label="N(0,1)")
        axes[i, 1].plot(x, norm.pdf(x, 0, 1), "k--", label="N(0,1)")

        axes[i, 0].set_title(f"t={t}: True Noise Distribution")
        axes[i, 1].set_title(f"t={t}: Predicted Noise Distribution")
        axes[i, 0].legend()
        axes[i, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "noise_distributions.png", dpi=150)
    plt.close()


def plot_spatial_correlations(
    results: NoiseAnalysisResults, output_dir: Path, max_lag: int = 100
):
    """Analyze spatial correlations in the noise prediction errors

    Args:
        results: Dictionary containing noise analysis results
        output_dir: Directory to save the plot
        max_lag: Maximum lag for autocorrelation calculation (default: 100)
    """
    os.makedirs(output_dir, exist_ok=True)
    timesteps = sorted(results.keys())

    # Create a single figure for all timesteps
    fig, axes = plt.subplots(
        len(timesteps),
        2,
        figsize=(15, 3 * len(timesteps)),
        squeeze=False,  # Ensure axes is always 2D even with single timestep
    )

    for i, t in enumerate(timesteps):
        try:
            # Calculate errors and ensure we have a 1D array
            error = (results[t]["true_noise"] - results[t]["pred_noise"]).squeeze()
            if isinstance(error, torch.Tensor):
                error = error.cpu().numpy()

            # Ensure we have enough data points
            if len(error) < 2:
                print(f"Warning: Not enough data points for timestep {t}")
                continue

            # Limit max_lag to be less than the sequence length
            valid_max_lag = min(max_lag, len(error) - 1)
            if valid_max_lag < 2:
                print(
                    f"Warning: Sequence too short for meaningful autocorrelation at timestep {t}"
                )
                continue

            # Calculate autocorrelation
            autocorr = [1.0]  # Autocorrelation at lag 0 is 1
            for l in range(1, valid_max_lag):
                if len(error[:-l]) >= 2:  # Need at least 2 points for correlation
                    corr = np.corrcoef(error[:-l], error[l:])[0, 1]
                    if not np.isnan(corr):  # Only add if correlation is valid
                        autocorr.append(corr)
                    else:
                        autocorr.append(0.0)  # Default to 0 if correlation is NaN
                else:
                    autocorr.append(0.0)  # Not enough points, default to 0

            # Plot autocorrelation
            axes[i, 0].plot(autocorr[:valid_max_lag])
            axes[i, 0].set_title(f"t={t}: Error Autocorrelation")
            axes[i, 0].set_xlabel("Lag")
            axes[i, 0].set_ylabel("Correlation")
            axes[i, 0].grid(True, alpha=0.3)

            # Plot power spectral density
            try:
                # Ensure we have enough points for the FFT
                nperseg = min(
                    256, len(error) // 2
                )  # Use smaller segments if sequence is short
                if nperseg >= 2:  # Need at least 2 points for FFT
                    f, Pxx = welch(error, nperseg=nperseg)
                    axes[i, 1].loglog(f, Pxx)
                    axes[i, 1].set_title(f"t={t}: Power Spectral Density")
                    axes[i, 1].set_xlabel("Frequency")
                    axes[i, 1].set_ylabel("Power")
                    axes[i, 1].grid(True, alpha=0.3)
                else:
                    axes[i, 1].text(
                        0.5,
                        0.5,
                        "Insufficient data for PSD",
                        ha="center",
                        va="center",
                        transform=axes[i, 1].transAxes,
                    )
                    axes[i, 1].axis("off")

            except Exception as e:
                print(f"Error calculating PSD for timestep {t}: {str(e)}")
                axes[i, 1].text(
                    0.5,
                    0.5,
                    "Error in PSD calculation",
                    ha="center",
                    va="center",
                    transform=axes[i, 1].transAxes,
                )
                axes[i, 1].axis("off")

        except Exception as e:
            print(f"Error processing timestep {t}: {str(e)}")
            for j in range(2):
                axes[i, j].text(
                    0.5,
                    0.5,
                    "Error in calculation",
                    ha="center",
                    va="center",
                    transform=axes[i, j].transAxes,
                )
                axes[i, j].axis("off")

    plt.tight_layout()
    plot_path = output_dir / "spatial_correlations.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_heatmap(results: NoiseAnalysisResults, output_dir: Path):
    """Create a heatmap of errors across timesteps and sequence positions"""
    timesteps = sorted(results.keys())
    seq_len = results[timesteps[0]]["true_noise"].shape[-1]

    # Initialize error matrix
    error_matrix = np.zeros((len(timesteps), seq_len))

    # Fill error matrix
    for i, t in enumerate(timesteps):
        error = (
            (results[t]["true_noise"] - results[t]["pred_noise"])
            .squeeze()
            .cpu()
            .numpy()
        )
        error_matrix[i] = np.mean(np.abs(error), axis=0)  # Average over batch

    # Plot heatmap
    plt.figure(figsize=(15, 8))
    plt.imshow(
        error_matrix,
        aspect="auto",
        cmap="viridis",
        extent=[0, seq_len, timesteps[-1], timesteps[0]],
    )
    plt.colorbar(label="MAE")
    plt.xlabel("Sequence Position")
    plt.ylabel("Timestep")
    plt.title("Error Heatmap Across Sequence and Timesteps")
    plt.tight_layout()
    plt.savefig(output_dir / "error_heatmap.png", dpi=150)
    plt.close()


def plot_noise_scales(results: NoiseAnalysisResults, output_dir: Path):
    """Analyze how noise scales with timesteps"""
    timesteps = sorted(results.keys())

    # Calculate statistics
    true_scales = [results[t]["true_noise"].std().item() for t in timesteps]
    pred_scales = [results[t]["pred_noise"].std().item() for t in timesteps]

    # Plot
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


def plot_error_statistics(results: NoiseAnalysisResults, output_dir: Path):
    """Plot various error statistics across timesteps"""
    timesteps = sorted(results.keys())

    # Calculate statistics
    mse = [results[t]["stats"]["mse"] for t in timesteps]
    mae = [results[t]["stats"]["mae"] for t in timesteps]
    max_ae = [results[t]["stats"]["max_ae"] for t in timesteps]
    snr = [results[t]["stats"]["signal_to_noise"] for t in timesteps]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot MSE
    axes[0].semilogy(timesteps, mse, "o-")
    axes[0].set_title("MSE vs Timestep")
    axes[0].set_xlabel("Timestep")
    axes[0].grid(True, alpha=0.3)

    # Plot MAE
    axes[1].semilogy(timesteps, mae, "o-")
    axes[1].set_title("MAE vs Timestep")
    axes[1].set_xlabel("Timestep")
    axes[1].grid(True, alpha=0.3)

    # Plot Max AE
    axes[2].semilogy(timesteps, max_ae, "o-")
    axes[2].set_title("Max Absolute Error vs Timestep")
    axes[2].set_xlabel("Timestep")
    axes[2].grid(True, alpha=0.3)

    # Plot SNR
    axes[3].plot(timesteps, snr, "o-")
    axes[3].set_title("Signal-to-Noise Ratio vs Timestep")
    axes[3].set_xlabel("Timestep")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_statistics.png", dpi=150)
    plt.close()
