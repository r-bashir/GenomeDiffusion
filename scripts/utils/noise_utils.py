#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

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


# ==================== Core Noise Analysis ====================
def run_noise_analysis(
    model: DiffusionModel,
    x0: Tensor,
    num_samples: int = 10,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
) -> NoiseAnalysisResults:
    """Run detailed noise analysis at specified timesteps.

    Args:
        model: The diffusion model to analyze
        x0: Clean input tensor of shape [batch_size, channels, seq_length]
        num_samples: Number of samples to analyze per timestep
        timesteps: List of timesteps to analyze. If None, uses full range from tmin to tmax in steps of 10
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping timesteps to their analysis results
    """
    if timesteps is None:
        # Use full range of timesteps from tmin to tmax in steps of 10
        tmin = model.forward_diffusion.tmin
        tmax = model.forward_diffusion.tmax
        timesteps = list(range(tmin, tmax + 1, 10))

        # Always include tmin and tmax
        if tmin not in timesteps:
            timesteps.insert(0, tmin)
        if tmax not in timesteps:
            timesteps.append(tmax)

    # Dictionary to store results for each timestep
    results: NoiseAnalysisResults = {}

    if verbose:
        print("\n" + "=" * 70)
        print(f" STARTING DETAILED NOISE ANALYSIS (timesteps: {len(timesteps)}) ")
        print("=" * 70)

    for t in timesteps:
        if verbose:
            print(f"\nAnalyzing noise prediction at timestep {t}...")

        results[t] = noise_prediction_at_timestep(
            model=model,
            x0=x0,
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
    num_samples: int = 10,
    verbose: bool = True,
) -> NoiseAnalysisResult:
    """Analyze noise prediction at a specific timestep.

    Args:
        model: The diffusion model
        x0: Clean input tensor of shape [batch_size, channels, seq_length]
        timestep: Timestep to analyze
        num_samples: Number of samples to analyze
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


# ==================== Plotting Functions ====================
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


def plot_noise_histogram_grid(
    results: NoiseAnalysisResults, output_dir: Path, num_bins: int = 50
):
    """Plot grid of true and predicted noise histograms at each timestep (top: true, bottom: predicted)."""
    timesteps = sorted(results.keys())
    n_timesteps = len(timesteps)

    fig, axes = plt.subplots(2, n_timesteps, figsize=(4 * n_timesteps, 6))
    # axes[0, :] = true, axes[1, :] = predicted

    for i, t in enumerate(timesteps):
        # Flatten the noise tensors
        true_noise = results[t]["true_noise"].flatten().cpu().numpy()
        pred_noise = results[t]["pred_noise"].flatten().cpu().numpy()

        # Plot true noise (top row)
        axes[0, i].hist(
            true_noise, bins=num_bins, alpha=0.7, label="True", density=True
        )
        x = np.linspace(
            min(true_noise.min(), pred_noise.min()),
            max(true_noise.max(), pred_noise.max()),
            100,
        )
        axes[0, i].plot(x, norm.pdf(x, 0, 1), "k--", label="N(0,1)")
        axes[0, i].set_title(f"t={t}: True Noise Distribution")
        axes[0, i].legend()

        # Plot predicted noise (bottom row)
        axes[1, i].hist(
            pred_noise,
            bins=num_bins,
            alpha=0.7,
            color="red",
            label="Predicted",
            density=True,
        )
        axes[1, i].plot(x, norm.pdf(x, 0, 1), "k--", label="N(0,1)")
        axes[1, i].set_title(f"t={t}: Predicted Noise Distribution")
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "noise_hist_grid.png", dpi=150)
    plt.close()


def plot_noise_correlation_scatter(results: dict, output_dir: Path):
    """
    For each timestep, plot a scatter plot of true_noise vs predicted_noise and show the Pearson correlation coefficient.
    Saves all subplots in a grid as noise_correlation_scatter.png in output_dir.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    timesteps = sorted(results.keys())
    n_timesteps = len(timesteps)
    ncols = min(n_timesteps, 5)
    nrows = (n_timesteps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if n_timesteps > 1 else [axes]
    for i, t in enumerate(timesteps):
        true_noise = results[t]["true_noise"].flatten().cpu().numpy()
        pred_noise = results[t]["pred_noise"].flatten().cpu().numpy()
        ax = axes[i]
        ax.scatter(true_noise, pred_noise, s=1, alpha=0.3, color="royalblue")
        r, _ = pearsonr(true_noise, pred_noise)
        ax.set_title(f"t={t} | r={r:.2f}")
        ax.set_xlabel("True Noise")
        ax.set_ylabel("Predicted Noise")
        ax.grid(True, linestyle=":", alpha=0.5)
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "noise_corr_scatter.png", dpi=200)
    plt.close()


def plot_noise_overlay_with_comparison(
    results: NoiseAnalysisResults,
    output_dir: Path,
    num_bins: int = 50,
    mode: str = "difference",
) -> None:
    """
    Plot overlaid true and predicted noise histograms with a comparison subplot for each timestep (HEP-style).
    Each column is a timestep. Top row: overlaid histograms. Bottom row: comparison (difference, ratio, or pull).

    Args:
        results: Dictionary mapping timesteps to analysis results (must contain 'true_noise' and 'pred_noise')
        output_dir: Directory to save the plot
        num_bins: Number of bins for the histograms
        mode: 'difference', 'ratio', or 'pull' (default: 'difference')
    """
    assert mode in (
        "difference",
        "ratio",
        "pull",
    ), "mode must be one of 'difference', 'ratio', or 'pull'"
    timesteps = sorted(results.keys())
    num_timesteps = len(timesteps)
    fig = plt.figure(figsize=(6 * num_timesteps, 8))
    gs = fig.add_gridspec(
        2, num_timesteps, height_ratios=[3, 1], hspace=0.3, wspace=0.4
    )

    for i, t in enumerate(timesteps):
        ax_main = fig.add_subplot(gs[0, i])
        ax_comp = fig.add_subplot(gs[1, i], sharex=ax_main)
        true_np = results[t]["true_noise"].flatten().cpu().numpy()
        pred_np = results[t]["pred_noise"].flatten().cpu().numpy()
        data_min = min(true_np.min(), pred_np.min())
        data_max = max(true_np.max(), pred_np.max())
        bins = np.linspace(data_min, data_max, num_bins + 1)
        hist_true, _ = np.histogram(true_np, bins=bins)
        hist_pred, _ = np.histogram(pred_np, bins=bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Main panel: Plot histograms
        ax_main.hist(
            bin_centers,
            bins=bins,
            weights=hist_true,
            histtype="step",
            color="blue",
            label="True Noise",
            lw=2,
        )
        ax_main.hist(
            bin_centers,
            bins=bins,
            weights=hist_pred,
            histtype="step",
            color="red",
            label="Predicted Noise",
            linestyle="--",
            lw=2,
        )
        ax_main.fill_between(
            bin_centers,
            hist_true - np.sqrt(hist_true),
            hist_true + np.sqrt(hist_true),
            step="mid",
            color="blue",
            alpha=0.2,
        )
        stats_text = f"t = {t}\nμ_true = {true_np.mean():.3f} ± {true_np.std():.3f}\nμ_pred = {pred_np.mean():.3f} ± {pred_np.std():.3f}"
        ax_main.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_main.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3.0),
        )
        # Comparison subplot
        if mode == "difference":
            comp = hist_pred - hist_true
            comp_err = np.sqrt(np.maximum(hist_pred, 0) + np.maximum(hist_true, 0))
            ax_comp.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_comp.step(
                bin_centers, comp, where="mid", color="red", lw=1.5, label="Pred - True"
            )
            ax_comp.fill_between(
                bin_centers,
                comp - comp_err,
                comp + comp_err,
                step="mid",
                color="red",
                alpha=0.3,
                label="Uncertainty (√N)",
            )
            if i == 0:
                ax_comp.set_ylabel("Pred - True")
        elif mode == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                comp = np.divide(hist_pred, hist_true, where=hist_true != 0)
                comp_err = comp * np.sqrt(
                    1 / np.maximum(hist_pred, 1e-8) + 1 / np.maximum(hist_true, 1e-8)
                )
            ax_comp.axhline(1, color="gray", linestyle="--", alpha=0.5)
            ax_comp.step(
                bin_centers, comp, where="mid", color="red", lw=1.5, label="Pred/True"
            )
            ax_comp.fill_between(
                bin_centers,
                comp - comp_err,
                comp + comp_err,
                step="mid",
                color="red",
                alpha=0.3,
                label="Uncertainty",
            )
            if i == 0:
                ax_comp.set_ylabel("Pred / True")
        elif mode == "pull":
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma = np.sqrt(np.maximum(hist_pred, 0) + np.maximum(hist_true, 0))
                sigma[sigma == 0] = 1e-8
                comp = (hist_pred - hist_true) / sigma
            ax_comp.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax_comp.step(
                bin_centers, comp, where="mid", color="red", lw=1.5, label="Pull"
            )
            if i == 0:
                ax_comp.set_ylabel("Pull")
        if i == 0:
            ax_main.set_ylabel("Counts")
        ax_comp.set_xlabel("Noise Value")
        ax_main.set_title(f"Timestep {t}")
        if i == 0:
            ax_main.legend(loc="upper right")
            ax_comp.legend(loc="upper right")
    plt.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"noise_hist_{mode}.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()


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
    plot_path = output_dir / "noise_analysis_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
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
