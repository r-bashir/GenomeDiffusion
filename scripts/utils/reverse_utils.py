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


# ==================== Core Reverse Process Analysis ====================
def to_numpy(tensor):
    """Convert a PyTorch tensor to numpy array."""
    if tensor is None:
        return None
    return tensor.cpu().detach().numpy()


# Generate timesteps for analysis
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


# Reverse Diffusion Process
def run_reverse_process(
    model: DiffusionModel,
    x0: Tensor,
    timesteps: List[int],
) -> ReverseDiffusionResults:
    """
    Run the reverse diffusion process analysis at specified timesteps for a single sample.
    Returns a dictionary mapping timesteps to their analysis results.
    """
    results: ReverseDiffusionResults = {}
    for t in timesteps:
        results[t] = run_reverse_step(model, x0, t)
    return results


# Reverse Diffusion Step
def run_reverse_step(
    model: DiffusionModel,
    x0: Tensor,
    timestep: int,
) -> ReverseDiffusionResult:
    """
    Run a single reverse diffusion step for a single sample and return all diagnostics.
    All variables are single-sample tensors or scalars for clarity.
    """
    model.eval()
    with torch.no_grad():
        # device: device of the input tensor
        device = x0.device

        # noise: random gaussian noise [1, 1, seq_len]
        noise = torch.randn_like(x0)

        # t_tensor: timestep as tensor [1]
        t_tensor = torch.tensor([timestep], device=device, dtype=torch.long)

        # xt: noisy sample at timestep t [1, 1, seq_len]
        xt = model.forward_diffusion.sample(x0, t_tensor, noise)

        # predicted_noise: model's prediction of noise at t [1, 1, seq_len]
        predicted_noise = model.predict_added_noise(xt, t_tensor)

        # Get all diagnostics from the core reverse step (dictionary output)
        reverse_dict = model.reverse_diffusion.reverse_diffusion_step(
            xt, t_tensor, return_all=True
        )

        # x_t_minus_1: denoised sample after reverse step [1, 1, seq_len]
        x_t_minus_1 = reverse_dict["x_prev"]

        # epsilon_theta: model's predicted noise [1, 1, seq_len]
        epsilon_theta = reverse_dict["epsilon_theta"]

        # scaled_pred_noise: β_t/√(1-ᾱ_t) * ε_θ(x_t, t) [1, 1, seq_len]
        scaled_pred_noise = reverse_dict["scaled_pred_noise"]

        # mu: denoised mean before noise is added [1, 1, seq_len]
        mu = reverse_dict["mean"]

        # sigma_t: standard deviation of noise at this step (float or tensor)
        sigma_t = reverse_dict["sigma_t"]

        # coef: scaling coefficient for predicted noise (float or tensor)
        coef = reverse_dict["coef"]

        # alpha_bar_t, beta_t, alpha_t: schedule parameters (floats or tensors)
        alpha_bar_t = reverse_dict["alpha_bar_t"]
        beta_t = reverse_dict["beta_t"]
        alpha_t = reverse_dict["alpha_t"]

        # Metrics for diagnostics only
        # noise_mse: MSE between model-predicted and true noise
        noise_mse = F.mse_loss(scaled_pred_noise, noise).item()

        # x0_diff: MSE between denoised output and original sample
        x0_diff = F.mse_loss(x_t_minus_1, x0).item()

        # noise_magnitude: mean absolute value of true noise
        noise_magnitude = torch.mean(torch.abs(noise)).item()

        # pred_noise_magnitude: mean absolute value of predicted noise
        pred_noise_magnitude = torch.mean(torch.abs(predicted_noise)).item()

        # SNR = |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
        # Convert alpha_bar_t to tensor
        alpha_bar_t = torch.as_tensor(alpha_bar_t, device=x0.device, dtype=x0.dtype)

        signal_power = torch.mean((torch.sqrt(alpha_bar_t) * x0) ** 2).item()
        noise_power = torch.mean((torch.sqrt(1 - alpha_bar_t) * noise) ** 2).item()
        signal_to_noise = signal_power / (noise_power + 1e-8)

        metrics = {
            "noise_mse": noise_mse,  # MSE between predicted and true noise
            "x0_diff": x0_diff,  # MSE between denoised output and original sample
            "sigma_t": sigma_t,  # std of noise added at this step
            "alpha_bar_t": alpha_bar_t,  # cumulative product of alphas
            "coef": coef,  # scaling coefficient for predicted noise
            "noise_magnitude": noise_magnitude,  # mean abs value of true noise
            "pred_noise_magnitude": pred_noise_magnitude,  # mean abs value of predicted noise
            "signal_to_noise": signal_to_noise,  # SNR at this step
        }

    return {
        "timestep": timestep,  # current timestep
        "x0": x0,  # original clean sample
        "xt": xt,  # noisy sample at timestep t
        "noise": noise,  # random gaussian noise
        "predicted_noise": predicted_noise,  # model's predicted noise
        "scaled_pred_noise": scaled_pred_noise,  # scaled predicted noise
        "x_t_minus_1": x_t_minus_1,  # denoised sample after reverse step
        "epsilon_theta": epsilon_theta,  # model's predicted noise (redundant, for clarity)
        "mu": mu,  # denoised mean before noise is added
        "metrics": metrics,  # dictionary of diagnostic metrics
        "beta_t": beta_t,  # noise schedule value
        "alpha_t": alpha_t,  # alpha for this step
        "alpha_bar_t": alpha_bar_t,  # cumulative product of alphas
        "sigma_t": sigma_t,  # std of noise added at this step
        "coef": coef,  # scaling coefficient for predicted noise
    }


# Print Reverse Statistics
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
        print(f"- Alpha_bar_t: {r['metrics']['alpha_bar_t']:.8f}")
        print(f"- Coef (beta/sqrt(1-alpha_bar)): {r['metrics']['coef']:.8f}")
        print(f"- Noise MSE: {r['metrics']['noise_mse']:.8f}")
        print(f"- Reconstruction MSE (x0 vs x_t-1): {r['metrics']['x0_diff']:.8f}")
        print(f"- Signal-to-Noise Ratio: {r['metrics']['signal_to_noise']:.8f}")
        print(f"- True Noise Magnitude: {r['metrics']['noise_magnitude']:.8f}")
        print(
            f"- Predicted Noise Magnitude: {r['metrics']['pred_noise_magnitude']:.8f}"
        )


# ==================== Evolution of Denoising ====================


# Visualize diffusion process at different timesteps
def visualize_diffusion_process_lineplot(
    results: ReverseDiffusionResults,
    timesteps: Optional[list] = None,
    output_dir: Optional[str] = None,
):
    """
    Visualize the forward and reverse diffusion process at different timesteps using line plots.
    Uses precomputed ReverseDiffusionResults (from run_reverse_process).

    For each timestep, shows:
    1. Original sample
    2. Added noise
    3. Noisy sample (forward diffusion)
    4. Predicted noise (both raw and scaled)
    5. Denoised sample (reverse diffusion)
    6. Denoised mean before noise is added (mu)
    7. Model's predicted noise (epsilon_theta)

    Args:
        results: ReverseDiffusionResults dict (timestep -> result dict)
        timesteps: List of timesteps to visualize (if None, use all in results)
        output_dir: Directory to save visualizations
    """
    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    n_timesteps = len(timesteps)

    fig, axes = plt.subplots(n_timesteps, 7, figsize=(21, 3 * n_timesteps))
    if n_timesteps == 1:
        axes = axes[np.newaxis, :]

    for i, t in enumerate(timesteps):
        r = results[t]
        x0 = r["x0"].squeeze().cpu().numpy()
        noise = r["noise"].squeeze().cpu().numpy()
        xt = r["xt"].squeeze().cpu().numpy()
        predicted_noise = r["predicted_noise"].squeeze().cpu().numpy()
        alpha_bar_t = r["alpha_bar_t"].squeeze().cpu().numpy()
        predicted_noise_scaled = predicted_noise / (np.sqrt(1.0 - alpha_bar_t))
        x_t_minus_1 = r["x_t_minus_1"].squeeze().cpu().numpy()
        epsilon_theta = r["epsilon_theta"].squeeze().cpu().numpy()
        scaled_epsilon_theta = r["scaled_pred_noise"].squeeze().cpu().numpy()
        mu = r["mu"].squeeze().cpu().numpy()
        seq_len = x0.shape[-1]
        x_axis = np.arange(seq_len)

        # Original sample
        axes[i, 0].plot(x_axis, x0, "b-", linewidth=1)
        axes[i, 0].set_title("Original x0")

        # Added noise
        axes[i, 1].plot(x_axis, noise, "r-", linewidth=1)
        axes[i, 1].set_title("Added Noise ε")

        # Noisy sample
        axes[i, 2].plot(x_axis, xt, "k-", linewidth=1)
        axes[i, 2].set_title("Noisy x_t")

        # Predicted noise (raw and scaled)
        axes[i, 3].plot(
            x_axis, predicted_noise, "g-", linewidth=1, label="Predicted noise: ε"
        )
        axes[i, 3].plot(
            x_axis,
            predicted_noise_scaled,
            "m-",
            linewidth=1,
            label="Scaled noise: ε/√(1-ᾱ_t)",
        )
        axes[i, 3].set_title("Predicted Noise")
        axes[i, 3].legend(fontsize=6)

        # Denoised sample (reverse step)
        axes[i, 4].plot(x_axis, x_t_minus_1, "c-", linewidth=1, label="x_{t-1}")
        axes[i, 4].plot(x_axis, mu, "y--", linewidth=1, label="μ_θ(x_t, t)")
        axes[i, 4].set_title("Denoised")
        axes[i, 4].legend(fontsize=6)

        # Model's predicted noise (epsilon_theta)
        axes[i, 5].plot(
            x_axis, epsilon_theta, color="orange", linewidth=1, label="ε_θ(x_t, t)"
        )
        axes[i, 5].plot(
            x_axis,
            scaled_epsilon_theta,
            "m-",
            linewidth=1,
            label="β_t/√(1-ᾱ_t) * ε_θ(x_t, t)",
        )
        axes[i, 5].set_title("ε_θ(x_t, t)")
        axes[i, 5].legend(fontsize=6)

        # Diagnostics: plot schedule parameters if desired (optional, e.g. coef, sigma_t)
        axes[i, 6].plot(
            x_axis,
            np.full_like(x_axis, r["sigma_t"] if "sigma_t" in r else 0),
            label="sigma_t",
        )
        axes[i, 6].plot(
            x_axis, np.full_like(x_axis, r["coef"] if "coef" in r else 0), label="coef"
        )
        axes[i, 6].set_title("Schedule Params")
        axes[i, 6].legend(fontsize=6)
        axes[i, 0].set_ylabel(f"t={t}")

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diffusion_lineplot.pdf")
        fig.savefig(f"{output_dir}/diffusion_lineplot.png")
        plt.close()
    else:
        plt.show()


# Visualize Superimposed Denoising Process
def visualize_diffusion_process_superimposed(
    results: ReverseDiffusionResults,
    timesteps: Optional[list] = None,
    output_dir: Optional[str] = None,
):
    """
    Visualize the superimposed comparison of original, noisy, and denoised signals at different timesteps.
    Uses precomputed ReverseDiffusionResults (from run_reverse_process).

    For each timestep (as columns), shows superimposed plots of:
    - Original sample (blue)
    - Noisy sample (red)
    - Denoised sample (green)

    Args:
        results: ReverseDiffusionResults dict (timestep -> result dict)
        timesteps: List of timesteps to visualize (if None, use all in results)
        output_dir: Directory to save visualizations
    """
    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    n_cols = len(timesteps)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for i, t in enumerate(timesteps):
        result = results[t]

        # Always plot only the first sample and first channel for all tensors
        # Robustly flatten all arrays for plotting
        x0_vis = to_numpy(result["x0"]).reshape(-1)
        x_t_vis = to_numpy(result["xt"]).reshape(-1)
        x_t_minus_1_vis = to_numpy(result["x_t_minus_1"]).reshape(-1)
        mu_vis = to_numpy(result["mu"]).reshape(-1)
        x_axis = np.arange(len(x0_vis))

        # Plot data
        axes[i].plot(x_axis, x0_vis, "b-", linewidth=2, label="Original x_0", alpha=0.8)
        axes[i].plot(x_axis, x_t_vis, "r-", linewidth=2, label="Noisy x_t", alpha=0.7)
        axes[i].plot(
            x_axis,
            x_t_minus_1_vis,
            "g-",
            linewidth=2,
            label="Denoised x_{t-1}",
            alpha=0.7,
        )
        axes[i].plot(
            x_axis,
            mu_vis,
            "y--",
            linewidth=2,
            label="Denoised mean μ_θ(x_t, t)",
            alpha=0.7,
        )
        axes[i].set_title(f"t={t}")
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
        initial_noise = np.mean(np.abs(to_numpy(result["xt"]) - to_numpy(result["x0"])))
        final_noise = np.mean(
            np.abs(to_numpy(result["x_t_minus_1"]) - to_numpy(result["x0"]))
        )
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
        fig.savefig(f"{output_dir}/diffusion_superimposed.pdf")
        fig.savefig(f"{output_dir}/diffusion_superimposed.png")
        plt.close()
    else:
        plt.show()


# ==================== Special Diagnostic Functions ====================
"""
Specialized diagnostic functions for analyzing the reverse diffusion process.
These functions provide comprehensive visualizations of key variables and metrics
at different timesteps to help diagnose model behavior, particularly around
critical transitions (e.g., t=999 to t=1000).

Key Diagnostic Plots:

1. Signal Plots (plot_diagnostic_signals):
   - Shows x₀ (original), x_t (noisy), μ (denoised mean), x_{t-1} (previous)
   - Value: Visualize how well the model reconstructs the original signal
     and how the denoising process evolves across timesteps

2. Noise Component Plots (plot_diagnostic_noise):
   - Shows true noise (ε), predicted noise (ε_θ), and scaled predicted noise
   - Value: Assess model's noise prediction accuracy and any systematic
     biases in noise estimation

3. Variance Analysis (plot_diagnostic_variance):
   - Compares standard deviations of x_t, μ, and x_{t-1}
   - Value: Detect variance collapse or amplification issues, particularly
     important in the high-noise regime

4. Schedule Parameters (plot_diagnostic_schedule):
   - Tracks σ_t (noise level) and coefficient values across timesteps
   - Value: Verify the diffusion schedule behaves as expected and identify
     any anomalies in parameter scaling

5. Distribution Analysis (plot_diagnostic_histograms):
   - Shows value distributions of x₀, μ, and x_{t-1}
   - Value: Detect distribution shifts, mode collapse, or other statistical
     artifacts during denoising

Usage:
    results = run_reverse_process(model, x0, timesteps)
    plot_reverse_diagnostics(results, timesteps, output_dir)
"""


def print_diagnostic_statistics(results, timesteps):
    """
    Print comprehensive diagnostic statistics for analyzing the reverse diffusion process.
    Focuses on variance amplification, noise prediction accuracy, and signal quality metrics.
    Particularly useful for diagnosing issues around critical timesteps (e.g., t=999 to t=1000).

    Args:
        results: Dict mapping timesteps to ReverseDiffusionResult
        timesteps: List of timesteps to analyze
    """
    # Prepare signals for analysis
    signals = prepare_diagnostic_signals(results, timesteps)

    print("\n" + "=" * 80)
    print(" REVERSE DIFFUSION DIAGNOSTIC STATISTICS ")
    print("=" * 80)

    # Print statistics for each timestep
    for i, t in enumerate(timesteps):
        s = signals[t]
        print(f"\nTimestep {t}:")
        print("-" * 40)

        # 1. Signal Statistics
        print("Signal Statistics:")
        for key in ["x0", "x_t", "mu", "x_t_minus_1"]:
            if s[key] is not None:
                arr = np.array(s[key])
                print(
                    f"  {key:12s}: mean={np.mean(arr):8.4f}, std={np.std(arr):8.4f}, "
                    f"min={np.min(arr):8.4f}, max={np.max(arr):8.4f}"
                )

        # 2. Noise Prediction Analysis
        print("\nNoise Prediction:")
        for key in ["noise", "predicted_noise", "scaled_pred_noise"]:
            if s[key] is not None:
                arr = np.array(s[key])
                print(f"  {key:12s}: mean={np.mean(arr):8.4f}, std={np.std(arr):8.4f}")

        # 3. Schedule Parameters
        print("\nSchedule Parameters:")
        print(f"  sigma_t     : {s['sigma_t']:8.4f}")
        print(f"  coef       : {s['coef']:8.4f}")

        # 4. Critical Metrics
        print("\nCritical Metrics:")
        # Variance ratios
        if s["x_t"] is not None and s["x_t_minus_1"] is not None:
            var_ratio = np.std(s["x_t_minus_1"]) / np.std(s["x_t"])
            print(f"  Variance Ratio (σ_{t-1}/σ_t)     : {var_ratio:8.4f}")

        # Noise prediction accuracy
        if s["noise"] is not None and s["predicted_noise"] is not None:
            noise_mse = np.mean((s["noise"] - s["predicted_noise"]) ** 2)
            print(f"  Noise Prediction MSE            : {noise_mse:8.4f}")

        # Signal quality
        if s["x0"] is not None and s["x_t_minus_1"] is not None:
            recon_mse = np.mean((s["x0"] - s["x_t_minus_1"]) ** 2)
            print(f"  Reconstruction MSE (x0 vs x_{t-1}): {recon_mse:8.4f}")

        # Compare consecutive timesteps
        if i > 0:
            prev_t = timesteps[i - 1]
            prev_s = signals[prev_t]
            if s["x_t_minus_1"] is not None and prev_s["x_t_minus_1"] is not None:
                var_change = np.std(s["x_t_minus_1"]) / np.std(prev_s["x_t_minus_1"])
                print(f"\nVariance Change from t={prev_t}:")
                print(f"  σ_{t}/σ_{prev_t}: {var_change:8.4f}")
                if var_change > 10:
                    print("  WARNING: Large variance increase detected!")
                elif var_change < 0.1:
                    print("  WARNING: Large variance collapse detected!")


def plot_reverse_diagnostics(results, timesteps, output_dir=None):
    """
    Generate comprehensive diagnostic plots for reverse diffusion.
    Args:
        results: Dict mapping timesteps to ReverseDiffusionResult
        timesteps: List of timesteps to analyze
        output_dir: If given, save plots there. Otherwise, plt.show()
    """
    # Filter timesteps to those present in results
    timesteps = sorted([t for t in timesteps if t in results])
    if not timesteps:
        return

    # Generate all diagnostic plots
    plot_diagnostic_signals(results, timesteps, output_dir)
    plot_diagnostic_noise(results, timesteps, output_dir)
    plot_diagnostic_variance(results, timesteps, output_dir)
    plot_diagnostic_schedule(results, timesteps, output_dir)
    plot_diagnostic_histograms(results, timesteps, output_dir)


def plot_diagnostic_signals(results, timesteps, output_dir=None):
    """Plot x0, x_t, mu, x_t_minus_1 for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    n_rows = (len(timesteps) + 1) // 2  # 2 plots per row
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axs = axs.flatten() if n_rows > 1 else [axs]

    for i, t in enumerate(timesteps):
        s = signals[t]
        axs[i].plot(s["x0"], label="x0", color="blue", linewidth=1.5)
        axs[i].plot(s["x_t"], label="x_t", color="black", alpha=0.6)
        axs[i].plot(
            s["mu"], label="Denoised mean μ_θ(x_t, t)", color="orange", linestyle="--"
        )
        axs[i].plot(
            s["x_t_minus_1"], label="Denoised x_{t-1}", color="green", alpha=0.7
        )
        axs[i].set_title(f"Reverse Diffusion Signals at t={t}")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_signals.png")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_noise(results, timesteps, output_dir=None):
    """Plot noise components for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    n_rows = (len(timesteps) + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axs = axs.flatten() if n_rows > 1 else [axs]

    for i, t in enumerate(timesteps):
        s = signals[t]
        axs[i].plot(s["noise"], label="True Noise", color="gray", alpha=0.7)
        axs[i].plot(
            s["predicted_noise"], label="Predicted Noise", color="red", alpha=0.7
        )
        if s["scaled_pred_noise"] is not None:
            axs[i].plot(
                s["scaled_pred_noise"],
                label="Scaled Pred Noise",
                color="purple",
                linestyle=":",
            )
        axs[i].set_title(f"Noise Comparison at t={t}")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_noise.png")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_variance(results, timesteps, output_dir=None):
    """Plot variance bar plots for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    labels = ["x_t", "μ_θ(x_t, t)", "x_{t-1}"]
    keys = ["x_t", "mu", "x_t_minus_1"]

    fig, ax = plt.subplots(figsize=(2 * len(timesteps) + 6, 5))
    x = np.arange(len(labels))
    width = 0.8 / len(timesteps)

    for i, t in enumerate(timesteps):
        stds = [np.std(signals[t][k]) for k in keys]
        offset = (i - len(timesteps) / 2 + 0.5) * width
        ax.bar(x + offset, stds, width, label=f"t={t}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Variance of Key Signals")
    ax.legend()
    fig.tight_layout()

    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_variance.png")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_schedule(results, timesteps, output_dir=None):
    """Plot schedule parameters across timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    fig, ax = plt.subplots(figsize=(10, 4))

    for param, color in [("sigma_t", "red"), ("coef", "blue")]:
        vals = [signals[t][param] for t in timesteps]
        ax.plot(timesteps, vals, marker="o", label=param, color=color)

    ax.set_title("Schedule Parameters Across Timesteps")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_schedule.png")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_histograms(results, timesteps, output_dir=None):
    """Plot histograms for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    n_rows = len(timesteps)
    fig, axs = plt.subplots(
        n_rows, 3, figsize=(14, 3 * n_rows), sharex="col", sharey="row"
    )
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    for i, t in enumerate(timesteps):
        s = signals[t]
        for j, key in enumerate(["x0", "mu", "x_t_minus_1"]):
            axs[i, j].hist(s[key], bins=40, alpha=0.7)
            axs[i, j].set_title(f"{key} at t={t}")

    for ax in axs[-1]:
        ax.set_xlabel("Value")
    for ax in axs[:, 0]:
        ax.set_ylabel("Count")

    fig.suptitle("Histograms of Key Signals")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_histograms.png")
        plt.close(fig)
    else:
        plt.show()


def prepare_diagnostic_signals(results, timesteps):
    """
    Extract and prepare all signals needed for diagnostics from results.
    Args:
        results: Dict mapping timesteps to ReverseDiffusionResult
        timesteps: List of timesteps to analyze
    Returns:
        Dict mapping timesteps to prepared signals
    """

    def flatten(x):
        return x.reshape(-1) if x is not None else None

    def get(key, r):
        return flatten(to_numpy(r[key])) if key in r and r[key] is not None else None

    signals = {}
    for t in timesteps:
        signals[t] = {
            "x0": get("x0", results[t]),
            "x_t": get("xt", results[t]),
            "mu": get("mu", results[t]),
            "x_t_minus_1": get("x_t_minus_1", results[t]),
            "noise": get("noise", results[t]),
            "predicted_noise": get("predicted_noise", results[t]),
            "scaled_pred_noise": get("scaled_pred_noise", results[t]),
            "sigma_t": float(results[t].get("sigma_t", 0)),  # Convert to scalar
            "coef": float(results[t].get("coef", 0)),  # Convert to scalar
        }
    return signals
