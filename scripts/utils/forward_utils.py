#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Forward Diffusion Utilities

This module contains utility functions for analyzing the forward diffusion process.
It includes functions for running the forward diffusion process, calculating statistics,
and visualizing the results.

Key Functions:
    - run_forward_process: Run forward diffusion at specified timesteps
    - run_forward_process_at_timestep: Run forward diffusion at a single timestep
    - print_forward_statistics: Print statistics about the diffusion process
    - save_forward_analysis: Save analysis results to a CSV file
    - generate_timesteps: Create timestep sequences for analysis
    - plot_forward_diffusion_sample: Visualize samples at different noise levels
    - plot_signal_noise_ratio: Plot SNR and signal/noise percentages
    - plot_diffusion_parameters: Plot diffusion parameters across timesteps
    - create_animation_frames: Generate animation showing noise progression
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.forward_diffusion import ForwardDiffusion

# Type aliases for better readability
ForwardDiffusionResult = Dict[str, Any]
ForwardDiffusionResults = Dict[int, ForwardDiffusionResult]
TimestepDict = Dict[str, List[int]]


# ==================== Core Forward Process Analysis ====================
def generate_timesteps(tmin: int = 1, tmax: int = 1000) -> TimestepDict:
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


# Forward Diffusion Process
def run_forward_process(
    forward_diff: ForwardDiffusion,
    x0: Tensor,
    timesteps: List[int],
) -> ForwardDiffusionResults:
    """
    Run forward diffusion process at specified timesteps.

    Args:
        forward_diff: The forward diffusion model
        x0: Original clean sample [batch_size, channels, seq_length]
        timesteps: List of timesteps to analyze

    Returns:
        Dictionary mapping timesteps to analysis results
    """
    # Dictionary to store results for each timestep
    results: ForwardDiffusionResults = {}

    # Run diffusion analysis for each timestep
    for t in timesteps:
        results[t] = run_forward_step(forward_diff, x0, t)

    return results


# Forward Diffusion Step
def run_forward_step(
    forward_diff: ForwardDiffusion,
    x0: Tensor,
    timestep: int,
) -> ForwardDiffusionResult:
    """
    Run forward diffusion process at a specific timestep.

    Args:
        forward_diff: The forward diffusion model
        x0: Original clean sample [batch_size, channels, seq_length]
        timestep: The timestep to analyze

    Returns:
        Dictionary containing analysis results for the timestep
    """
    device = x0.device

    # Convert timestep to tensor
    t_tensor = torch.tensor([timestep], device=device)

    # Generate noise for this timestep
    eps = torch.randn_like(x0)

    # Apply forward diffusion to get x_t
    # q(x_t|x_0) = N(x_t; √(ᾱ_t) * x_0, (1-ᾱ_t)I)
    # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0, I)
    x_t = forward_diff.sample(x0, t_tensor, eps)

    # Get diffusion parameters using internal methods
    # noise schedule at timestep t: β_t
    beta_t = forward_diff.beta(t_tensor).item()

    # signal retention rate at timestep t: α_t = 1 - β_t
    alpha_t = forward_diff.alpha(t_tensor).item()

    # cumulative product of alphas up to t: ᾱ_t = ∏_{s=1}^{t} α_s
    alpha_bar_t = forward_diff.alpha_bar(t_tensor).item()

    # standard deviation of noise at timestep t: σ_t = √(1-ᾱ_t)
    sigma_t = forward_diff.sigma(t_tensor).item()

    # signal component of x_t: √(ᾱ_t) * x_0
    signal_t = torch.sqrt(torch.tensor(alpha_bar_t, device=device)) * x0

    # noise component of x_t: σ_t * ε = √(1-ᾱ_t) * ε
    noise_t = torch.tensor(sigma_t, device=device) * eps

    # Signal-to-noise ratio (SNR): |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
    # This matches the literature definition (Nichol & Dhariwal, 2021)
    signal_power = torch.mean(signal_t**2).item()
    noise_power = torch.mean(noise_t**2).item()
    snr = signal_power / (noise_power + 1e-8) if alpha_bar_t < 1 else 1000.0

    # Calculate metrics for analysis
    # Average magnitude of noise component: |σ_t * ε|
    noise_magnitude = torch.mean(torch.abs(noise_t)).item()

    # Average magnitude of signal component: |√(ᾱ_t) * x_0|
    signal_magnitude = torch.mean(torch.abs(signal_t)).item()

    # Average magnitude of noisy sample: |x_t|
    x_t_magnitude = torch.mean(torch.abs(x_t)).item()

    # Return results
    return {
        "timestep": timestep,
        "x_t": x_t.detach().clone(),
        "eps": eps.detach().clone(),
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "alpha_bar_t": alpha_bar_t,
        "sigma_t": sigma_t,
        "snr": snr,
        "noise_magnitude": noise_magnitude,
        "signal_magnitude": signal_magnitude,
        "x_t_magnitude": x_t_magnitude,
        "signal_percentage": signal_power / (signal_power + noise_power) * 100,
        "noise_percentage": noise_power / (signal_power + noise_power) * 100,
    }


# Print Forward Statistics
def print_forward_statistics(
    results: ForwardDiffusionResults, timesteps: Optional[List[int]] = None
) -> None:
    """
    Print statistics about the forward diffusion process at specified timesteps.

    Args:
        results: Results from run_forward_process
        timesteps: List of timesteps to print statistics for (if None, use all timesteps)
    """
    # The forward diffusion process is defined by the following parameters:
    # β_t: noise schedule at timestep t
    # α_t = 1 - β_t: signal retention rate at timestep t
    # ᾱ_t = ∏_{s=1}^{t} α_s: cumulative product of alphas up to t
    # σ_t = √(1-ᾱ_t): standard deviation of noise at timestep t
    #
    # The noisy sample x_t is given by:
    # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0, I)
    #
    # Signal-to-noise ratio (SNR) is defined as:
    # SNR = |√(ᾱ_t) * x_0| / |√(1-ᾱ_t) * ε|

    print("\nForward statistics at selected timesteps:")

    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to those available in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    # Print statistics for each timestep
    for t in timesteps:
        r = results[t]
        print(f"\nTimestep t = {t}:")
        print(f"  β_t = {r['beta_t']:.8f}  (noise schedule at step t = 1-α_t)")
        print(f"  α_t = {r['alpha_t']:.8f}  (signal retention rate at step t)")
        print(f"  ᾱ_t = {r['alpha_bar_t']:.8f}  (cumulative product of alphas up to t)")
        print(f"  σ_t = {r['sigma_t']:.8f}  (noise standard deviation at t = √(1-ᾱ_t))")

        print(f"  SNR = {r['snr']:.8f}  (signal-to-noise ratio = |√(ᾱ_t)*x_0|/|σ_t*ε|)")
        print(f"  Signal Magnitude = {r['signal_magnitude']:.8f}  (|√(ᾱ_t)*x_0|)")
        print(f"  Noise Magnitude = {r['noise_magnitude']:.8f}  (|σ_t*ε|)")
        print(
            f"  Signal [%] = {r['signal_percentage']:.2f}%  (signal proportion in x_t)"
        )
        print(f"  Noise [%] = {r['noise_percentage']:.2f}%  (noise proportion in x_t)")


# ==================== Plotting Functions ====================


# Plot Sample Evolution through Timesteps
def plot_sample_evolution(
    results: ForwardDiffusionResults,
    x0: Tensor,
    timesteps: Optional[List[int]] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot how a sample evolves through forward diffusion timesteps.

    Layout: Time progresses down rows, components across columns
    - Column 1: Original sample x₀ (always the same)
    - Column 2: Added noise ε at each timestep
    - Column 3: Resulting noisy sample x_t

    Args:
        results: Results from run_forward_process
        x0: Original clean sample
        timesteps: Timesteps to visualize (default: all available)
        save_dir: Save directory (default: display plot)
    """

    print("\nPlotting sample evolution through timesteps...")

    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    # Prepare original sample
    x0_np = x0.squeeze().detach().cpu().numpy()
    if x0_np.ndim == 0:
        x0_np = np.array([x0_np])

    n_timesteps = len(timesteps)

    # Layout: n_timesteps rows × 3 columns (original, noise, noisy)
    fig, axes = plt.subplots(n_timesteps, 3, figsize=(12, 2.5 * n_timesteps))

    # Handle single timestep case
    if n_timesteps == 1:
        axes = axes.reshape(1, 3)

    # Get consistent y-axis limits
    all_values = [x0_np]
    for t in timesteps:
        x_t = results[t]["x_t"].squeeze().detach().cpu().numpy()
        noise = results[t]["eps"].squeeze().detach().cpu().numpy()
        if x_t.ndim == 0:
            x_t = np.array([x_t])
        if noise.ndim == 0:
            noise = np.array([noise])
        all_values.extend([x_t, noise])

    y_min = min(np.min(vals) for vals in all_values)
    y_max = max(np.max(vals) for vals in all_values)
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Plot each timestep (rows)
    for i, t in enumerate(timesteps):
        # Get data
        x_t = results[t]["x_t"].squeeze().detach().cpu().numpy()
        noise = results[t]["eps"].squeeze().detach().cpu().numpy()
        snr = results[t]["snr"]

        if x_t.ndim == 0:
            x_t = np.array([x_t])
        if noise.ndim == 0:
            noise = np.array([noise])

        # Column 1: Original sample
        axes[i, 0].plot(x0_np, "b-", linewidth=1.5)
        axes[i, 0].set_ylim(y_min, y_max)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Original x₀", fontsize=12)

        # Column 2: Added noise
        axes[i, 1].plot(noise, "r-", linewidth=1.5)
        axes[i, 1].set_ylim(y_min, y_max)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("Added Noise ε", fontsize=12)

        # Column 3: Noisy sample
        axes[i, 2].plot(x_t, "g-", linewidth=1.5)
        axes[i, 2].set_ylim(y_min, y_max)
        axes[i, 2].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 2].set_title("Noisy Sample x_t", fontsize=12)

        # Add timestep and SNR info on the left
        axes[i, 0].set_ylabel(f"t={t}\nSNR={snr:.2f}", fontsize=10)

        # Add x-axis label only for bottom row
        if i == n_timesteps - 1:
            for col in range(3):
                axes[i, col].set_xlabel("Position")

    plt.suptitle("Sample Evolution Through Forward Diffusion", fontsize=14)
    plt.tight_layout()

    # Save or show
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "sample_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Plot Diffusion Parameters
def plot_diffusion_parameters(
    results: ForwardDiffusionResults,
    x0: Tensor,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot diffusion parameters across timesteps.

    Args:
        results: Results from analyze_forward_diffusion
        x0: Original clean sample [batch_size, channels, seq_length]
        save_dir: Directory to save the plot (if None, display plot)
    """
    print("\nPlotting diffusion parameters...")

    # Handle timesteps
    timesteps = sorted(results.keys())

    # Extract parameters
    beta_ts = [results[t]["beta_t"] for t in timesteps]
    alpha_ts = [results[t]["alpha_t"] for t in timesteps]
    alpha_bar_ts = [results[t]["alpha_bar_t"] for t in timesteps]
    sigma_ts = [results[t]["sigma_t"] for t in timesteps]

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)

    # Create time arrays for different parameters
    # Note: beta and alpha are defined for t=1 to T,
    # alpha_bar and sigma are defined for t=0 to T
    t_diff = np.array(timesteps)  # t=1 to T for beta, alpha
    t_cumul = np.array([0] + timesteps)  # t=0 to T for alpha_bar, sigma

    # Main parameter plot (spans two columns)
    ax_main = fig.add_subplot(gs[0, :2])

    # Plot parameters with distinct styles (only for t≥1)
    ax_main.plot(
        t_diff,
        beta_ts,
        "b-",
        marker="o",
        markersize=4,
        label="β (noise)",
        linewidth=2,
    )
    ax_main.plot(
        t_diff,
        alpha_ts,
        "g-",
        marker="s",
        markersize=4,
        label="α (signal)",
        linewidth=2,
    )

    ax_main.set_xlabel("timesteps", fontsize=12)
    ax_main.set_ylabel("Parameter Value", fontsize=12)
    ax_main.grid(True)
    ax_main.legend(loc="center left", fontsize=10)
    ax_main.set_title("Diffusion Parameters Evolution (t=1→T)", pad=10)

    # Alpha bar plot (log scale)
    ax_alpha_bar = fig.add_subplot(gs[0, 2])
    # For alpha_bar, include t=0 value (=1.0) and t=1 to T values
    alpha_bar_full = np.concatenate(([1.0], alpha_bar_ts))
    ax_alpha_bar.plot(
        t_cumul,
        alpha_bar_full,
        "r-",
        marker="^",
        markersize=4,
        label="ᾱ (cumulative)",
        linewidth=2,
    )
    ax_alpha_bar.set_xlabel("timesteps", fontsize=12)
    ax_alpha_bar.set_ylabel("ᾱ", fontsize=12)
    ax_alpha_bar.grid(True)
    ax_alpha_bar.legend(loc="upper right", fontsize=10)
    ax_alpha_bar.set_title("Cumulative Signal (t=0→T)", pad=10)

    # Sigma plot
    ax_sigma = fig.add_subplot(gs[1, 0])
    # For sigma, include t=0 value (=0.0) and t=1 to T values
    sigma_full = np.concatenate(([0.0], sigma_ts))
    ax_sigma.plot(
        t_cumul,
        sigma_full,
        "m-",
        marker="x",
        markersize=4,
        label="σ (noise std)",
        linewidth=2,
    )
    ax_sigma.set_xlabel("timesteps", fontsize=12)
    ax_sigma.set_ylabel("σ", fontsize=12)
    ax_sigma.grid(True)
    ax_sigma.legend(loc="upper left", fontsize=10)
    ax_sigma.set_title("Noise Level (t=0→T)", pad=10)

    # x0 sample progression
    ax_sample = fig.add_subplot(gs[1, 1:])
    x0_np = x0[0].squeeze().detach().cpu().numpy()
    if x0_np.ndim == 0:
        x0_np = np.array([x0_np])

    # Plot original and key samples
    ax_sample.plot(x0_np, "k-", label="Original", alpha=0.8)
    key_ts = [1, 500, 999, 1000]
    for t in key_ts:
        if t in results:
            x_t = results[t]["x_t"].squeeze().detach().cpu().numpy()
            if x_t.ndim == 0:
                x_t = np.array([x_t])
            ax_sample.plot(x_t, alpha=0.6, label=f"t={t}")
    ax_sample.set_title("Sample Evolution", pad=10)
    ax_sample.set_xlabel("Position", fontsize=12)
    ax_sample.set_ylabel("Value", fontsize=12)
    ax_sample.legend(loc="upper right", fontsize=10)
    ax_sample.grid(True)

    # Adjust layout
    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            save_path / "diffusion_parameters.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


# Plot Signal to Noise Ratio (SNR)
def plot_snr(
    forward_diff: ForwardDiffusion,
    results: ForwardDiffusionResults,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> None:
    """
    Plot a simplified view of signal-to-noise ratio during forward diffusion.

    This function focuses on clearly showing how SNR changes as noise is added
    during the forward diffusion process, comparing actual measured values with theoretical
    values from the noise schedule.

    Args:
        forward_diff: ForwardDiffusion instance for theoretical values
        results: Results from run_forward_process for actual values
        save_dir: Directory to save the plot (if None, display plot)
        verbose: Whether to print additional information
    """
    print("\nPlotting signal to noise ratio (SNR)...")

    # Sort timesteps (t=1 to T)
    timesteps = sorted(results.keys())

    # Calculate theoretical SNR for all timesteps using proper indexing
    # Get the number of diffusion steps from the length of the alphas array
    num_steps = len(forward_diff.alphas)

    # Get device from one of the tensors
    device = forward_diff.alphas.device
    t_tensor = torch.arange(0, num_steps + 1, device=device)
    alpha_bar_values = forward_diff.alpha_bar(t_tensor).cpu().numpy()

    # Calculate theoretical SNR: α_bar_t / (1-α_bar_t)
    # For t=0, SNR is infinite (set to a large value for visualization)
    theoretical_snr = np.zeros_like(alpha_bar_values)
    theoretical_snr[0] = 10000.0  # Large value for t=0
    theoretical_snr[1:] = alpha_bar_values[1:] / (
        1 - alpha_bar_values[1:] + 1e-8
    )  # Add epsilon for numerical stability

    # Prepare actual SNR values from results
    actual_timesteps = [0] + timesteps  # Include t=0
    actual_snr = [10000.0]  # t=0 SNR (infinite, set to large value)

    for t in timesteps:
        actual_snr.append(results[t]["snr"])

    # Print comparison if verbose
    if verbose:
        print("\nSNR Comparison (Theoretical vs Actual):")
        for i, t in enumerate(actual_timesteps):
            if t == 0:
                print(f"  t={t}: Theoretical=∞, Actual=∞ (pure signal)")
            else:
                theo_snr = theoretical_snr[t]
                act_snr = actual_snr[i]
                print(
                    f"  t={t}: Theoretical={theo_snr:.4f}, Actual={act_snr:.4f}, Diff={act_snr-theo_snr:.4f}"
                )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot theoretical SNR curve
    t_values = np.arange(0, num_steps + 1)
    ax.plot(
        t_values, theoretical_snr, "b-", alpha=0.7, linewidth=2, label="Theoretical SNR"
    )

    # Plot actual SNR points
    ax.plot(actual_timesteps, actual_snr, "ro", markersize=2, label="Actual SNR")

    # Enhance plot appearance
    ax.set_title("Signal-to-Noise Ratio During Forward Diffusion", fontsize=12)
    ax.set_xlabel("Timestep (t)", fontsize=12)
    ax.set_ylabel("SNR (log scale)", fontsize=12)
    ax.set_yscale("log")  # Log scale for better visualization
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add annotations for key timesteps
    for i, t in enumerate([1, num_steps // 2, num_steps]):
        if t in actual_timesteps:
            idx = actual_timesteps.index(t)
            snr_val = actual_snr[idx]
            ax.annotate(
                f"t={t}\nSNR={snr_val:.2f}",
                xy=(t, snr_val),
                xytext=(10, 10 if i % 2 == 0 else -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            )

    # Set appropriate limits
    ax.set_xlim(-5, num_steps + 5)

    plt.tight_layout()

    # Save or show the plot
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "snr_plot.png", dpi=300)
        plt.close()
    else:
        plt.show()
