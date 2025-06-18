#!/usr/bin/env python
# coding: utf-8

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

import os
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


def run_forward_process(
    forward_diff: ForwardDiffusion,
    x0: Tensor,
    timesteps: Optional[List[int]] = None,
    verbose: bool = True,
) -> ForwardDiffusionResults:
    """
    Run forward diffusion process at specified timesteps.

    Args:
        forward_diff: The forward diffusion model
        x0: Original clean sample [batch_size, channels, seq_length]
        timesteps: List of timesteps to analyze (if None, will generate a subset)
        verbose: Whether to print progress information

    Returns:
        Dictionary mapping timesteps to analysis results
    """
    if timesteps is None:
        # Use a subset of timesteps from tmin to tmax
        tmin = forward_diff.tmin
        tmax = forward_diff.tmax

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
    results: ForwardDiffusionResults = {}

    if verbose:
        print("\n" + "=" * 70)
        print(f" STARTING DIFFUSION ANALYSIS (timesteps: {len(timesteps)}) ")
        print("=" * 70)

    # Generate noise once for consistency across timesteps
    # This ensures we're analyzing the same noise pattern at different timesteps
    # eps = torch.randn_like(x0)

    # Run diffusion analysis for each timestep
    for t in timesteps:
        if verbose:
            print(f"\nAnalyzing forward diffusion process at timestep {t}...")

        # Generate noise for this timestep
        eps = torch.randn_like(x0)

        # Run forward process at this timestep
        results[t] = run_forward_process_at_timestep(forward_diff, x0, t, eps)

        if verbose:
            print("\n" + "-" * 50 + "\n")

    return results


def run_forward_process_at_timestep(
    forward_diff: ForwardDiffusion,
    x0: Tensor,
    timestep: int,
    eps: Tensor,
) -> ForwardDiffusionResult:
    """
    Run forward diffusion process at a specific timestep.

    Args:
        forward_diff: The forward diffusion model
        x0: Original clean sample [batch_size, channels, seq_length]
        timestep: The timestep to analyze
        eps: Pre-generated noise tensor.

    Returns:
        Dictionary containing analysis results for the timestep
    """
    device = x0.device

    # Ensure x0 is on the correct device
    x0 = x0.to(device)

    # Convert timestep to tensor
    t_tensor = torch.tensor([timestep], device=device)

    # Apply forward diffusion to get x_t
    # q(x_t|x_0) = N(x_t; √(ᾱ_t) * x_0, (1-ᾱ_t)I)
    # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0, I)
    x_t = forward_diff.sample(x0, t_tensor, eps)

    # Get diffusion parameters using internal methods
    # β_t - noise schedule at timestep t
    beta_t = forward_diff.beta(t_tensor).item()

    # α_t = 1 - β_t - signal retention rate at timestep t
    alpha_t = forward_diff.alpha(t_tensor).item()

    # ᾱ_t = ∏_{s=1}^{t} α_s - cumulative product of alphas up to t
    alpha_bar_t = forward_diff.alpha_bar(t_tensor).item()

    # σ_t = √(1-ᾱ_t) - standard deviation of noise at timestep t
    sigma_t = forward_diff.sigma(t_tensor).item()

    # Calculate signal and noise components of x_t
    # Signal component: √(ᾱ_t) * x_0
    signal_component = torch.sqrt(torch.tensor(alpha_bar_t, device=device)) * x0

    # Noise component: σ_t * ε = √(1-ᾱ_t) * ε
    noise_component = torch.tensor(sigma_t, device=device) * eps

    # Signal-to-noise ratio (SNR): |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
    # This matches the literature definition (Nichol & Dhariwal, 2021)
    signal_power = torch.mean(signal_component**2).item()
    noise_power = torch.mean(noise_component**2).item()
    snr = signal_power / (noise_power + 1e-8) if alpha_bar_t < 1 else 1000.0

    # Calculate metrics for analysis
    # Average magnitude of noise component: |σ_t * ε|
    noise_magnitude = torch.mean(torch.abs(noise_component)).item()

    # Average magnitude of signal component: |√(ᾱ_t) * x_0|
    signal_magnitude = torch.mean(torch.abs(signal_component)).item()

    # Average magnitude of noisy sample: |x_t|
    x_t_magnitude = torch.mean(torch.abs(x_t)).item()

    # Return results
    return {
        "timestep": timestep,
        "x_t": x_t.detach().clone(),
        "noise": eps.detach().clone(),
        "alpha_t": alpha_t,
        "alpha_bar_t": alpha_bar_t,
        "sigma_t": sigma_t,
        "beta_t": beta_t,
        "snr": snr,
        "noise_magnitude": noise_magnitude,
        "signal_magnitude": signal_magnitude,
        "x_t_magnitude": x_t_magnitude,
        "signal_percentage": signal_power / (signal_power + noise_power) * 100,
        "noise_percentage": noise_power / (signal_power + noise_power) * 100,
    }


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

    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to those available in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    for t in timesteps:
        r = results[t]
        print(f"\nTimestep t = {t}:")
        print(f"  α_t = {r['alpha_t']:.8f}  (signal retention rate at step t)")
        print(f"  ᾱ_t = {r['alpha_bar_t']:.8f}  (cumulative product of alphas up to t)")
        print(
            f"  σ_t = {r['sigma_t']:.8f}  (noise standard deviation at step t = √(1-ᾱ_t))"
        )
        print(f"  β_t = {r['beta_t']:.8f}  (noise schedule at step t = 1-α_t)")
        print(f"  SNR = {r['snr']:.8f}  (signal-to-noise ratio = |√(ᾱ_t)*x_0|/|σ_t*ε|)")
        print(f"  Signal Magnitude = {r['signal_magnitude']:.8f}  (|√(ᾱ_t)*x_0|)")
        print(f"  Noise Magnitude = {r['noise_magnitude']:.8f}  (|σ_t*ε|)")
        print(
            f"  Signal Percentage = {r['signal_percentage']:.2f}%  (signal proportion in x_t)"
        )
        print(
            f"  Noise Percentage = {r['noise_percentage']:.2f}%  (noise proportion in x_t)"
        )


def save_forward_analysis(
    results: ForwardDiffusionResults,
    output_dir: Path,
    filename: str = "forward_analysis_results.csv",
) -> None:
    """Save forward analysis results to a CSV file.

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
        row.update(result["metrics"])
        rows.append(row)

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir / filename
    df.to_csv(csv_path, index=False)

    print(f"Saved forward analysis results to {csv_path}")


# ==================== Plotting Functions ====================


# Forward Diffusion Sample
def plot_forward_diffusion_sample(
    results: ForwardDiffusionResults,
    x0: Tensor,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot a sample going through the forward diffusion process at different timesteps.

    Args:
        x0: Original clean sample [batch_size, channels, seq_length]
        results: Results from analyze_forward_diffusion
        save_dir: Directory to save the plot (if None, display plot)
    """
    # Sort timesteps
    timesteps = sorted(results.keys())

    # Create figure with subplots
    n_plots = len(timesteps) + 1  # +1 for the original sample
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True)

    # Extract original sample data and ensure it's a numpy array
    # Handle dimensions: squeeze unnecessary dimensions and convert to numpy
    x0_np = x0.squeeze().detach().cpu().numpy()

    # Ensure x0_np is at least 1D
    if x0_np.ndim == 0:
        x0_np = np.array([x0_np])

    # Plot original sample with clear formatting
    axes[0].plot(x0_np, "b-", linewidth=2)
    axes[0].set_title("Original Sample (t=0)", fontweight="bold")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Get data range for consistent y-axis
    min_val = float(np.min(x0_np))
    max_val = float(np.max(x0_np))

    # Plot noisy samples at each timestep
    for i, t in enumerate(timesteps):
        x_t = results[t]["x_t"]
        # Handle dimensions: squeeze unnecessary dimensions and convert to numpy
        x_t_np = x_t.squeeze().detach().cpu().numpy()

        # Ensure x_t_np is at least 1D
        if x_t_np.ndim == 0:
            x_t_np = np.array([x_t_np])

        min_val = min(min_val, float(np.min(x_t_np)))
        max_val = max(max_val, float(np.max(x_t_np)))

        axes[i + 1].plot(x_t_np, "r-")
        axes[i + 1].set_title(f'Noisy Sample (t={t}), SNR={results[t]["snr"]:.4f}')
        axes[i + 1].set_ylabel("Value")
        axes[i + 1].grid(True, linestyle="--", alpha=0.7)

    # Set consistent y-axis limits with padding
    y_padding = (max_val - min_val) * 0.1
    for ax in axes:
        ax.set_ylim(min_val - y_padding, max_val + y_padding)

    axes[-1].set_xlabel("Position")
    fig.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "forward_diffusion_sample.png")
        plt.close()
    else:
        plt.show()


# Sample Evolution through Timesteps
def plot_sample_evolution(
    forward_diff: ForwardDiffusion,
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
        forward_diff: ForwardDiffusion instance
        results: Results from run_forward_process
        x0: Original clean sample
        timesteps: Timesteps to visualize (default: all available)
        save_dir: Save directory (default: display plot)
    """
    # Get timesteps to plot
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        timesteps = [t for t in timesteps if t in results]

    if not timesteps:
        print("No valid timesteps found.")
        return

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
        noise = results[t]["noise"].squeeze().detach().cpu().numpy()
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
        noise = results[t]["noise"].squeeze().detach().cpu().numpy()
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
        plt.savefig(save_path / "sample_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Signal Noise Ratio (SNR)
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
    theoretical_snr[0] = 1000.0  # Large value for t=0
    theoretical_snr[1:] = alpha_bar_values[1:] / (
        1 - alpha_bar_values[1:] + 1e-8
    )  # Add epsilon for numerical stability

    # Prepare actual SNR values from results
    actual_timesteps = [0] + timesteps  # Include t=0
    actual_snr = [1000.0]  # t=0 SNR (infinite, set to large value)

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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot theoretical SNR curve
    t_values = np.arange(0, num_steps + 1)
    ax.plot(
        t_values, theoretical_snr, "b-", alpha=0.7, linewidth=2, label="Theoretical SNR"
    )

    # Plot actual SNR points
    ax.plot(actual_timesteps, actual_snr, "ro", markersize=6, label="Measured SNR")

    # Enhance plot appearance
    ax.set_title("Signal-to-Noise Ratio During Forward Diffusion", fontsize=14)
    ax.set_xlabel("Timestep (t)", fontsize=12)
    ax.set_ylabel("SNR (log scale)", fontsize=12)
    ax.set_yscale("log")  # Log scale for better visualization
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

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
        plt.savefig(save_path / "snr_plot.png", dpi=300)
        plt.close()
    else:
        plt.show()


# Diffusion Parameters
def plot_diffusion_parameters(
    results: ForwardDiffusionResults,
    x0: Optional[Tensor] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot diffusion parameters across timesteps.

    Args:
        results: Results from analyze_forward_diffusion
        x0: Original clean sample [batch_size, channels, seq_length]
        save_dir: Directory to save the plot (if None, display plot)
    """
    # Sort timesteps
    timesteps = sorted(results.keys())

    # Extract parameters
    alpha_ts = [results[t]["alpha_t"] for t in timesteps]
    alpha_bar_ts = [results[t]["alpha_bar_t"] for t in timesteps]
    sigma_ts = [results[t]["sigma_t"] for t in timesteps]
    beta_ts = [results[t]["beta_t"] for t in timesteps]

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3)

    # Create time arrays for different parameters
    # Note: beta and alpha are defined for t=1 to T
    #       alpha_bar and sigma are defined for t=0 to T
    t_diff = np.array(timesteps)  # t=1 to T for beta, alpha
    t_cumul = np.array([0] + timesteps)  # t=0 to T for alpha_bar, sigma

    # Main parameter plot (spans two columns)
    ax_main = fig.add_subplot(gs[0, :2])
    lines = []
    # Plot parameters with distinct styles (only for t≥1)
    l1 = ax_main.plot(
        t_diff, beta_ts, "b-", marker="o", markersize=4, label="β (noise)"
    )
    l2 = ax_main.plot(
        t_diff, alpha_ts, "g-", marker="s", markersize=4, label="α (signal)"
    )
    lines.extend([l1[0], l2[0]])
    ax_main.set_xlabel("Timestep (t)")
    ax_main.set_ylabel("Parameter Value")
    ax_main.grid(True)
    ax_main.set_title("Diffusion Parameters Evolution (t=1→T)", pad=10)

    # Alpha bar plot (log scale)
    ax_alpha = fig.add_subplot(gs[0, 2])
    # For alpha_bar, include t=0 value (=1.0) and t=1 to T values
    alpha_bar_full = np.concatenate(([1.0], alpha_bar_ts))
    l3 = ax_alpha.semilogy(
        t_cumul, alpha_bar_full, "r-", marker="^", markersize=4, label="ᾱ (cumulative)"
    )
    lines.append(l3[0])
    ax_alpha.set_xlabel("Timestep (t)")
    ax_alpha.set_ylabel("ᾱ (log scale)")
    ax_alpha.grid(True)
    ax_alpha.set_title("Cumulative Signal (t=0→T)", pad=10)

    # Sigma plot
    ax_sigma = fig.add_subplot(gs[1, 0])
    # For sigma, include t=0 value (=0.0) and t=1 to T values
    sigma_full = np.concatenate(([0.0], sigma_ts))
    l4 = ax_sigma.plot(
        t_cumul, sigma_full, "m-", marker="x", markersize=4, label="σ (noise std)"
    )
    lines.append(l4[0])
    ax_sigma.set_xlabel("Timestep (t)")
    ax_sigma.set_ylabel("σ")
    ax_sigma.grid(True)
    ax_sigma.set_title("Noise Level (t=0→T)", pad=10)

    # Sample progression if x0 is provided
    if x0 is not None:
        ax_sample = fig.add_subplot(gs[1, 1:])
        x0_np = x0[0].squeeze().detach().cpu().numpy()
        if x0_np.ndim == 0:
            x0_np = np.array([x0_np])

        # Plot original and key samples
        ax_sample.plot(x0_np, "k-", label="Original", alpha=0.8)
        key_ts = [min(timesteps), timesteps[len(timesteps) // 2], max(timesteps)]
        for t in key_ts:
            if t in results:
                x_t = results[t]["x_t"].squeeze().detach().cpu().numpy()
                if x_t.ndim == 0:
                    x_t = np.array([x_t])
                ax_sample.plot(x_t, alpha=0.6, label=f"t={t}")
        ax_sample.set_title("Sample Evolution", pad=10)
        ax_sample.set_xlabel("Position")
        ax_sample.set_ylabel("Value")
        ax_sample.legend()
        ax_sample.grid(True)

    # Create a unified legend for all parameters
    fig.legend(
        lines,
        [l.get_label() for l in lines],
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
    )

    # Adjust layout
    plt.tight_layout()
    # Make room for the unified legend
    plt.subplots_adjust(right=0.85)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            save_path / "diffusion_parameters.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


# Animation Frames
def create_animation_frames(
    x0: Tensor,
    results: ForwardDiffusionResults,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Create animation frames showing the progressive addition of noise.

    Args:
        x0: Original clean sample [batch_size, channels, seq_length]
        results: Results from analyze_forward_diffusion
        save_dir: Directory to save the frames (required)
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified for create_animation_frames")

    # Create directory for frames
    save_path = Path(save_dir)
    frames_dir = save_path / "animation_frames"
    frames_dir.mkdir(exist_ok=True, parents=True)

    # Sort timesteps
    timesteps = sorted(results.keys())

    # Get data range for consistent y-axis
    # Handle dimensions: squeeze unnecessary dimensions and convert to numpy
    x0_np = x0.squeeze().detach().cpu().numpy()

    # Ensure x0_np is at least 1D
    if x0_np.ndim == 0:
        x0_np = np.array([x0_np])

    # Print information about the original sample
    print("\nCreating animation frames with original sample:")
    print(f"  - Shape: {x0_np.shape}")
    print(f"  - Unique values: {np.unique(x0_np)}")
    print(f"  - Non-zero count: {np.count_nonzero(x0_np)} out of {len(x0_np)}")

    # Collect all data points for y-axis limits
    min_val = float(np.min(x0_np))
    max_val = float(np.max(x0_np))

    for t in timesteps:
        x_t_np = results[t]["x_t"].squeeze().detach().cpu().numpy()
        if x_t_np.ndim == 0:
            x_t_np = np.array([x_t_np])

        min_val = min(min_val, float(np.min(x_t_np)))
        max_val = max(max_val, float(np.max(x_t_np)))

    # Add padding to y-axis limits
    y_min = min_val - 0.1
    y_max = max_val + 0.1

    # Create frame for original sample
    plt.figure(figsize=(10, 6))

    # Plot original sample with stem plot to highlight non-zero values
    plt.stem(
        np.arange(len(x0_np)),
        x0_np,
        linefmt="b-",
        markerfmt="bo",
        basefmt=" ",
        label="Original Sample (t=0)",
    )

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.title("Original Sample (t=0)", fontweight="bold", fontsize=14)
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig(frames_dir / "frame_000.png")
    plt.close()

    # Create frames for each timestep
    for i, t in enumerate(timesteps):
        x_t = results[t]["x_t"]
        # Handle dimensions: squeeze unnecessary dimensions and convert to numpy
        x_t_np = x_t.squeeze().detach().cpu().numpy()
        if x_t_np.ndim == 0:
            x_t_np = np.array([x_t_np])

        plt.figure(figsize=(10, 6))

        # Plot original clean sample for reference using stem plot
        plt.stem(
            np.arange(len(x0_np)),
            x0_np,
            linefmt="b-",
            markerfmt="bo",
            basefmt=" ",
            label="Original Sample (t=0)",
        )

        # Plot noisy sample at current timestep
        plt.plot(x_t_np, "r-", linewidth=1.5, alpha=0.8, label=f"Noisy Sample (t={t})")

        # Add a horizontal line at y=0 for reference
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.title(
            f'Forward Diffusion Process - Timestep {t} (SNR={results[t]["snr"]:.4f})',
            fontweight="bold",
            fontsize=14,
        )
        plt.xlabel("Position", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.ylim(y_min, y_max)

        # Save frame
        plt.savefig(frames_dir / f"frame_{i+1:03d}.png")
        plt.close()

    print("\nAnimation frames created successfully.")
    print(f"Frames saved to {frames_dir}")
    print("To create a video from these frames, you can use ffmpeg:")
    print(
        f"ffmpeg -framerate 2 -i {frames_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {save_path}/forward_diffusion.mp4"
    )
