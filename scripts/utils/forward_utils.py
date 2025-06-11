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

    # Signal-to-noise ratio (SNR): |√(ᾱ_t) * x_0| / |√(1-ᾱ_t) * ε|
    # As t increases, ᾱ_t decreases and SNR decreases
    snr = torch.mean(
        torch.abs(signal_component) / (torch.abs(noise_component) + 1e-8)
    ).item()

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
        "signal_percentage": signal_magnitude
        / (signal_magnitude + noise_magnitude)
        * 100,
        "noise_percentage": noise_magnitude
        / (signal_magnitude + noise_magnitude)
        * 100,
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

    print("\n" + "=" * 80)
    print("FORWARD DIFFUSION STATISTICS")
    print("=" * 80)

    for t in timesteps:
        r = results[t]
        print(f"\nTimestep t = {t}:")
        print(f"  α_t = {r['alpha_t']:.6f}  (signal retention rate at step t)")
        print(f"  ᾱ_t = {r['alpha_bar_t']:.6f}  (cumulative product of alphas up to t)")
        print(
            f"  σ_t = {r['sigma_t']:.6f}  (noise standard deviation at step t = √(1-ᾱ_t))"
        )
        print(f"  β_t = {r['beta_t']:.6f}  (noise schedule at step t = 1-α_t)")
        print(f"  SNR = {r['snr']:.6f}  (signal-to-noise ratio = |√(ᾱ_t)*x_0|/|σ_t*ε|)")
        print(f"  Signal Magnitude = {r['signal_magnitude']:.6f}  (|√(ᾱ_t)*x_0|)")
        print(f"  Noise Magnitude = {r['noise_magnitude']:.6f}  (|σ_t*ε|)")
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


# Signal Noise Ratio
def plot_signal_noise_ratio(
    results: ForwardDiffusionResults,
    x0: Optional[Tensor] = None,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> None:
    """
    Plot signal-to-noise ratio and other metrics across timesteps.

    Args:
        results: Results from run_forward_process
        x0: Original clean sample [batch_size, channels, seq_length]
        save_dir: Directory to save the plot (if None, display plot)
        verbose: Whether to print additional information
    """
    # Sort timesteps
    timesteps = sorted(results.keys())

    # Add t=0 (pure signal) to the visualization
    # For t=0, we have 100% signal and 0% noise
    all_timesteps = [0] + timesteps

    # For t=0, SNR is theoretically infinite, but we'll use a large value for visualization
    # For signal and noise percentages, t=0 is 100% signal, 0% noise
    snrs = [1000.0] + [results[t]["snr"] for t in timesteps]
    signal_percentages = [100.0] + [results[t]["signal_percentage"] for t in timesteps]
    noise_percentages = [0.0] + [results[t]["noise_percentage"] for t in timesteps]

    # Print numerical values for comparison, especially near t=1000
    if verbose:
        print("\nTimesteps for visualization:")
        print(f"  All timesteps: {all_timesteps}")
        print(f"  Signal percentages: {[round(p, 1) for p in signal_percentages]}")
        print(f"  Noise percentages: {[round(p, 1) for p in noise_percentages]}")

        # Print detailed comparison between t=0 and t=1
        print("\nComparison between t=0 (pure signal) and t=1 (first diffusion step):")
        print(
            f"  t=0: Signal={signal_percentages[0]:.2f}%, Noise={noise_percentages[0]:.2f}%, SNR={snrs[0]:.2f}"
        )
        print(
            f"  t=1: Signal={signal_percentages[1]:.2f}%, Noise={noise_percentages[1]:.2f}%, SNR={snrs[1]:.2f}"
        )
        print(
            f"  Difference: Signal={signal_percentages[0]-signal_percentages[1]:.2f}%, Noise={noise_percentages[1]-noise_percentages[0]:.2f}%, SNR={snrs[0]-snrs[1]:.2f}"
        )

        # Print values for the last few timesteps to examine the drop near t=1000
        print("\nValues for the last few timesteps:")
        last_indices = [-5, -4, -3, -2, -1]  # Last 5 timesteps
        for i in last_indices:
            t = all_timesteps[i]
            print(
                f"  t={t}: Signal={signal_percentages[i]:.4f}%, Noise={noise_percentages[i]:.4f}%, SNR={snrs[i]:.6f}"
            )

    # Create figure with subplots
    if x0 is not None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 16))
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot SNR
    axs[0].plot(all_timesteps, snrs, "b-", marker="o")
    axs[0].set_title("Signal-to-Noise Ratio (SNR) vs. Timestep")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("SNR")
    axs[0].set_yscale("log")  # Log scale for better visualization
    axs[0].grid(True)

    # Add vertical line at t=1 to highlight difference between t=0 and t=1
    axs[0].axvline(
        x=1, color="r", linestyle="--", alpha=0.7, label="t=1 (First diffusion step)"
    )

    # Highlight t=0 and t=1 points
    axs[0].plot(0, snrs[0], "ro", markersize=8, label="t=0 (Pure signal)")
    axs[0].plot(1, snrs[1], "go", markersize=8, label="t=1")

    # Set x-axis limits
    axs[0].set_xlim(-5, max(all_timesteps) + 5)  # Add some padding

    # Add legend
    axs[0].legend()

    # Plot signal and noise percentages
    axs[1].stackplot(
        all_timesteps,
        signal_percentages,
        noise_percentages,
        labels=["Signal %", "Noise %"],
        alpha=0.7,
    )

    # Use consistent x-axis limits with the first plot
    axs[1].set_xlim(-5, max(all_timesteps) + 5)  # Same padding as first plot

    # Add vertical line at t=1 in the second plot as well
    axs[1].axvline(
        x=1, color="r", linestyle="--", alpha=0.7, label="t=1 (First diffusion step)"
    )

    # Highlight t=0 and t=1 points
    axs[1].plot(0, signal_percentages[0], "ro", markersize=8, label="t=0 (Pure signal)")
    axs[1].plot(1, signal_percentages[1], "go", markersize=8, label="t=1")

    axs[1].set_title("Signal and Noise Percentage vs. Timestep")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Percentage (%)")
    axs[1].set_ylim(0, 100)
    axs[1].legend()
    axs[1].grid(True)

    # Plot sample progression if x0 is provided
    if x0 is not None:
        # Get original sample
        # Handle dimensions: squeeze unnecessary dimensions and convert to numpy
        x0_np = x0.squeeze().detach().cpu().numpy()

        # Ensure x0_np is at least 1D
        if x0_np.ndim == 0:
            x0_np = np.array([x0_np])

        # Plot original sample
        axs[2].plot(x0_np, "b-", label="Original (t=0)")

        # Plot a few key timesteps
        key_timesteps = [
            min(timesteps),
            timesteps[len(timesteps) // 4],
            timesteps[len(timesteps) // 2],
            timesteps[3 * len(timesteps) // 4],
            max(timesteps),
        ]

        for t in key_timesteps:
            x_t_np = results[t]["x_t"].squeeze().detach().cpu().numpy()
            if x_t_np.ndim == 0:
                x_t_np = np.array([x_t_np])
            axs[2].plot(x_t_np, alpha=0.7, label=f"t={t}")

        axs[2].set_title("Sample Progression at Key Timesteps")
        axs[2].set_xlabel("Position")
        axs[2].set_ylabel("Value")
        axs[2].legend()
        axs[2].grid(True)

    fig.tight_layout(pad=0.5)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "signal_noise_ratio.png", dpi=300)
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

    # Create figure with subplots
    if x0 is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot parameters
    ax1.plot(timesteps, alpha_ts, "b-", marker="o", markersize=4, label="α_t")
    ax1.plot(timesteps, alpha_bar_ts, "g-", marker="s", markersize=4, label="ᾱ_t")
    ax1.plot(timesteps, sigma_ts, "r-", marker="^", markersize=4, label="σ_t")
    ax1.plot(timesteps, beta_ts, "c-", marker="x", markersize=4, label="β_t")

    ax1.set_title("Diffusion Parameters vs. Timestep")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Parameter Value")
    ax1.legend()
    ax1.grid(True)

    # Plot sample progression if x0 is provided
    if x0 is not None:
        # Get original sample
        x0_np = x0[0].squeeze().detach().cpu().numpy()

        # Ensure x0_np is at least 1D
        if x0_np.ndim == 0:
            x0_np = np.array([x0_np])

        # Plot original sample
        ax2.plot(x0_np, "b-", label="Original (t=0)")

        # Plot samples at key parameter values
        key_timesteps = [
            min(timesteps),
            timesteps[len(timesteps) // 4],
            timesteps[len(timesteps) // 2],
            timesteps[3 * len(timesteps) // 4],
            max(timesteps),
        ]

        for t in key_timesteps:
            if t in results:
                x_t_np = results[t]["x_t"].squeeze().detach().cpu().numpy()
                if x_t_np.ndim == 0:
                    x_t_np = np.array([x_t_np])
                ax2.plot(
                    x_t_np, alpha=0.7, label=f't={t}, α_t={results[t]["alpha_t"]:.4f}'
                )

        ax2.set_title("Sample Progression with Diffusion Parameters")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

    fig.tight_layout(pad=0.5)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path / "diffusion_parameters.png", dpi=300)
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
