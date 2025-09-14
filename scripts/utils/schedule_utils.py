#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""Utilities for analyzing diffusion schedules and parameters."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor

from src.forward_diffusion import ForwardDiffusion


# Analyze Schedule Parameters
def analyze_schedule_parameters(
    forward_diff: ForwardDiffusion,
    save_dir: Optional[Union[str, Path]] = None,
    schedule_type: str = "cosine",
) -> None:
    """Analyze and plot the diffusion schedule parameters across all timesteps.

    This function creates detailed plots of beta, alpha, alpha_bar, and sigma values
    across all timesteps, with special focus on the end of the schedule.

    Args:
        forward_diff: The forward diffusion model
        save_dir: Directory to save the plots (if None, display plot)
        schedule_type: Type of schedule being analyzed ('cosine' or 'linear')
    """

    print(f"Plotting {schedule_type} beta schedule parameters...")

    # Get schedule parameters
    betas = forward_diff.betas.cpu().numpy()
    alphas = forward_diff.alphas.cpu().numpy()
    alphas_bar = forward_diff.alphas_bar.cpu().numpy()
    sigmas = forward_diff.sigmas.cpu().numpy()

    # Create timestep arrays for different parameters
    # Note: beta and alpha are defined for t=1 to T,
    # alpha_bar and sigma are defined for t=0 to T
    t_diff = np.arange(1, len(alphas_bar))  # t=1 to T
    t_cumul = np.arange(0, len(alphas_bar))  # t=0 to T

    # Subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{schedule_type} schedule analysis", fontsize=16)

    # Plot 1: Beta values (t=1 to T)
    ax1.plot(t_diff, betas, "b-", label="β (noise schedule)", linewidth=2)
    ax1.set_title("β values (t=1→T)", fontsize=14)
    ax1.set_xlabel("timesteps", fontsize=12)
    ax1.set_ylabel("β", fontsize=12)
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=12)

    # Add inset for last few timesteps
    axins1 = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
    last_n = 10
    axins1.plot(t_diff[-last_n:], betas[-last_n:], "b-")
    axins1.set_title(f"Last {last_n} Steps")
    axins1.grid(True)

    # Plot 2: Alpha values (t=1 to T)
    ax2.plot(t_diff, alphas, "g-", label="α (signal preservation)", linewidth=2)
    ax2.set_title("α values (t=1→T)", fontsize=14)
    ax2.set_xlabel("timesteps", fontsize=12)
    ax2.set_ylabel("α", fontsize=12)
    ax2.grid(True)
    ax2.legend(loc="lower left", fontsize=12)

    # Add inset for last few timesteps
    axins2 = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins2.plot(t_diff[-last_n:], alphas[-last_n:], "g-")
    axins2.set_title(f"Last {last_n} Steps", fontsize=12)
    axins2.grid(True)

    # Plot 3: Alpha bar values with log scale (t=0 to T)
    ax3.plot(t_cumul, alphas_bar, "r-", label="ᾱ (cumulative signal)", linewidth=2)
    ax3.set_title("ᾱ values (t=0→T)", fontsize=14)
    ax3.set_xlabel("timesteps", fontsize=12)
    ax3.set_ylabel("ᾱ", fontsize=12)
    ax3.grid(True)
    ax3.legend(loc="lower left", fontsize=12)

    # Add inset for last few timesteps
    axins3 = ax3.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins3.plot(t_cumul[-last_n:], alphas_bar[-last_n:], "r-")
    axins3.set_title(f"Last {last_n} Steps", fontsize=12)
    axins3.grid(True)

    # Plot 4: Sigma values (t=0 to T)
    ax4.plot(t_cumul, sigmas, "m-", label="σ (noise level)", linewidth=2)
    ax4.set_title("σ values (t=0→T)", fontsize=14)
    ax4.set_xlabel("timesteps", fontsize=12)
    ax4.set_ylabel("σ", fontsize=12)
    ax4.grid(True)
    ax4.legend(loc="upper left", fontsize=12)

    # Add inset for last few timesteps
    axins4 = ax4.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins4.plot(t_cumul[-last_n:], sigmas[-last_n:], "m-")
    axins4.set_title(f"Last {last_n} Steps", fontsize=12)
    axins4.grid(True)

    # Print statistics for the final few timesteps
    print(f"Printing {schedule_type} beta schedule, final {last_n} timesteps:")
    for i in range(-last_n, 0):
        t_idx = len(betas) + i
        print(
            f"\nt={t_idx+1}, β_t: {betas[t_idx]:.8f}, α_t: {alphas[t_idx]:.8f}, ᾱ_t: {alphas_bar[t_idx+1]:.8f}, σ_t: {sigmas[t_idx+1]:.8f}"
        )

        # if i <= -1:
        # Calculate relative changes
        alpha_bar_change = (
            (alphas_bar[t_idx + 1] - alphas_bar[t_idx]) / alphas_bar[t_idx] * 100
        )
        sigma_change = (sigmas[t_idx + 1] - sigmas[t_idx]) / sigmas[t_idx] * 100
        print(f"relative change in ᾱ_t: {alpha_bar_change:.8f}%")
        print(f"relative change in σ_t: {sigma_change:.8f}%")

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_path / f"{schedule_type}_schedule.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


# Print Schedule Parameters
def print_schedule_parameters(
    forward_diff: "ForwardDiffusion",
    save_dir: Optional[Union[str, Path]] = None,
    schedule_type: str = "cosine",
) -> None:
    """
    Print and optionally save all schedule parameters (beta, alpha, alpha_bar, sigma) for all timesteps.

    Args:
        forward_diff: The forward diffusion model
        save_dir: Directory to save the CSV file (if None, only print)
        schedule_type: Type of schedule being analyzed ('cosine' or 'linear')
    """

    # Get schedule parameters
    betas = forward_diff.betas.cpu().numpy()
    alphas = forward_diff.alphas.cpu().numpy()
    alphas_bar = forward_diff.alphas_bar.cpu().numpy()
    sigmas = forward_diff.sigmas.cpu().numpy()

    # Timesteps: beta/alpha are t=1..T, alpha_bar/sigma are t=0..T
    T = len(betas)
    print(f"\n{schedule_type.title()} Schedule Parameters:")
    print(f"{'t':>5}  {'beta':>12}  {'alpha':>12}  {'alpha_bar':>16}  {'sigma':>16}")
    rows = []
    for t in range(1, T + 1):
        # t: 1-based index for beta/alpha, t for alpha_bar/sigma is t
        row = [
            t,
            betas[t - 1],
            alphas[t - 1],
            alphas_bar[t],
            sigmas[t],
        ]
        print(
            f"{t:5d}  {betas[t-1]:12.10f}  {alphas[t-1]:12.10f}  {alphas_bar[t]:16.10f}  {sigmas[t]:16.10f}"
        )
        rows.append(row)

    # Save as CSV if requested
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        csv_path = save_path / f"{schedule_type}_schedule_parameters.csv"
        df = pd.DataFrame(rows, columns=["t", "beta", "alpha", "alpha_bar", "sigma"])
        df.to_csv(csv_path, index=False)
        print(f"\nSaved schedule parameters to: {csv_path}")


# Plot Schedule Comparison (Theoretical vs Actual)
def plot_schedule_comparison(
    forward_diff: ForwardDiffusion,
    results: Dict[int, Dict[str, Union[Tensor, float]]],
    save_dir: Optional[Union[str, Path]] = None,
    schedule_type: str = "cosine",
) -> None:
    """Compare theoretical schedule parameters with actual values during forward diffusion.

    Args:
        forward_diff: The forward diffusion model
        results: Results from forward diffusion process
        save_dir: Directory to save plots (if None, display plot)
        schedule_type: Type of schedule being analyzed ('cosine' or 'linear')
    """

    print(f"Plotting {schedule_type} beta schedule comparison...")

    # Get theoretical parameters from schedule
    theoretical_params = {
        "betas": forward_diff.betas.cpu().numpy(),
        "alphas": forward_diff.alphas.cpu().numpy(),
        "alphas_bar": forward_diff.alphas_bar.cpu().numpy(),
        "sigmas": forward_diff.sigmas.cpu().numpy(),
    }

    # Extract actual parameters from results
    actual_params = {
        "timesteps": sorted(results.keys()),
        "betas": [results[t]["beta_t"] for t in sorted(results.keys())],
        "alphas": [results[t]["alpha_t"] for t in sorted(results.keys())],
        "alphas_bar": [results[t]["alpha_bar_t"] for t in sorted(results.keys())],
        "sigmas": [results[t]["sigma_t"] for t in sorted(results.keys())],
    }

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Schedule vs Actual Parameters ({schedule_type})", fontsize=16)

    # Plot beta comparison
    ax1.plot(
        np.arange(1, len(theoretical_params["betas"]) + 1),
        theoretical_params["betas"],
        "b-",
        label="Schedule β",
    )
    ax1.plot(actual_params["timesteps"], actual_params["betas"], "r.", label="Actual β")
    ax1.set_title("Beta Values")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("β")
    ax1.legend()
    ax1.grid(True)

    # Plot alpha comparison
    ax2.plot(
        np.arange(1, len(theoretical_params["alphas"]) + 1),
        theoretical_params["alphas"],
        "b-",
        label="Schedule α",
    )
    ax2.plot(
        actual_params["timesteps"], actual_params["alphas"], "r.", label="Actual α"
    )
    ax2.set_title("Alpha Values")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("α")
    ax2.legend()
    ax2.grid(True)

    # Plot alpha_bar comparison (log scale)
    ax3.semilogy(
        np.arange(0, len(theoretical_params["alphas_bar"])),
        theoretical_params["alphas_bar"],
        "b-",
        label="Schedule ᾱ",
    )
    ax3.semilogy(
        actual_params["timesteps"], actual_params["alphas_bar"], "r.", label="Actual ᾱ"
    )
    ax3.set_title("Alpha Bar Values")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("ᾱ (log scale)")
    ax3.legend()
    ax3.grid(True)

    # Plot sigma comparison
    ax4.plot(
        np.arange(0, len(theoretical_params["sigmas"])),
        theoretical_params["sigmas"],
        "b-",
        label="Schedule σ",
    )
    ax4.plot(
        actual_params["timesteps"], actual_params["sigmas"], "r.", label="Actual σ"
    )
    ax4.set_title("Sigma Values")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("σ")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            save_path / f"{schedule_type}_schedule_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


# Print Schedule Comparison (Theoretical vs Actual)
def print_schedule_comparison(
    forward_diff: ForwardDiffusion,
    results: Dict[int, Dict[str, Union[Tensor, float]]],
    timesteps: Optional[List[int]] = None,
) -> None:
    """Print detailed comparison of schedule vs actual parameters.

    Args:
        forward_diff: The forward diffusion model
        results: Results from forward diffusion process
        timesteps: List of timesteps to compare (if None, use all timesteps in results)
    """

    print("\nSchedule parameter comparison at selected timesteps:")

    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    for t in timesteps:
        print(f"\nTimestep t = {t}:")
        print(
            f"  β_t:  Schedule={forward_diff.betas[t-1].item():.8f}, "
            f"Actual={results[t]['beta_t']:.8f}, "
            f"Diff={forward_diff.betas[t-1].item()-results[t]['beta_t']:.8f}"
        )
        print(
            f"  α_t:  Schedule={forward_diff.alphas[t-1].item():.8f}, "
            f"Actual={results[t]['alpha_t']:.8f}, "
            f"Diff={forward_diff.alphas[t-1].item()-results[t]['alpha_t']:.8f}"
        )
        print(
            f"  ᾱ_t:  Schedule={forward_diff.alphas_bar[t].item():.8f}, "
            f"Actual={results[t]['alpha_bar_t']:.8f}, "
            f"Diff={forward_diff.alphas_bar[t].item()-results[t]['alpha_bar_t']:.8f}"
        )
        print(
            f"  σ_t:  Schedule={forward_diff.sigmas[t].item():.8f}, "
            f"Actual={results[t]['sigma_t']:.8f}, "
            f"Diff={forward_diff.sigmas[t].item()-results[t]['sigma_t']:.8f}"
        )
