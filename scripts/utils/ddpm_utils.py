#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
DDPM Markov Chain Utilities
---------------------------
Utilities for analyzing the reverse diffusion Markov chain during inference.

This module provides:
- get_noisy_sample(): add forward-diffusion noise at a specified timestep T (1-indexed)
- run_denoising_process(): run the reverse diffusion chain from x_T → x_o and optionally
  print global and masked metrics per step
- plot helpers to visualize denoising behavior

As reverse diffusion progresses (t → 0):
- MSE(x_{t-1}, x_0) typically decreases (samples approach the clean signal)
- MSE(x_{t-1}, x_t) typically increases (samples diverge from the noisy input)
- r(x_{t-1}, x_0) typically increases (correlation to the clean signal strengthens)
- r(x_{t-1}, x_t) typically decreases (correlation to the noisy input weakens)
"""

from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    # Only imported for type hints; avoids runtime import cycles
    from src.ddpm import DiffusionModel


def torch_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array (detached, moved to CPU)."""
    if tensor is None:
        return None
    return tensor.cpu().detach().numpy()


def compute_metrics(a: Tensor, b: Tensor) -> tuple[float, float]:
    """Compute elementwise MSE and Pearson r between two tensors.

    Both inputs may be of any broadcastable shape; for the correlation we flatten
    both to 1D vectors. If correlation is undefined (fewer than 2 elements or
    zero variance in either vector), r is returned as NaN.
    Returns a tuple (mse, r).
    """
    mse = torch.mean((a - b) ** 2).item()

    u = a.flatten()
    v = b.flatten()

    # Safety checks for correlation
    if u.numel() < 2 or v.numel() < 2:
        r = float("nan")
    elif torch.std(u) == 0 or torch.std(v) == 0:
        r = float("nan")
    else:
        try:
            r = torch.corrcoef(torch.stack([u, v]))[0, 1].item()
        except Exception:
            r = float("nan")
    return mse, r


def _compute_masked_metrics(a: Tensor, b: Tensor, mask: Tensor) -> tuple[float, float]:
    """Compute MSE and Pearson r between a and b under a boolean mask.

    Args:
        a: Tensor of shape [B, C, L] (or broadcastable to mask shape).
        b: Tensor of the same shape as a.
        mask: Boolean tensor where True indicates elements to include. Expected
              shape [1, 1, L] or broadcastable to a/b.

    Returns:
        (mse, r): Metrics computed only over masked elements. r is NaN if
        undefined (fewer than 2 elements or zero variance).
    """
    # Broadcast mask to a/b shape and select elements
    sel = mask.expand_as(a)
    a_sel = a[sel]
    b_sel = b[sel]

    if a_sel.numel() == 0:
        return float("nan"), float("nan")

    return compute_metrics(a_sel, b_sel)


def print_log_metrics(
    step_label: str,
    x_t_minus_1: Tensor,
    x_0: Tensor,
    x_t: Tensor,
    mixing_mask: Optional[Tensor] = None,
) -> None:
    """Print a single-line summary of metrics for both x_{t-1} vs x_0 and x_{t-1} vs x_t.

    Args:
        step_label: String to prefix the log line, e.g. "t=1000" or f"t={t-1}".
        x_t_minus_1: Current denoised estimate at this step (x_{t-1}), shape [B,C,L].
        x_0: Ground-truth clean sample, shape [B,C,L].
        x_t: Initial noisy sample x_T, shape [B,C,L].
        mixing_mask: Optional boolean mask [1,1,L]; True=staircase, False=real.
    """
    # Total metrics
    mse_x0_total, r_x0_total = compute_metrics(x_t_minus_1, x_0)
    mse_xt_total, r_xt_total = compute_metrics(x_t_minus_1, x_t)

    if mixing_mask is None:
        # Single-line without masked breakdowns
        print(
            f"{step_label} | MSE(x_t_minus_1, x_0): {mse_x0_total:.6f}, r(x_t_minus_1, x_0): {r_x0_total:.6f}"
            # + " | "
            # f"MSE(x_t_minus_1, x_t): {mse_xt_total:.6f}, r(x_t_minus_1, x_t): {r_xt_total:.6f}"
        )
        return

    # With masked breakdowns (staircase, real)
    mm = mixing_mask.bool()
    mse_x0_stair, r_x0_stair = _compute_masked_metrics(x_t_minus_1, x_0, mm)
    mse_x0_real, r_x0_real = _compute_masked_metrics(x_t_minus_1, x_0, ~mm)
    mse_xt_stair, r_xt_stair = _compute_masked_metrics(x_t_minus_1, x_t, mm)
    mse_xt_real, r_xt_real = _compute_masked_metrics(x_t_minus_1, x_t, ~mm)

    print(
        f"{step_label} | "
        f"MSE(x_t_minus_1, x_0): {mse_x0_total:.6f} ({mse_x0_stair:.6f}, {mse_x0_real:.6f}), "
        f"r(x_t_minus_1, x_0): {r_x0_total:.6f} ({r_x0_stair:.6f}, {r_x0_real:.6f})"
        # + ", | "
        # f"MSE(x_t_minus_1, x_t): {mse_xt_total:.6f} ({mse_xt_stair:.6f}, {mse_xt_real:.6f}), "
        # f"r(x_t_minus_1, x_t): {r_xt_total:.6f} ({r_xt_stair:.6f}, {r_xt_real:.6f})"
    )


# Utility: Add noise to x_0 at a given timestep
def get_noisy_sample(
    model: "DiffusionModel", x_0: Tensor, t: int, eps: Optional[Tensor] = None
) -> Tensor:
    """Add Gaussian noise to x_0 at timestep t using the model’s forward diffusion.

    Timestep indexing is 1-based (consistent with the training sampler).

    Args:
        model: Diffusion model exposing forward_diffusion.sample.
        x_0: Original clean sample (shape [B, ...]).
        t: Timestep at which to add noise (1-indexed).
        eps: Optional pre-sampled noise. If None, standard normal noise is used.

    Returns:
        The noisy sample x_t at timestep t.
    """
    assert isinstance(t, int), f"t must be int, got {type(t)}"
    assert t >= 1, f"t must be >= 1, got {t}"
    device = x_0.device
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    if eps is None:
        eps = torch.randn_like(x_0)
    return model.forward_diffusion.sample(x_0, t_tensor, eps)


# Denoising Wrapper: Denoising from Noisy x_t at T
def run_denoising_process(
    model: "DiffusionModel",
    x_0: Tensor,  # Original clean sample
    x_t: Tensor,  # Noisy sample at T
    T: int,  # Start timestep for denoising process
    device: torch.device,
    return_all_steps: bool = True,
    print_mse: bool = False,
    true_x0: Optional[Tensor] = None,  # Ground-truth for imputation
    imputation_mask: Optional[
        Tensor
    ] = None,  # Mask for imputation (1=known, 0=unknown)
    mixing_mask: Optional[
        Tensor
    ] = None,  # Boolean mask [1,1,L]: True=staircase, False=real
) -> Dict[int, Tensor]:
    """Run reverse diffusion from x_T to x_0 and (optionally) log stepwise metrics.

    Args:
        model: Diffusion model providing reverse_diffusion.reverse_diffusion_step.
        x_0: Clean target sample used only for metrics, shape [B,C,L].
        x_t: Noisy starting sample at timestep T, shape [B,C,L].
        T: Starting timestep (1-indexed). The loop runs t=T,T-1,...,1.
        device: Device for tensor ops.
        return_all_steps: If True, returns all intermediate samples {t: x_{t}}.
        print_mse: If True, prints global and (optionally) masked MSE and r at each step.
        true_x0: Optional ground-truth clean sample for imputation blending.
        imputation_mask: Optional mask [B,C,L] where 1=known (use true_x0), 0=unknown (use prediction).
        mixing_mask: Optional boolean mask [1,1,L] where True=staircase, False=real, used only for metrics.

    Returns:
        Dict[int, Tensor]: Dictionary mapping timesteps to samples. If return_all_steps
        is False, returns only {0: x_0}.
    """
    model.eval()
    with torch.no_grad():
        # Store all samples in a dictionary, starting with x_t_start
        samples = {T: x_t.clone()}
        x_t_minus_1 = x_t  # Current sample in the chain

        if print_mse:
            print_log_metrics(f"t={T}", x_t_minus_1, x_0, x_t, mixing_mask)

        # Reverse diffusion: from t_start down to 1 (after loop x_curr will be x_0)
        for t in range(T, 0, -1):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            x_t_minus_1 = model.reverse_diffusion.reverse_diffusion_step(
                x_t_minus_1,
                t_tensor,
                true_x0=true_x0,
                imputation_mask=imputation_mask,
                return_all=False,
            )
            samples[t - 1] = x_t_minus_1.clone()

            # Print final MSEs
            if print_mse:
                print_log_metrics(f"t={t-1}", x_t_minus_1, x_0, x_t, mixing_mask)

    if return_all_steps:
        return samples
    else:
        return {0: samples[0]}


# Denoising Comparison Plots
def plot_denoising_comparison(x_0, x_t, x_t_minus_1, T, output_path):
    """
    Plots and saves a comparison of original, noisy, and denoised signals using matplotlib OOP API.
    Left: x_0 and x_t; Right: x_0, x_t, x_t_minus_1. Annotates MSE and correlation metrics.
    Args:
        x_0 (Tensor): Original clean sample
        x_t (Tensor): Noisy sample at timestep T
        x_t_minus_1 (Tensor): Denoised sample
        T (int): Timestep
        output_path (Path): Directory to save plots
    Returns:
        Tuple[float, float, float, float]: (MSE(x_t_minus_1, x_0), r(x_t_minus_1, x_0), MSE(x_t_minus_1, x_t), r(x_t_minus_1, x_t))
    """

    x0_np = torch_to_numpy(x_0).flatten()
    x_t_np = torch_to_numpy(x_t).flatten()
    x_t_minus_1_np = torch_to_numpy(x_t_minus_1).flatten()

    # Limit visualization to first 100 SNPs for performance
    max_snps_to_plot = min(100, len(x0_np))
    x0_np = x0_np[:max_snps_to_plot]
    x_t_np = x_t_np[:max_snps_to_plot]
    x_t_minus_1_np = x_t_minus_1_np[:max_snps_to_plot]

    mse_x0 = np.mean((x0_np - x_t_minus_1_np) ** 2)
    corr_x0 = np.corrcoef(x0_np, x_t_minus_1_np)[0, 1]
    mse_xt = np.mean((x_t_np - x_t_minus_1_np) ** 2)
    corr_xt = np.corrcoef(x_t_np, x_t_minus_1_np)[0, 1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left subplot: x_0 and x_t
    axs[0].plot(x0_np, label=r"Original $x_0$", color="blue")
    axs[0].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[0].set_title(f"Original vs Noisy (T={T})")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Value")
    if T <= 100:
        axs[0].set_ylim(-1, 1)
    else:
        axs[0].set_ylim(-3, 3)
    axs[0].legend()

    # Right subplot: x_0, x_t, x0_recon
    axs[1].plot(x0_np, label=r"Original $x_0$", color="blue")
    # axs[1].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[1].plot(x_t_minus_1_np, label=r"Denoised $x_{t-1}$", color="green", alpha=0.7)
    axs[1].set_title(f"Denoising Comparison (T={T})")
    axs[1].set_xlabel("SNP Position (first 100)")
    axs[1].legend()

    # Annotate MSE and correlation
    axs[1].annotate(
        f"MSE ($x_{{t-1}}$, $x_0$): {mse_x0:.6f}, r ($x_{{t-1}}$, $x_0$): {corr_x0:.6f}\nMSE ($x_{{t-1}}$, $x_t$): {mse_xt:.6f}, r ($x_{{t-1}}$, $x_t$): {corr_xt:.6f}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=10,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(
        str(output_path / f"denoising_comparison_t{T}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        str(output_path / f"denoising_comparison_t{T}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return mse_x0, corr_x0, mse_xt, corr_xt


def plot_denoising_trajectory(x_0, x_t, samples_dict, T, output_path):
    """
    Plots the denoising trajectory of x0_recon at each timestep during Markov reverse diffusion.
    Args:
        x_0: The original clean sample (Tensor or np.ndarray)
        x_t: The noisy sample at timestep T (Tensor or np.ndarray)
        samples_dict: Dict of denoised samples at each step (shape: [num_steps, seq_len])
        T: The starting timestep (int, for labeling)
        output_path: Path to save the plot
    """
    x0_np = torch_to_numpy(x_0).flatten()
    x_t_np = torch_to_numpy(x_t).flatten()

    # Limit visualization to first 100 SNPs for performance
    max_snps_to_plot = min(100, len(x0_np))
    x0_np = x0_np[:max_snps_to_plot]
    x_t_np = x_t_np[:max_snps_to_plot]

    # Prepare x0_recon_arr and timesteps from samples_dict (preserve insertion order)
    timesteps = list(samples_dict.keys())
    x0_recon_arr = np.stack(
        [
            torch_to_numpy(samples_dict[t]).flatten()[:max_snps_to_plot]
            for t in timesteps
        ],
        axis=0,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left subplot: x_0 and x_t
    axs[0].plot(x0_np, label=r"Original $x_0$", color="blue")
    axs[0].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[0].set_title(f"Original vs Noisy (T={T})")
    axs[0].set_xlabel("SNP Position (first 100)")
    axs[0].legend()
    axs[0].set_ylabel("Value")
    axs[0].set_ylim(axs[0].get_ylim())  # Match y-axis

    # Right subplot: all x0_recon
    # axs[1].plot(x_t_np, "m--", label=r"Noisy $x_t$", alpha=0.6)
    for i, x0_recon in enumerate(x0_recon_arr):
        x0_recon_flat = x0_recon.flatten()
        if timesteps[i] < 10:
            axs[1].plot(x0_recon_flat, label=f"Step {timesteps[i]}", alpha=0.7)

    axs[1].set_title(f"Denoising Trajectory (T={T})")
    axs[1].set_xlabel("SNP Position (first 100)")
    axs[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        str(output_path / f"denoising_trajectory_t{T}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        str(output_path / f"denoising_trajectory_t{T}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# === Locality Analysis ===
def compute_locality_metrics(values, outputs, snp_index):
    """Compute on-target and off-target metrics for locality analysis.

    Args:
        values (np.ndarray): Input values at target SNP
        outputs (np.ndarray): Model outputs for all SNPs
        snp_index (int): Index of the target SNP

    Returns:
        dict: Dictionary containing all metrics
    """
    from sklearn.metrics import mean_squared_error, r2_score

    # On-target metrics
    on_target_output = outputs[:, snp_index]
    metrics = {
        "on_target": {
            "slope": np.polyfit(values, on_target_output, 1)[0],
            "correlation": np.corrcoef(values, on_target_output)[0, 1],
            "mse": mean_squared_error(values, on_target_output),
            "r2": r2_score(values, on_target_output),
        }
    }

    # Off-target metrics
    off_target = np.delete(outputs, snp_index, axis=1)
    metrics["off_target"] = {
        "mean_abs": np.mean(np.abs(off_target)),
        "max_abs": np.max(np.abs(off_target)),
        "total_energy": np.sum(off_target**2),
        "std_dev": np.std(off_target),
        "regional_means": compute_regional_metrics(off_target),
    }

    return metrics


def compute_regional_metrics(off_target):
    """Compute metrics for each region of SNPs.

    Args:
        off_target (np.ndarray): Off-target effects array

    Returns:
        dict: Dictionary containing metrics for each region
    """
    regions = {
        "region_0": (0, 25),  # First region (0)
        "region_025": (25, 75),  # Middle region (0.25)
        "region_05": (75, 100),  # Last region (0.5)
    }

    return {
        name: {
            "mean": np.mean(np.abs(off_target[:, start:end])),
            "max": np.max(np.abs(off_target[:, start:end])),
            "energy": np.sum(off_target[:, start:end] ** 2),
        }
        for name, (start, end) in regions.items()
    }


def plot_locality_analysis(values, outputs, snp_index, output_dir):
    """Create all locality analysis plots.

    Args:
        values (np.ndarray): Input values at target SNP
        outputs (np.ndarray): Model outputs for all SNPs
        snp_index (int): Index of the target SNP
        output_dir (Path): Directory to save plots
    """
    # Line plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot truncated outputs (first 100 SNPs
    truncated_outputs = outputs[:, :100]
    for i, val in enumerate(values):
        label = f"SNP 60 = {val:.2f}" if i % 2 == 0 else None
        ax.plot(truncated_outputs[i], alpha=0.7, label=label)

    ax.axvline(snp_index, color="red", linestyle="--", label="SNP 60")
    ax.set_xlabel("SNP Position")
    ax.set_ylabel("Output Value")
    ax.set_title("Output at all SNPs for each SNP 60 value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=1)
    plt.tight_layout()
    plt.savefig(output_dir / "snp60_locality_lines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(values, outputs[:, snp_index], label="Model Output", color="blue")
    ax.plot(values, values, "k--", label="Ideal (y=x)")
    ax.set_xlabel("Input value at SNP 60")
    ax.set_ylabel("Output at SNP 60")
    ax.set_title("Output vs Input at SNP 60")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "snp60_locality_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Off-target effects
    off_target = np.delete(outputs, snp_index, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        values, np.max(np.abs(off_target), axis=1), label="Max off-target", color="red"
    )
    ax.plot(
        values,
        np.mean(np.abs(off_target), axis=1),
        label="Mean off-target",
        color="blue",
    )

    # Add regional means
    for region, (start, end) in {
        "Region 0": (0, 25),
        "Region 0.25": (25, 75),
        "Region 0.5": (75, 100),
    }.items():
        mean = np.mean(np.abs(off_target[:, start:end]), axis=1)
        ax.plot(values, mean, "--", label=f"{region} mean", alpha=0.6)

    ax.set_xlabel("Input value at SNP 60")
    ax.set_ylabel("Off-target output magnitude")
    ax.set_title("Off-target Effects vs. Input at SNP 60")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "snp60_locality_offtarget.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def format_metrics_report(metrics):
    """Format metrics into a detailed report.

    Args:
        metrics (dict): Dictionary of computed metrics

    Returns:
        str: Formatted report string
    """
    sections = [
        (
            "On-Target Analysis",
            [
                f"Slope: {metrics['on_target']['slope']:.4f} (1.0 is ideal)",
                f"Pearson r: {metrics['on_target']['correlation']:.4f} (1.0 is ideal, 0=no correlation)",
                f"MSE: {metrics['on_target']['mse']:.6f}",
                f"R²: {metrics['on_target']['r2']:.4f} (1.0 is ideal, 0=no fit)",
            ],
        ),
        (
            "Off-Target Analysis",
            [
                f"Mean absolute: {metrics['off_target']['mean_abs']:.6f}",
                f"Max absolute: {metrics['off_target']['max_abs']:.6f}",
                f"Total energy: {metrics['off_target']['total_energy']:.6f}",
                f"Standard deviation: {metrics['off_target']['std_dev']:.6f}",
            ],
        ),
        (
            "Regional Analysis",
            [
                "\nRegion 0 (SNPs 0-24):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_0']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_0']['max']:.6f}",
                "\nRegion 0.25 (SNPs 25-74):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_025']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_025']['max']:.6f}",
                "\nRegion 0.5 (SNPs 75-99):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_05']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_05']['max']:.6f}",
            ],
        ),
    ]

    return "\n\n".join(
        f"=== {title} ===\n" + "\n".join(items) for title, items in sections
    )
