#!/usr/bin/env python
# coding: utf-8
from typing import Any, Dict, List, Optional

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


def torch_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    if tensor is None:
        return None
    return tensor.cpu().detach().numpy()


# ==================== Core Reverse Process Analysis ====================


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

        # Get all diagnostics from the core reverse step (dictionary output)
        reverse_dict = model.reverse_diffusion.reverse_diffusion_step(
            xt, t_tensor, return_all=True
        )

        # x_t_minus_1: denoised sample after reverse step [1, 1, seq_len]
        x_t_minus_1 = reverse_dict["x_t_minus_1"]

        # epsilon_theta: model's predicted noise [1, 1, seq_len]
        epsilon_theta = reverse_dict["epsilon_theta"]

        # scaled_pred_noise: β_t/√(1-ᾱ_t) * ε_θ(x_t, t) [1, 1, seq_len]
        scaled_pred_noise = reverse_dict["scaled_pred_noise"]

        # Metrics for diagnostics only
        noise_mse = F.mse_loss(scaled_pred_noise, noise).item()
        x0_diff = F.mse_loss(x_t_minus_1, x0).item()

        # noise_magnitude: mean absolute value of true noise
        noise_magnitude = torch.mean(torch.abs(noise)).item()

        # pred_noise_magnitude: mean absolute value of predicted noise
        pred_noise_magnitude = torch.mean(torch.abs(epsilon_theta)).item()

        # SNR = |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
        alpha_bar_t = reverse_dict["alpha_bar_t"]
        alpha_bar_t = torch.as_tensor(alpha_bar_t, device=x0.device, dtype=x0.dtype)
        signal_power = torch.mean((torch.sqrt(alpha_bar_t) * x0) ** 2).item()
        noise_power = torch.mean((torch.sqrt(1 - alpha_bar_t) * noise) ** 2).item()
        signal_to_noise = signal_power / (noise_power + 1e-8)

        metrics = {
            "noise_mse": noise_mse,  # MSE between predicted and true noise
            "x0_diff": x0_diff,  # MSE between denoised output and original sample
            "noise_magnitude": noise_magnitude,  # mean abs value of true noise
            "pred_noise_magnitude": pred_noise_magnitude,  # mean abs value of predicted noise
            "signal_to_noise": signal_to_noise,  # SNR at this step
        }

    return {
        "timestep": timestep,  # current timestep
        "x0": x0,  # x_0: original clean sample
        "noise": noise,  # ε: random gaussian noise
        "xt": xt,  # x_t: noisy sample at timestep t
        "epsilon_theta": reverse_dict[
            "epsilon_theta"
        ],  # ε_θ(x_t, t): model's predicted noise
        "coef": reverse_dict[
            "coef"
        ],  # β_t/√(1-ᾱ_t): scaling coefficient for predicted noise
        "scaled_pred_noise": reverse_dict[
            "scaled_pred_noise"
        ],  # β_t/√(1-ᾱ_t) * ε_θ: scaled predicted noise
        "mu": reverse_dict["mean"],  # μ_θ(x_t, t): denoised mean before noise is added
        "x_t_minus_1": reverse_dict[
            "x_t_minus_1"
        ],  # x_{t-1}: denoised sample after reverse step
        "beta_t": reverse_dict["beta_t"],  # β_t: noise schedule value
        "alpha_t": reverse_dict["alpha_t"],  # α_t: alpha for this step
        "inv_sqrt_alpha_t": reverse_dict[
            "inv_sqrt_alpha_t"
        ],  # 1/√α_t: inverse alpha for this step
        "alpha_bar_t": reverse_dict["alpha_bar_t"],  # ᾱ_t: cumulative product of alphas
        "sigma_t": reverse_dict["sigma_t"],  # σ_t=√β_t: std of noise added at this step
        "metrics": metrics,  # dictionary of diagnostic metrics
    }


# Print Reverse Statistics
def print_reverse_statistics(
    results: ReverseDiffusionResults, timesteps: Optional[List[int]] = None
) -> None:
    """Print formatted diffusion statistics for specified timesteps.
    Works directly with PyTorch tensors for all calculations.

    Args:
        results: ReverseDiffusionResults dictionary mapping timesteps to result dictionaries
        timesteps: List of timesteps to print statistics for (if None, use all in results)
    """
    print("\n" + "=" * 80)
    print(" REVERSE DIFFUSION STATISTICS ")
    print("=" * 80)

    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    for t in timesteps:
        r = results[t]

        # Diffusion parameters section
        print(f"\nTimestep t = {t}:")
        print("Diffusion Parameters:")
        print(f"- β_t: {r['beta_t'].item():.10f}")
        print(f"- α_t: {r['alpha_t'].item():.10f}")
        print(f"- ᾱ_t: {r['alpha_bar_t'].item():.10f}")
        print(f"- 1/√α_t: {r['inv_sqrt_alpha_t'].item():.10f}")
        print(f"- β_t/√(1-ᾱ_t): {r['coef'].item():.10f}")
        print(f"- σ_t=√β_t: {r['sigma_t'].item():.10f}")

        # Noise prediction metrics
        print("\nNoise Prediction Metrics:")
        print(f"- Noise MSE: {r['metrics']['noise_mse']:.8f}")
        print(f"- |ε| (true noise magnitude): {r['metrics']['noise_magnitude']:.8f}")
        print(
            f"- |εᵩ| (pred noise magnitude): {r['metrics']['pred_noise_magnitude']:.8f}"
        )

        # Calculate relative error directly with tensors
        noise_mag = r["metrics"]["noise_magnitude"]
        pred_noise_mag = r["metrics"]["pred_noise_magnitude"]
        rel_error = (pred_noise_mag / max(noise_mag, 1e-8)) - 1
        print(f"- Relative error: {rel_error:.8f}")

        # Reconstruction metrics
        print("\nReconstruction Metrics:")
        print(f"- x₀ vs x_{t-1} MSE: {r['metrics']['x0_diff']:.8f}")
        print(f"- SNR: {r['metrics']['signal_to_noise']:.8f}")

        # Additional statistics
        print("\nAdditional Information:")
        print(f"- Progress: {t/1000:.1%} through diffusion process")
        print(f"- Signal strength: {torch.sqrt(r['alpha_bar_t']).item():.6f}")
        print(f"- Noise strength: {torch.sqrt(1 - r['alpha_bar_t']).item():.6f}")

        # Tensor shape information
        print("\nTensor Shapes:")
        print(f"- Original x₀: {r['x0'].shape}")
        print(f"- Noisy x_t: {r['xt'].shape}")
        print(f"- Denoised x_{t-1}: {r['x_t_minus_1'].shape}")
        print("-" * 50)


def plot_schedule_parameters(
    results: ReverseDiffusionResults,
    timesteps: Optional[List[int]] = None,
    output_dir=None,
):
    """
    Plot the schedule parameters (β_t, α_t, ᾱ_t, 1/√α_t, β_t/√(1-ᾱ_t), σ_t=√β_t)
    for the given results dictionary as a function of time.
    """
    print("\n" + "=" * 80)
    print(" REVERSE DIFFUSION SCHEDULE PARAMETERS ")
    print("=" * 80)

    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    # Extract Schedule Parameters
    beta_t = [float(torch_to_numpy(results[t]["beta_t"])) for t in timesteps]
    alpha_t = [float(torch_to_numpy(results[t]["alpha_t"])) for t in timesteps]
    alpha_bar_t = [float(torch_to_numpy(results[t]["alpha_bar_t"])) for t in timesteps]
    inv_sqrt_alpha_t = [
        float(torch_to_numpy(results[t]["inv_sqrt_alpha_t"])) for t in timesteps
    ]
    coef = [float(torch_to_numpy(results[t]["coef"])) for t in timesteps]
    sigma_t = [float(torch_to_numpy(results[t]["sigma_t"])) for t in timesteps]

    # Plot Schedule Parameters
    plt.figure(figsize=(7, 6))
    plt.plot(timesteps, beta_t, "b-", label=r"$\beta_t$")
    plt.plot(timesteps, alpha_t, "g-", label=r"$\alpha_t$")
    plt.plot(timesteps, alpha_bar_t, "r--", label=r"$\bar{\alpha}_t$")
    plt.plot(timesteps, inv_sqrt_alpha_t, "m--", label=r"$1/\sqrt{\alpha_t}$")
    plt.plot(timesteps, coef, "k--", label=r"$coef=\beta_t/\sqrt{1-\bar{\alpha}_t}$")
    plt.plot(timesteps, sigma_t, "y--", label=r"$\sigma_t=\sqrt{\beta_t}$")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title("Schedule Parameters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/schedule_parameters.png")
        plt.savefig(f"{output_dir}/schedule_parameters.pdf")
        plt.close()
    else:
        plt.show()


# ==================== Evolution of Denoising ====================


# Visualize diffusion process at different timesteps
def visualize_diffusion_process_lineplot(
    results: ReverseDiffusionResults,
    timesteps: Optional[List[int]] = None,
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

    n_rows, n_cols = len(timesteps), 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, t in enumerate(timesteps):
        r = results[t]
        x0 = torch_to_numpy(r["x0"].squeeze())
        xt = torch_to_numpy(r["xt"].squeeze())
        noise = torch_to_numpy(r["noise"].squeeze())
        scaled_noise = torch_to_numpy(
            (r["epsilon_theta"] / (torch.sqrt(1.0 - r["alpha_bar_t"]))).squeeze()
        )
        epsilon_theta = torch_to_numpy(r["epsilon_theta"].squeeze())
        scaled_epsilon_theta = torch_to_numpy(r["scaled_pred_noise"].squeeze())
        mu = torch_to_numpy(r["mu"].squeeze())
        x_t_minus_1 = torch_to_numpy(r["x_t_minus_1"].squeeze())
        seq_len = x0.shape[-1]
        x_axis = np.arange(seq_len)

        # Set titles
        if i == 0:
            axes[i, 0].set_title(r"Original sample: $x_{0}$", fontsize=6)
            axes[i, 1].set_title(r"Added noise: $\epsilon$", fontsize=6)
            axes[i, 2].set_title(r"Noisy sample: $x_{t}$", fontsize=6)
            axes[i, 3].set_title(
                r"Predicted noise: $\epsilon_{\theta}(x_t, t)$", fontsize=6
            )
            axes[i, 4].set_title(
                r"Scaled predicted noise: $s * \epsilon_{\theta}(x_t, t)$", fontsize=6
            )
            axes[i, 5].set_title(r"Denoising", fontsize=6)
        if i == n_rows - 1:
            for j in range(axes.shape[1]):
                axes[i, j].set_xlabel("Markers", fontsize=6)

        axes[i, 0].set_ylabel(f"t={t}")

        # Original sample
        axes[i, 0].plot(x_axis, x0, "b-", linewidth=1)

        # Added noise
        axes[i, 1].plot(
            x_axis, noise, "r-", linewidth=1, label=r"Added noise: $\epsilon$"
        )

        # Noisy sample
        axes[i, 2].plot(x_axis, xt, "k-", linewidth=1)

        # Predicted noise
        axes[i, 3].plot(
            x_axis,
            noise,
            "r-",
            linewidth=1,
            label=r"Added noise: $\epsilon$",
        )
        axes[i, 3].plot(
            x_axis,
            epsilon_theta,
            "g-",
            linewidth=1,
            label=r"Predicted noise: $\epsilon_{\theta}$",
        )
        axes[i, 3].legend(fontsize=6)

        # Scaled predicted noise
        # axes[i, 4].plot(
        #     x_axis,
        #     epsilon_theta,
        #     "g-",
        #     linewidth=1,
        #     label=r"Predicted noise: $\epsilon_{\theta}$",
        # )
        axes[i, 4].plot(
            x_axis,
            scaled_epsilon_theta,  # scaling by beta_t/sqrt(1-alpha_bar_t) (Reverse Process)
            "m-",
            linewidth=1,
            label=r"Scaled predicted noise: $\beta_t/\sqrt{1-\bar{\alpha_t}} * \epsilon_{\theta}$",
        )
        axes[i, 4].plot(
            x_axis,
            scaled_noise,  # scaling by 1/sqrt(1-alpha_bar_t) (Forward Process)
            "y--",
            linewidth=1,
            label=r"Scaled predicted noise: $1/\sqrt{1-\bar{\alpha_t}} *\epsilon_{\theta}$",
        )

        axes[i, 4].legend(fontsize=6)

        # Denoising (mean and sample)
        axes[i, 5].plot(x_axis, mu, "y--", linewidth=1, label=r"$\mu_{\theta}$")
        axes[i, 5].plot(x_axis, x_t_minus_1, "c-", linewidth=1, label=r"$x_{t-1}$")
        axes[i, 5].legend(fontsize=6)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diffusion_lineplot.png")
        fig.savefig(f"{output_dir}/diffusion_lineplot.pdf")
        plt.close()
    else:
        plt.show()


# Visualize Superimposed Denoising Process
def visualize_diffusion_process_superimposed(
    results: ReverseDiffusionResults,
    timesteps: Optional[List[int]] = None,
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
        x0_vis = torch_to_numpy(result["x0"]).flatten()
        x_t_vis = torch_to_numpy(result["xt"]).flatten()
        x_t_minus_1_vis = torch_to_numpy(result["x_t_minus_1"]).flatten()
        mu_vis = torch_to_numpy(result["mu"]).flatten()
        x_axis = np.arange(len(x0_vis))

        # Plot data
        axes[i].plot(
            x_axis,
            x0_vis,
            color="blue",
            linestyle="-",
            linewidth=1,
            label=r"Original sample: $x_{0}$",
            alpha=0.4,
        )
        axes[i].plot(
            x_axis,
            x_t_vis,
            color="orange",
            linestyle="-",
            linewidth=1,
            label=r"Noisy sample: $x_{t}$",
            alpha=0.6,
        )
        axes[i].plot(
            x_axis,
            mu_vis,
            color="green",
            linestyle="-",
            linewidth=2,
            label=r"Denoised mean: $\mu_\theta(x_{t}, t)$",
            alpha=0.8,
        )
        axes[i].plot(
            x_axis,
            x_t_minus_1_vis,
            color="red",
            linestyle="--",
            linewidth=1,
            label=r"Denoised sample: $x_{t-1}$",
            alpha=1.0,
        )
        axes[i].set_title(f"t={t}")
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

        # Calculate noise reduction
        initial_noise = np.mean(
            np.abs(torch_to_numpy(result["xt"]) - torch_to_numpy(result["x0"]))
        )
        final_noise = np.mean(
            np.abs(torch_to_numpy(result["x_t_minus_1"]) - torch_to_numpy(result["x0"]))
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
        fig.savefig(f"{output_dir}/diffusion_superimposed.png")
        fig.savefig(f"{output_dir}/diffusion_superimposed.pdf")
        plt.close()
    else:
        plt.show()


def plot_reverse_mean_components(results, timesteps, output_dir=None):
    """
    Plot all variables entering the mean calculation in the reverse diffusion step for each timestep.
    Args:
        results: ReverseDiffusionResults dict (timestep -> result dict)
        timesteps: List of timesteps to visualize
        output_dir: Directory to save visualizations
    """
    # Handle timesteps
    if timesteps is None:
        timesteps = sorted(results.keys())
    else:
        # Filter timesteps to only those in results
        timesteps = [t for t in timesteps if t in results]
        timesteps.sort()

    n_rows, n_cols = len(timesteps), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, t in enumerate(timesteps):

        r = results[t]
        xt = torch_to_numpy(r["xt"].squeeze())
        epsilon_theta = torch_to_numpy(r["epsilon_theta"].squeeze())
        coef = torch_to_numpy(r["coef"].squeeze())
        scaled_pred_noise = torch_to_numpy(r["scaled_pred_noise"].squeeze())
        mu = torch_to_numpy(r["mu"].squeeze())
        inv_sqrt_alpha_t = torch_to_numpy(r["inv_sqrt_alpha_t"].squeeze())
        x_t_minus_1 = torch_to_numpy(r["x_t_minus_1"].squeeze())
        seq_len = xt.shape[-1]
        x_axis = np.arange(seq_len)

        # Set titles
        if i == 0:
            axes[i, 0].set_title(r"Noisy sample: $x_{t}$")
            axes[i, 1].set_title(r"$\epsilon_\theta(x_{t}, t)$")
            axes[i, 2].set_title(r"$\beta_t/\sqrt{1-\bar{\alpha_t}} * \epsilon_\theta$")
            axes[i, 3].set_title(
                r"$\mu_\theta(x_{t}, t) = 1/\sqrt{\alpha_t} * (x_t - coef * \epsilon_\theta)$"
            )
            axes[i, 4].set_title(r"$x_{t-1} = \mu_\theta(x_{t}, t) + \sigma_t * z$")
        if i == n_rows - 1:
            for j in range(axes.shape[1]):
                axes[i, j].set_xlabel("SNP Markers")

        # x_t: Noisy sample
        axes[i, 0].plot(x_axis, xt, "b-", linewidth=1)

        # predicted_noise = ε_θ(x_t, t): Predicted noise
        axes[i, 1].plot(
            x_axis,
            epsilon_theta,
            "r-",
            linewidth=1,
            label=r"$\beta_t$ = %.6f, $\alpha_t$ = %.6f" % (r["beta_t"], r["alpha_t"]),
        )
        axes[i, 1].legend(fontsize=6)

        # scaled_pred_noise = β_t/√(1-ᾱ_t) * ε_θ(x_t, t)
        label_scaled_pred_noise = (
            r"$\beta_t$ = %.6f, $\bar{\alpha_t}$ = %.6f, $coef$ = %.6f"
            % (r["beta_t"], r["alpha_t"], coef),
        )
        axes[i, 2].plot(
            x_axis,
            scaled_pred_noise,
            "k-",
            linewidth=1,
            label=label_scaled_pred_noise,
        )
        axes[i, 2].legend(fontsize=6)

        # mu = μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
        label_mu = r"$1/\sqrt{\alpha_t}$ = %.6f, $coef$ = %.6f" % (
            inv_sqrt_alpha_t,
            coef,
        )
        axes[i, 3].plot(
            x_axis,
            mu,
            "y-",
            linewidth=1,
            label=label_mu,
        )
        axes[i, 3].legend(fontsize=6)

        # x_{t-1} = μ_θ(x_t, t) + σ_t * z
        axes[i, 4].plot(
            x_axis,
            mu,
            "y-",
            linewidth=1,
            label=label_mu,
        )
        axes[i, 4].plot(
            x_axis,
            x_t_minus_1,
            "g-",
            linewidth=1,
            label=r"$\sigma_t$ = %.6f" % r["sigma_t"],
        )
        axes[i, 4].legend(fontsize=6)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/reverse_mean_components.png")
        fig.savefig(f"{output_dir}/reverse_mean_components.pdf")
        plt.close(fig)
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
        for key in ["x0", "xt", "mu", "x_t_minus_1"]:
            if s[key] is not None:
                arr = np.array(s[key])
                print(
                    f"  {key:12s}: mean={np.mean(arr):8.4f}, std={np.std(arr):8.4f}, "
                    f"min={np.min(arr):8.4f}, max={np.max(arr):8.4f}"
                )

        # 2. Noise Prediction Analysis
        print("\nNoise Prediction:")
        for key in ["noise", "epsilon_theta", "scaled_pred_noise"]:
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
        if s["xt"] is not None and s["x_t_minus_1"] is not None:
            var_ratio = np.std(s["x_t_minus_1"]) / np.std(s["xt"])
            print(f"  Variance Ratio (σ_{t-1}/σ_t)     : {var_ratio:8.4f}")

        # Noise prediction accuracy
        if s["noise"] is not None and s["epsilon_theta"] is not None:
            noise_mse = np.mean((s["noise"] - s["epsilon_theta"]) ** 2)
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
    """Plot x0, xt, mu, x_t_minus_1 for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    n_rows = (len(timesteps) + 1) // 2  # 2 plots per row
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axs = axs.flatten() if n_rows > 1 else [axs]

    for i, t in enumerate(timesteps):
        s = signals[t]
        axs[i].plot(
            s["x0"],
            color="blue",
            linestyle="-",
            linewidth=1,
            label=r"Original sample: $x_{0}$",
            alpha=0.4,
        )
        axs[i].plot(
            s["xt"],
            color="orange",
            linestyle="-",
            linewidth=1,
            label=r"Noisy sample: $x_{t}$",
            alpha=0.6,
        )
        axs[i].plot(
            s["mu"],
            color="green",
            linestyle="-",
            linewidth=2,
            label=r"Denoised mean: $\mu_\theta(x_{t}, t)$",
            alpha=0.8,
        )
        axs[i].plot(
            s["x_t_minus_1"],
            color="red",
            linestyle="--",
            linewidth=1,
            label=r"Denoised sample: $x_{t-1}$",
            alpha=1.0,
        )
        axs[i].set_title(f"Reverse Diffusion Signals at t={t}")
        axs[i].legend(loc="lower right", fontsize=8)
        axs[i].grid(True, alpha=0.3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_signals.png")
        fig.savefig(f"{output_dir}/diagnostic_signals.pdf")
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
        axs[i].plot(
            s["noise"],
            color="black",
            linestyle="-",
            linewidth=1,
            label=r"Added noise: $\epsilon$",
            alpha=0.4,
        )
        axs[i].plot(
            s["epsilon_theta"],
            color="magenta",
            linestyle="-",
            linewidth=1,
            label=r"Predicted noise: $\epsilon_{\theta}$",
            alpha=0.6,
        )
        axs[i].plot(
            s["scaled_pred_noise"],
            color="#56B4E9",  # Teal/Cyan
            linestyle="-",
            linewidth=2,
            label=r"Scaled predicted noise: $\beta_t/\sqrt{1-\bar{\alpha_t}} * \epsilon_{\theta}$",
            alpha=0.8,
        )
        pred_noise = s["epsilon_theta"] / np.sqrt(1 - s["alpha_bar_t"])
        axs[i].plot(
            pred_noise,
            color="#F0E442",  # Dark Yellow
            linestyle="--",
            linewidth=1,
            label=r"Scaled predicted noise: $1/\sqrt{1-\bar{\alpha_t}} * \epsilon_{\theta}$",
            alpha=1.0,
        )
        axs[i].set_title(f"Noise Comparison at t={t}")
        axs[i].legend(loc="lower right", fontsize=8)
        axs[i].grid(True, alpha=0.3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_noise.png")
        fig.savefig(f"{output_dir}/diagnostic_noise.pdf")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_variance(results, timesteps, output_dir=None):
    """Plot variance bar plots for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    labels = [r"$x_{t}$", r"$\mu_\theta(x_{t}, t)$", r"$x_{t-1}$"]
    keys = ["xt", "mu", "x_t_minus_1"]

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
        fig.savefig(f"{output_dir}/diagnostic_variance.pdf")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_schedule(results, timesteps, output_dir=None):
    """Plot schedule parameters across timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    fig, ax = plt.subplots(figsize=(10, 4))

    for param, color in [("sigma_t", "red"), ("coef", "blue")]:
        vals = [signals[t][param] for t in timesteps]
        ax.plot(
            timesteps,
            vals,
            marker="o",
            label=r"$\sigma_{t}$" if param == "sigma_t" else r"$\beta_{t}$",
            color=color,
        )

    ax.set_title("Schedule Parameters Across Timesteps")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_schedule.png")
        fig.savefig(f"{output_dir}/diagnostic_schedule.pdf")
        plt.close(fig)
    else:
        plt.show()


def plot_diagnostic_histograms(results, timesteps, output_dir=None):
    """Plot histograms for all timesteps."""
    signals = prepare_diagnostic_signals(results, timesteps)
    n_rows = len(timesteps)
    fig, axs = plt.subplots(
        n_rows, 4, figsize=(16, 3 * n_rows), sharex="col", sharey="row"
    )
    if n_rows == 1:
        # Convert 1D array to 2D array with shape (1, 4) for consistent indexing
        axs = axs.reshape(1, 4)

    for i, t in enumerate(timesteps):
        s = signals[t]
        # Order: original sample, noisy sample, denoised mean, denoised sample
        for j, key in enumerate(["x0", "xt", "mu", "x_t_minus_1"]):
            axs[i, j].hist(
                s[key],
                bins=100,
                alpha=0.7,
                label=(
                    r"$x_0$"
                    if key == "x0"
                    else (
                        r"$\mu_\theta(x_{t}, t)$"
                        if key == "mu"
                        else r"$x_{t}$" if key == "xt" else r"$x_{t-1}$"
                    )
                ),
            )
            # More descriptive titles
            titles = {
                "x0": f"$x_0$: Original sample at t={t}",
                "xt": f"$x_t$: Noisy sample at t={t}",
                "mu": rf"$\mu_{{\theta}}$: Denoised mean at t={t}",
                "x_t_minus_1": f"$x_{{t-1}}$: Denoised sample at t={t}",
            }
            axs[i, j].set_title(titles[key])

    for ax in axs[-1]:
        ax.set_xlabel("Value")
    for ax in axs[:, 0]:
        ax.set_ylabel("Count")

    fig.suptitle("Histograms of Key Signals")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir:
        fig.savefig(f"{output_dir}/diagnostic_histograms.png")
        fig.savefig(f"{output_dir}/diagnostic_histograms.pdf")
        plt.close(fig)
    else:
        plt.show()


def prepare_diagnostic_signals(results, timesteps):
    """
    Extract and prepare all signals needed for diagnostics from results.
    Converts tensors to numpy arrays and reshapes them for plotting and analysis.

    Args:
        results: Dict mapping timesteps to ReverseDiffusionResult
        timesteps: List of timesteps to analyze
    Returns:
        Dict mapping timesteps to prepared signals
    """
    signals = {}
    for t in timesteps:
        r = results[t]

        signals[t] = {
            # Signal values (flattened to 1D for easier analysis)
            "x0": torch_to_numpy(r["x0"]).flatten(),
            "xt": torch_to_numpy(r["xt"]).flatten(),
            "mu": torch_to_numpy(r["mu"]).flatten(),
            "x_t_minus_1": torch_to_numpy(r["x_t_minus_1"]).flatten(),
            "noise": torch_to_numpy(r["noise"]).flatten(),
            "epsilon_theta": torch_to_numpy(r["epsilon_theta"]).flatten(),
            "scaled_pred_noise": torch_to_numpy(r["scaled_pred_noise"]).flatten(),
            # Scalar parameters (converted to Python scalars)
            "alpha_bar_t": r["alpha_bar_t"].item(),
            "alpha_t": r["alpha_t"].item(),
            "beta_t": r["beta_t"].item(),
            "sigma_t": r["sigma_t"].item(),
            "coef": r["coef"].item(),
            "inv_sqrt_alpha_t": r["inv_sqrt_alpha_t"].item(),
        }
    return signals
