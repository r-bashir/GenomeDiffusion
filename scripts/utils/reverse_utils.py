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

        # epsilon: random gaussian noise [1, 1, seq_len]
        epsilon = torch.randn_like(x0)

        # t_tensor: timestep as tensor [1]
        t_tensor = torch.tensor([timestep], device=device, dtype=torch.long)

        # xt: noisy sample at timestep t [1, 1, seq_len]
        xt = model.forward_diffusion.sample(x0, t_tensor, epsilon)

        # Get all diagnostics from the core reverse step (dictionary output)
        reverse_dict = model.reverse_diffusion.reverse_diffusion_step(
            xt, t_tensor, return_all=True
        )

        # x_t_minus_1: denoised sample after reverse step [1, 1, seq_len]
        sample_mse = F.mse_loss(reverse_dict["x_t_minus_1"], x0).item()

        # epsilon_theta: model's predicted noise [1, 1, seq_len]
        noise_mse = F.mse_loss(reverse_dict["epsilon_theta"], epsilon).item()

        # noise_magnitude: mean absolute value of true noise
        noise_magnitude = torch.mean(torch.abs(epsilon)).item()

        # pred_noise_magnitude: mean absolute value of predicted noise
        pred_noise_magnitude = torch.mean(torch.abs(reverse_dict["epsilon_theta"]))

        # SNR = |√(ᾱ_t) * x_0|² / |√(1-ᾱ_t) * ε|²
        alpha_bar_t = reverse_dict["alpha_bar_t"]
        alpha_bar_t = torch.as_tensor(alpha_bar_t, device=x0.device, dtype=x0.dtype)
        signal_power = torch.mean((torch.sqrt(alpha_bar_t) * x0) ** 2).item()
        noise_power = torch.mean((torch.sqrt(1 - alpha_bar_t) * epsilon) ** 2).item()
        signal_to_noise = signal_power / (noise_power + 1e-8)

        metrics = {
            "noise_mse": noise_mse,  # MSE between predicted and true noise
            "sample_mse": sample_mse,  # MSE between denoised output and original sample
            "noise_magnitude": noise_magnitude,  # mean abs value of true noise
            "pred_noise_magnitude": pred_noise_magnitude,  # mean abs value of predicted noise
            "signal_to_noise": signal_to_noise,  # SNR at this step
        }

    return {
        "timestep": timestep,  # current timestep
        "epsilon": epsilon,  # ε: random gaussian noise
        "epsilon_theta": reverse_dict[
            "epsilon_theta"
        ],  # ε_θ(x_t, t): model's predicted noise
        "x0_true": x0,  # x_0: original clean sample
        "x0_pred": reverse_dict["x0"],  # x_0: predicted clean sample
        "xt": reverse_dict["xt"],  # x_t: noisy sample at timestep t
        "coef_x0": reverse_dict[
            "coef_x0"
        ],  # (√ᾱ_{t-1}*β_t)/(1-ᾱ_t): Coefficient for x_0
        "coef_xt": reverse_dict[
            "coef_xt"
        ],  # (√α_t*(1-ᾱ_{t-1}))/(1-ᾱ_t): Coefficient for x_t
        "mu": reverse_dict["mean"],  # μ̃_t(x_t, x_0): Denoised mean before adding noise
        "x_t_minus_1": reverse_dict[
            "x_t_minus_1"
        ],  # x_{t-1}: denoised sample after reverse step
        "beta_t": reverse_dict["beta_t"],  # β_t: noise schedule value
        "alpha_t": reverse_dict["alpha_t"],  # α_t: alpha for this step
        "alpha_bar_t": reverse_dict["alpha_bar_t"],  # ᾱ_t: cumulative product of alphas
        "sigma_t": reverse_dict["sigma_t"],  # √β̃_t: Noise std added at this step
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
        print(f"- σ_t=√β̃_t: {r['sigma_t'].item():.10f}")

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
        print(f"- x₀ vs x_{t-1} MSE: {r['metrics']['sample_mse']:.8f}")
        print(f"- SNR: {r['metrics']['signal_to_noise']:.8f}")

        # Additional statistics
        print("\nAdditional Information:")
        print(f"- Progress: {t/1000:.1%} through diffusion process")
        print(f"- Signal strength: {torch.sqrt(r['alpha_bar_t']).item():.6f}")
        print(f"- Noise strength: {torch.sqrt(1 - r['alpha_bar_t']).item():.6f}")

        # Tensor shape information
        print("\nTensor Shapes:")
        print(f"- Original x₀: {r['x0_true'].shape}")
        print(f"- Predicted x₀: {r['x0_pred'].shape}")
        print(f"- Noisy x_t: {r['xt'].shape}")
        print(f"- Denoised x_{t-1}: {r['x_t_minus_1'].shape}")
        print("-" * 50)


def plot_schedule_parameters(
    results: ReverseDiffusionResults,
    timesteps: Optional[List[int]] = None,
    output_dir=None,
):
    """
    Plot the schedule parameters (β_t, α_t, ᾱ_t, σ_t=√β̃_t)
    for the given results dictionary as a function of time.
    """
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
    sigma_t = [float(torch_to_numpy(results[t]["sigma_t"])) for t in timesteps]

    # Plot Schedule Parameters
    plt.figure(figsize=(7, 6))
    plt.plot(timesteps, beta_t, "b-", label=r"$\beta_t$")
    plt.plot(timesteps, alpha_t, "g-", label=r"$\alpha_t$")
    plt.plot(timesteps, alpha_bar_t, "r--", label=r"$\bar{\alpha}_t$")
    plt.plot(timesteps, sigma_t, "y--", label=r"$\sigma_t = \sqrt{\tilde{\beta_t}}$")
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

    n_rows, n_cols = len(timesteps), 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, t in enumerate(timesteps):
        r = results[t]
        epsilon = torch_to_numpy(r["epsilon"].squeeze())
        epsilon_theta = torch_to_numpy(r["epsilon_theta"].squeeze())
        # scaled_pred_noise = torch_to_numpy(  # manual scaling by 1/sqrt(1-alpha_bar_t) (Forward Process)
        #     (r["epsilon_theta"] / (torch.sqrt(1.0 - r["alpha_bar_t"]))).squeeze()
        # )
        x0_true = torch_to_numpy(r["x0_true"].squeeze())
        x0_pred = torch_to_numpy(r["x0_pred"].squeeze())
        xt = torch_to_numpy(r["xt"].squeeze())
        mu = torch_to_numpy(r["mu"].squeeze())
        x_t_minus_1 = torch_to_numpy(r["x_t_minus_1"].squeeze())
        seq_len = x0_true.shape[-1]
        x_axis = np.arange(seq_len)

        # Set titles
        if i == 0:
            axes[i, 0].set_title(r"Original sample: $x_{0}$", fontsize=6)
            axes[i, 1].set_title(r"Added noise: $\epsilon$", fontsize=6)
            axes[i, 2].set_title(r"Noisy sample: $x_{t}$", fontsize=6)
            axes[i, 3].set_title(
                r"Predicted noise: $\epsilon_{\theta}(x_t, t)$", fontsize=6
            )
            # axes[i, 4].set_title(
            #     r"Scaled predicted noise: $s * \epsilon_{\theta}(x_t, t)$", fontsize=6
            # )
            axes[i, 4].set_title(r"Denoising", fontsize=6)
        if i == n_rows - 1:
            for j in range(axes.shape[1]):
                axes[i, j].set_xlabel("Markers", fontsize=6)

        axes[i, 0].set_ylabel(f"t={t}")

        # Original sample
        axes[i, 0].plot(
            x_axis, x0_true, "b-", linewidth=1, label=r"Original sample: $x_0$"
        )
        axes[i, 0].plot(
            x_axis, x0_pred, "b--", linewidth=1, label=r"Predicted sample: $\hat{x_0}$"
        )

        # Added noise
        axes[i, 1].plot(
            x_axis, epsilon, "r-", linewidth=1, label=r"Added noise: $\epsilon$"
        )

        # Noisy sample
        axes[i, 2].plot(x_axis, xt, "k-", linewidth=1, label=r"Noisy sample: $x_t$")

        # Noise and predicted noise
        axes[i, 3].plot(
            x_axis,
            epsilon,
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

        # Predicted, and scaled predicted noise
        # axes[i, 4].plot(
        #     x_axis,
        #     epsilon_theta,
        #     "g-",
        #     linewidth=1,
        #     label=r"Predicted noise: $\epsilon_{\theta}$",
        # )

        # axes[i, 4].plot(
        #     x_axis,
        #     scaled_pred_noise,  # scaling by 1/sqrt(1-alpha_bar_t) (Forward Process)
        #     "y--",
        #     linewidth=1,
        #     label=r"Scaled predicted noise: $1/\sqrt{1-\bar{\alpha_t}} *\epsilon_{\theta}$",
        # )

        # axes[i, 4].legend(fontsize=6)

        # Denoising (mean and sample)
        axes[i, 4].plot(x_axis, mu, "y--", linewidth=1, label=r"$\mu_{\theta}$")
        axes[i, 4].plot(x_axis, x_t_minus_1, "c-", linewidth=1, label=r"$x_{t-1}$")
        axes[i, 4].legend(fontsize=6)

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
    - True Denoised sample (green)
    - Predicted Denoised sample (yellow)

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
        x0_pred = torch_to_numpy(result["x0_pred"]).flatten()
        xt = torch_to_numpy(result["xt"]).flatten()
        mu = torch_to_numpy(result["mu"]).flatten()
        x_t_minus_1 = torch_to_numpy(result["x_t_minus_1"]).flatten()
        x_axis = np.arange(len(x0_pred))

        # Plot data
        axes[i].plot(
            x_axis,
            x0_pred,
            color="green",
            linestyle="-",
            linewidth=1,
            label=r"Predicted sample: $\hat{x}_0$",
            alpha=0.4,
        )
        axes[i].plot(
            x_axis,
            xt,
            color="orange",
            linestyle="-",
            linewidth=1,
            label=r"Noisy sample: $x_{t}$",
            alpha=0.6,
        )
        axes[i].plot(
            x_axis,
            mu,
            color="green",
            linestyle="-",
            linewidth=2,
            label=r"Denoised mean: $\tilde{\mu}_\theta$",
            alpha=0.8,
        )
        axes[i].plot(
            x_axis,
            x_t_minus_1,
            color="red",
            linestyle="--",
            linewidth=1,
            label=r"Denoised sample: $x_{t-1}$",
            alpha=1.0,
        )
        axes[i].set_title(f"t={t}")
        axes[i].legend(loc="lower right", fontsize=8)
        axes[i].grid(True, alpha=0.3)

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

        epsilon_theta = torch_to_numpy(r["epsilon_theta"].squeeze())
        x0_true = torch_to_numpy(r["x0_true"].squeeze())
        x0_pred = torch_to_numpy(r["x0_pred"].squeeze())
        xt = torch_to_numpy(r["xt"].squeeze())
        coef_x0 = torch_to_numpy(r["coef_x0"].squeeze())
        coef_xt = torch_to_numpy(r["coef_xt"].squeeze())
        mu = torch_to_numpy(r["mu"].squeeze())
        sigma_t = torch_to_numpy(r["sigma_t"].squeeze())
        x_t_minus_1 = torch_to_numpy(r["x_t_minus_1"].squeeze())
        seq_len = xt.shape[-1]
        x_axis = np.arange(seq_len)

        # Set titles
        if i == 0:
            axes[i, 0].set_title(r"Noisy sample: $x_{t}$")
            axes[i, 1].set_title(r"$\epsilon_\theta(x_{t}, t)$")
            axes[i, 2].set_title(
                r"Original sample: $x_0, \hat{x}_0 = x_t - \epsilon_\theta$"
            )
            axes[i, 3].set_title(
                r"$\tilde{\mu}_\theta(x_t, t) = \text{coef_x0} \cdot \hat{x}_0 - \text{coef_xt} \cdot x_t$"
            )
            axes[i, 4].set_title(
                r"$x_{t-1} = \tilde{\mu}_\theta(x_t, t) + \sigma_t * z$"
            )
        if i == n_rows - 1:
            for j in range(axes.shape[1]):
                axes[i, j].set_xlabel("SNP Markers")

        # x_t: Noisy sample
        axes[i, 0].plot(x_axis, xt, "b-", linewidth=1)
        axes[i, 0].set_ylim(-3, 3)

        # epsilon_theta = ε_θ(x_t, t): Predicted noise
        axes[i, 1].plot(
            x_axis,
            epsilon_theta,
            "r-",
            linewidth=1,
            label=r"$\beta_t$ = %.6f, $\alpha_t$ = %.6f" % (r["beta_t"], r["alpha_t"]),
        )
        axes[i, 1].legend(fontsize=6)
        axes[i, 1].set_ylim(-3, 3)

        # True and Predicted x0
        axes[i, 2].plot(x_axis, x0_true, "b-", linewidth=1, label=r"Original: $x_0$")
        axes[i, 2].plot(
            x_axis, x0_pred, "r--", linewidth=1, label=r"Predicted: $\hat{x}_0$"
        )
        axes[i, 2].legend(fontsize=6, loc="lower right")
        axes[i, 2].set_ylim(-3, 3)

        # mu = μ̃_t(x_t, x_0) = (√ᾱ_{t-1}*β_t)/(1-ᾱ_t) * x_0 + (√α_t*(1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
        label_mu = r"$\tilde{{\mu}}_\theta(x_t, t) = {:.6f} \cdot \hat{{x}}_0 - {:.6f} \cdot x_t$".format(
            coef_x0, coef_xt
        )
        axes[i, 3].plot(
            x_axis,
            mu,
            "y-",
            linewidth=1,
            label=label_mu,
        )
        axes[i, 3].legend(fontsize=6)
        axes[i, 3].set_ylim(-3, 3)

        # x_{t-1} = μ̃_t(x_t, x_0) + σ_t * z
        axes[i, 4].plot(
            x_axis,
            mu,
            "y-",
            linewidth=1,
            label=r"$\tilde{\mu}_\theta(x_t, t)$",
        )
        axes[i, 4].plot(
            x_axis,
            x_t_minus_1,
            "g-",
            linewidth=1,
            label=r"$x_{{t-1}} = \tilde{{\mu}}_\theta(x_t, t) + {:.6f} \cdot z$".format(
                sigma_t
            ),
        )
        axes[i, 4].legend(fontsize=6)
        axes[i, 4].set_ylim(-3, 3)

    fig.tight_layout()
    if output_dir:
        fig.savefig(f"{output_dir}/reverse_mean_components.png")
        fig.savefig(f"{output_dir}/reverse_mean_components.pdf")
        plt.close(fig)
    else:
        plt.show()
