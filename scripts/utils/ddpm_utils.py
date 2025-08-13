"""
DDPM Markov Chain Utilities
--------------------------
Clean, correct implementations for Markov chain reverse diffusion, both for generation (from noise) and denoising a real sample (from a noisy version of x0).

All functions use the correct Markov chain: at each step, the output of the previous step is fed as input to the next.

1. MSE(x_t_minus_1, x0) should DECREASE as t → 0 because:
- We start at t=T with a noisy sample far from x0
- Each denoising step should bring us closer to x0
- At t=0, we should have our best reconstruction of x0
- Therefore, MSE(x_t_minus_1, x0) should steadily decrease

2. MSE(x_t_minus_1, x_t) should INCREASE as t → 0 because:
- We start at t=T where x_t_minus_1 is very close to x_t_start
- As we denoise, we move away from the noisy x_t_start
- By t=0, we should be very different from x_t_start
- Therefore, MSE(x_t_minus_1, x_t_start) should steadily increase

3. Corr(x_t_minus_1, x_t) should DECREASE as t → 0 because:
- We start at t=T where x_t_minus_1 is very close to x_t_start
- As we denoise, we move away from the noisy x_t_start
- By t=0, we should be very different from x_t_start
- Therefore, Corr(x_t_minus_1, x_t_start) should steadily decrease
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def torch_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to numpy array."""
    if tensor is None:
        return None
    return tensor.cpu().detach().numpy()


# Utility: Add noise to x0 at a given timestep
def get_noisy_sample(
    model: "DiffusionModel", x0: Tensor, t: int, eps: Optional[Tensor] = None
) -> Tensor:
    """
    Add Gaussian noise to a real sample x0 at a specified timestep t using the model's forward diffusion process.
    Args:
        model (DiffusionModel): The diffusion model instance containing the forward diffusion process.
        x0 (Tensor): The original clean sample (shape: [batch, ...]).
        t (int): The timestep at which to add noise (0-indexed).
        eps (Optional[Tensor]): Optional pre-sampled noise tensor. If None, uses standard normal noise.
    Returns:
        Tensor: The noisy sample x_t at timestep t.
    """
    assert isinstance(t, int), f"t must be int, got {type(t)}"
    device = x0.device
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    if eps is None:
        eps = torch.randn_like(x0)
    x_t = model.forward_diffusion.sample(x0, t_tensor, eps)
    return x_t


# Markov Chain Reverse Process: Denoising from Noisy x_t at T
def run_markov_reverse_process(
    model: "DiffusionModel",
    x0: Tensor,  # Original clean sample
    x_t: Tensor,  # Noisy sample at T
    T: int,
    device: torch.device,
    return_all_steps: bool = True,
    print_mse: bool = False,
) -> Dict[int, Tensor]:
    """
    Run the Markov chain reverse diffusion process starting from a noisy sample x_t at timestep T.
    Args:
        model (DiffusionModel): The diffusion model instance containing the reverse diffusion process.
        x0 (Tensor): The original clean sample for MSE comparison.
        x_t (Tensor): The noisy sample at timestep T (shape: [batch, ...]).
        T (int): The timestep from which to start the reverse process.
        device (torch.device): The device (CPU/GPU) for all tensor operations.
        return_all_steps (bool): If True, return all intermediate samples. If False, only return x0.
        print_mse (bool): If True, print MSE and correlation metrics between denoised samples and both x0 and x_t.
    Returns:
        Dict[int, Tensor]: Dictionary mapping timesteps to samples.
            If return_all_steps is False, only {0: x0_pred} is returned.
    """
    model.eval()
    with torch.no_grad():
        # Store all samples in a dictionary, starting with x_t_start
        samples = {T: x_t.clone()}
        x_t_minus_1 = x_t  # Current sample in the chain

        # Print initial MSEs
        # MSE(x_t_minus_1, x0) should decrease as we denoise (t → 0)
        # Corr(x_t_minus_1, x0) should increase as we denoise (t → 0)
        # MSE(x_t_minus_1, x_t) should increase as we denoise (t → 0)
        # Corr(x_t_minus_1, x_t) should decrease as we denoise (t → 0)
        if print_mse:
            mse_x0 = torch.mean((x_t_minus_1 - x0) ** 2).item()
            corr_x0 = torch.corrcoef(  # Pearson correlation coefficient
                torch.stack([x_t_minus_1.flatten(), x0.flatten()])
            )[0, 1]
            mse_xt = torch.mean((x_t_minus_1 - x_t) ** 2).item()
            corr_xt = torch.corrcoef(  # Pearson correlation coefficient
                torch.stack([x_t_minus_1.flatten(), x_t.flatten()])
            )[0, 1]
            print(
                f"t={T} | MSE(x_t_minus_1, x0): {mse_x0:.6f}, Corr(x_t_minus_1, x0): {corr_x0:.6f} | MSE(x_t_minus_1, x_t): {mse_xt:.6f}, Corr(x_t_minus_1, x_t): {corr_xt:.6f}"
            )

        # Reverse diffusion: from t_start down to 1 (after loop x_curr will be x0)
        for t in range(T, 0, -1):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            x_t_minus_1 = model.reverse_diffusion.reverse_diffusion_step(
                x_t_minus_1, t_tensor
            )
            samples[t - 1] = x_t_minus_1.clone()

            # Print MSEs if requested
            if print_mse:
                mse_x0 = torch.mean((x_t_minus_1 - x0) ** 2).item()
                mse_xt = torch.mean((x_t_minus_1 - x_t) ** 2).item()
                corr_x0 = torch.corrcoef(  # Pearson correlation coefficient
                    torch.stack([x_t_minus_1.flatten(), x0.flatten()])
                )[0, 1]
                corr_xt = torch.corrcoef(  # Pearson correlation coefficient
                    torch.stack([x_t_minus_1.flatten(), x_t.flatten()])
                )[0, 1]
                print(
                    f"t={t-1} | MSE(x_t_minus_1, x0): {mse_x0:.6f}, Corr(x_t_minus_1, x0): {corr_x0:.6f} | MSE(x_t_minus_1, x_t): {mse_xt:.6f}, Corr(x_t_minus_1, x_t): {corr_xt:.6f}"
                )

    if return_all_steps:
        return samples
    else:
        return {0: samples[0]}


def plot_denoising_comparison(x0, x_t, x_t_minus_1, T, output_path):
    """
    Plots and saves a comparison of original, noisy, and denoised signals using matplotlib OOP API.
    Left: x0 and x_t; Right: x0, x_t, x_t_minus_1. Annotates MSE and correlation metrics.
    Args:
        x0 (Tensor): Original clean sample
        x_t (Tensor): Noisy sample at timestep T
        x_t_minus_1 (Tensor): Denoised sample
        T (int): Timestep
        output_path (Path): Directory to save plots
    Returns:
        Tuple[float, float, float, float]: (MSE(x_t_minus_1, x0), Corr(x_t_minus_1, x0), MSE(x_t_minus_1, x_t), Corr(x_t_minus_1, x_t))
    """

    x0_np = torch_to_numpy(x0).flatten()
    x_t_np = torch_to_numpy(x_t).flatten()
    x_t_minus_1_np = torch_to_numpy(x_t_minus_1).flatten()

    mse_x0 = np.mean((x0_np - x_t_minus_1_np) ** 2)
    corr_x0 = np.corrcoef(x0_np, x_t_minus_1_np)[0, 1]
    mse_xt = np.mean((x_t_np - x_t_minus_1_np) ** 2)
    corr_xt = np.corrcoef(x_t_np, x_t_minus_1_np)[0, 1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left subplot: x0 and x_t
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

    # Right subplot: x0, x_t, x0_recon
    axs[1].plot(x0_np, label=r"Original $x_0$", color="blue")
    # axs[1].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[1].plot(x_t_minus_1_np, label=r"Denoised $x_{t-1}$", color="green", alpha=0.7)
    axs[1].set_title(f"Denoising Comparison (T={T})")
    axs[1].set_xlabel("Position")
    axs[1].legend()

    # Annotate MSE and correlation
    axs[1].annotate(
        f"MSE ($x_{{t-1}}$, $x_0$): {mse_x0:.6f}, Corr ($x_{{t-1}}$, $x_0$): {corr_x0:.6f}\nMSE ($x_{{t-1}}$, $x_t$): {mse_xt:.6f}, Corr ($x_{{t-1}}$, $x_t$): {corr_xt:.6f}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=10,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(str(output_path / f"markov_reverse_t{T}.png"))
    fig.savefig(str(output_path / f"markov_reverse_t{T}.pdf"))
    plt.close(fig)

    return mse_x0, corr_x0, mse_xt, corr_xt


def plot_denoising_trajectory(x0, x_t, samples_dict, T, output_path):
    """
    Plots the denoising trajectory of x0_recon at each timestep during Markov reverse diffusion.
    Args:
        x0: The original clean sample (Tensor or np.ndarray)
        x_t: The noisy sample at timestep T (Tensor or np.ndarray)
        samples_dict: Dict of denoised samples at each step (shape: [num_steps, seq_len])
        T: The starting timestep (int, for labeling)
        output_path: Path to save the plot
    """
    x0_np = torch_to_numpy(x0).flatten()
    x_t_np = torch_to_numpy(x_t).flatten()

    # Prepare x0_recon_arr and timesteps from samples_dict (preserve insertion order)
    timesteps = list(samples_dict.keys())
    x0_recon_arr = np.stack(
        [torch_to_numpy(samples_dict[t]).flatten() for t in timesteps], axis=0
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left subplot: x0 and x_t
    axs[0].plot(x0_np, label=r"Original $x_0$", color="blue")
    axs[0].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[0].set_title(f"Original vs Noisy (T={T})")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Value")
    axs[0].set_ylim(axs[0].get_ylim())  # Match y-axis

    # Right subplot: all x0_recon
    axs[1].plot(x_t_np, "m--", label=r"Noisy $x_t$", alpha=0.6)
    for i, x0_recon in enumerate(x0_recon_arr):
        x0_recon_flat = x0_recon.flatten()
        if timesteps[i] < 10:
            axs[1].plot(x0_recon_flat, label=f"Step {timesteps[i]}", alpha=0.7)
    axs[1].set_title(f"Denoising Trajectory (T={T})")
    axs[1].set_xlabel("Position")
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
                f"\nRegion 0 (SNPs 0-24):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_0']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_0']['max']:.6f}",
                f"\nRegion 0.25 (SNPs 25-74):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_025']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_025']['max']:.6f}",
                f"\nRegion 0.5 (SNPs 75-99):",
                f"  Mean abs: {metrics['off_target']['regional_means']['region_05']['mean']:.6f}",
                f"  Max abs: {metrics['off_target']['regional_means']['region_05']['max']:.6f}",
            ],
        ),
    ]

    return "\n\n".join(
        f"=== {title} ===\n" + "\n".join(items) for title, items in sections
    )


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
    for i, val in enumerate(values):
        label = f"SNP 60 = {val:.2f}" if i % 2 == 0 else None
        ax.plot(outputs[i], alpha=0.7, label=label)

    ax.axvline(snp_index, color="red", linestyle="--", label="SNP 60")
    ax.set_xlabel("SNP Position")
    ax.set_ylabel("Output Value")
    ax.set_title("Output at all SNPs for each SNP 60 value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=1)

    # Add region backgrounds
    ax.axvspan(0, 25, alpha=0.1, color="blue", label="Region 0")
    ax.axvspan(25, 75, alpha=0.1, color="green", label="Region 0.25")
    ax.axvspan(75, 100, alpha=0.1, color="red", label="Region 0.5")

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
