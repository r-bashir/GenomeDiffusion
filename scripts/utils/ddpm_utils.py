"""
DDPM Markov Chain Utilities
--------------------------
Clean, correct implementations for Markov chain reverse diffusion, both for generation (from noise) and denoising a real sample (from a noisy version of x0).

All functions use the correct Markov chain: at each step, the output of the previous step is fed as input to the next.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


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


# Markov Chain Reverse Process: Denoising from Noisy x_t at t_start
def run_markov_reverse_process(
    model: "DiffusionModel",
    x_t_start: Tensor,
    t_start: int,
    device: torch.device,
    return_all_steps: bool = True,
) -> Dict[int, Tensor]:
    """
    Run the Markov chain reverse diffusion process starting from a noisy sample x_t_start at timestep t_start.
    Args:
        model (DiffusionModel): The diffusion model instance containing the reverse diffusion process.
        x_t_start (Tensor): The noisy sample at timestep t_start (shape: [batch, ...]).
        t_start (int): The timestep from which to start the reverse process (should match the noise level in x_t_start).
        device (torch.device): The device (CPU/GPU) for all tensor operations.
        return_all_steps (bool): If True, return all intermediate samples for each timestep. If False, only return final denoised sample.
    Returns:
        Dict[int, Tensor]: Dictionary mapping timesteps to samples (including t_start and 0).
            If return_all_steps is False, only {0: x_0_recon} is returned.
    """
    model.eval()
    with torch.no_grad():
        # Initialize samples dictionary with the starting noisy sample at t_start
        samples = {t_start: x_t_start.clone()}
        x_t = x_t_start
        # Reverse diffusion: iteratively denoise from x_t (at t_start) down to x_1
        for t in range(t_start, 1, -1):
            # t: current timestep we are denoising from (x_t)
            # t_tensor: target timestep after denoising (t-1), i.e., we want x_{t-1}
            t_tensor = torch.full((1,), t - 1, device=device, dtype=torch.long)
            # Perform one reverse diffusion step: x_t -> x_{t-1}
            x_t = model.reverse_diffusion.reverse_diffusion_step(x_t, t_tensor)
            samples[t - 1] = x_t.clone()  # Store the result for timestep t-1
        # After the final step, x_t is x_0 (fully denoised sample)
        samples[0] = x_t.clone()  # Store the final denoised output at t=0

    if return_all_steps:
        return samples
    else:
        return {0: samples[0]}


def plot_denoising_comparison(x0, x_t, x0_recon, t_markov, output_path):
    """
    Plots and saves a comparison of original, noisy, and denoised signals using matplotlib OOP API.
    Left: x0 and x_t; Right: x0, x_t, x0_recon. Annotates MSE and correlation on right.
    Returns MSE and correlation between x0 and x0_recon.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    x0_np = x0.detach().cpu().numpy().flatten()
    x_t_np = x_t.detach().cpu().numpy().flatten()
    x0_recon_np = x0_recon.detach().cpu().numpy().flatten()

    mse = np.mean((x0_np - x0_recon_np) ** 2)
    corr = np.corrcoef(x0_np, x0_recon_np)[0, 1]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left subplot: x0 and x_t
    axs[0].plot(x0_np, label=r"Original $x_0$", color="blue")
    axs[0].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[0].set_title(f"Original vs Noisy (T=100)")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Value")
    if t_markov <= 100:
        axs[0].set_ylim(-1, 1)
    else:
        axs[0].set_ylim(-3, 3)
    axs[0].legend()

    # Right subplot: x0, x_t, x0_recon
    axs[1].plot(x0_np, label=r"Original $x_0$", color="blue")
    axs[1].plot(x_t_np, label=r"Noisy $x_t$", color="red", alpha=0.6)
    axs[1].plot(x0_recon_np, label=r"Denoised $x_{t-1}$", color="green", alpha=0.7)
    axs[1].set_title(f"Denoising Comparison (T={t_markov})")
    axs[1].set_xlabel("Position")
    axs[1].legend()

    # Annotate MSE and correlation
    axs[1].annotate(
        f"MSE: {mse:.2e}\nCorr: {corr:.4f}",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=10,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(str(output_path / f"markov_reverse_t{t_markov}.png"))
    fig.savefig(str(output_path / f"markov_reverse_t{t_markov}.pdf"))
    plt.close(fig)

    return mse, corr
