#!/usr/bin/env python
# coding: utf-8

"""Diffusion model implementation for SNP data."""

from typing import Dict

import torch
import torch.nn.functional as F

from .forward_diffusion import ForwardDiffusion
from .mlp import (
    ComplexMLP,
    LinearMLP,
    zero_out_model_parameters,
)
from .network_base import NetworkBase
from .reverse_diffusion import ReverseDiffusion
from .time_sampler import UniformContinuousTimeSampler
from .unet import UNet1D
from .utils import bcast_right, tensor_to_device


def build_mixing_mask(seq_length: int, pattern: list, interval: int, device=None):
    """
    Build a 1D mask for separating staircase vs real regions.

    Args:
        seq_length: total sequence length (L).
        pattern: list of [start, end, value] entries from config.
        interval: how many "real" steps between staircase blocks.
    Returns:
        mask: torch.BoolTensor of shape [1, 1, L].
              True = staircase region, False = real region.
    """
    mask = torch.zeros(seq_length, dtype=torch.bool)

    step = 0
    while step + interval + (pattern[-1][1] - pattern[0][0]) <= seq_length:
        for start, end, _ in pattern:
            mask[step + start : step + end] = True
        step += interval + (pattern[-1][1] - pattern[0][0])

    if device is not None:
        mask = mask.to(device)
    return mask.view(1, 1, -1)


class DiffusionModel(NetworkBase):
    """Diffusion model with 1D Convolutional network for SNP data.

    Implements both forward diffusion (data corruption) and reverse
    diffusion (denoising) processes for SNP data. The forward process
    gradually adds noise to the data following a predefined schedule,
    while the reverse process learns to denoise the data using a UNet.

    Inherits from NetworkBase to leverage PyTorch Lightning functionality.
    """

    def __init__(self, hparams: Dict):
        super().__init__(hparams)

        # Set data shape
        self._data_shape = (hparams["unet"]["channels"], hparams["data"]["seq_length"])

        # Continuous time sampler
        self.time_sampler = UniformContinuousTimeSampler(
            tmin=1, tmax=hparams["diffusion"]["timesteps"]
        )

        # ForwardDiffusion: Forward diffusion process
        self.forward_diffusion = ForwardDiffusion(
            time_steps=hparams["diffusion"]["timesteps"],
            beta_start=hparams["diffusion"]["beta_start"],
            beta_end=hparams["diffusion"]["beta_end"],
            beta_schedule=hparams["diffusion"]["beta_schedule"],
        )

        # Noise Predictor (UNet1D, LinearMLP, etc.)
        self.noise_predictor = UNet1D(
            emb_dim=hparams["unet"]["embedding_dim"],
            dim_mults=hparams["unet"]["dim_mults"],
            channels=hparams["unet"]["channels"],
            in_channels=hparams["unet"]["in_channels"],
            with_time_emb=hparams["unet"]["with_time_emb"],
            time_dim=hparams["unet"]["time_dim"],
            with_pos_emb=hparams["unet"]["with_pos_emb"],
            pos_dim=hparams["unet"]["pos_dim"],
            norm_groups=hparams["unet"]["norm_groups"],
            seq_length=hparams["data"]["seq_length"],
            edge_pad=hparams["unet"]["edge_pad"],
            enable_checkpointing=hparams["unet"]["enable_checkpointing"],
            strict_resize=hparams["unet"]["strict_resize"],
            pad_value=hparams["unet"]["pad_value"],
            dropout=hparams["unet"]["dropout"],
            use_scale_shift_norm=hparams["unet"]["use_scale_shift_norm"],
            use_attention=hparams["unet"]["use_attention"],
            attention_heads=hparams["unet"]["attention_heads"],
            attention_dim_head=hparams["unet"]["attention_dim_head"],
        )

        # ReverseDiffusion: Reverse diffusion process
        self.reverse_diffusion = ReverseDiffusion(
            self.forward_diffusion,
            self.noise_predictor,
            self._data_shape,
            denoise_step=hparams["diffusion"]["denoise_step"],
            discretize=hparams["diffusion"]["discretize"],
            in_channels=hparams["unet"]["in_channels"],
        )

        # Testing: Zero noise predictor (zero out model params)
        # Beware that in MLP if we input staircase sample, output
        # is always zero as both weights and biases are set to 0.
        # zero_out_model_parameters(self.noise_predictor)

        # Testing: In mixing, get staircase, real loss separately
        if hparams["data"].get("mixing", False):
            mask = build_mixing_mask(
                hparams["data"]["seq_length"],
                hparams["data"]["mixing_pattern"],
                hparams["data"]["mixing_interval"],
            )
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    # ==================== Training Methods ====================
    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            xt: Noisy input data of shape [B, C=1, L].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C=1, L].
        """
        return self.predict_added_noise(xt, t)

    def predict_added_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise that was added during forward diffusion.

        Args:
            xt: Noisy input data of shape [B, C=1, L].
            t: Timesteps of shape [B].

        Returns:
            torch.Tensor: Predicted noise of shape [B, C=1, L].
        """
        # Ensure tensors have the correct shape of [B, C=1, L]
        device = xt.device
        xt = self._prepare_batch(xt)
        t = tensor_to_device(t, device)

        # Build second channel: xt / sqrt(alpha_bar_t)
        alpha_bar_t = self.forward_diffusion.alpha_bar(t)  # [B, L]
        alpha_bar_t = bcast_right(alpha_bar_t, xt.ndim)  # [B, C, L]

        # numerical stability
        denom = torch.sqrt(alpha_bar_t + 1e-8)
        xt_scaled = xt / denom

        # Concatenate along channel dim -> [B, 2, L]
        xt_2ch = torch.cat([xt, xt_scaled], dim=1)

        # UNet now outputs x0_hat; convert to epsilon prediction: ε_hat = xt - x0_hat
        x0_hat = self.noise_predictor(xt_2ch, t)
        return xt - x0_hat

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Wrapper function to compute loss for a batch of input data.

        Args:
            x0: Input batch from dataloader of shape [B, C=1, L]
        Returns:
            torch.Tensor: MSE loss.
        """
        return self.compute_loss_Nicole(x0)

    def compute_loss_lowT(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the secondary low-T loss by restricting timestep sampling to a small range.

        Uses hparams["diffusion"].get("lowT_max_T", 10) as the default cap.

        Args:
            x0: Input batch of shape [B, C=1, L]
        Returns:
            torch.Tensor: MSE loss computed with timesteps sampled in [1, lowT_max_T].
        """
        return self.compute_loss_Nicole(
            x0, max_T=self.hparams["diffusion"]["lowT_val_timesteps"]
        )

    def compute_loss_Nicole(
        self, x0: torch.Tensor, max_T: int | None = None
    ) -> torch.Tensor:
        """
        Compute MSE between (xt - x0) and predicted noise for a batch.
        This method performs the forward diffusion process, predicts noise,
        and calculates the loss. One can scale the MSE with a scale factor.

        Optional data augmentation (controlled by config) can replace a
        proportion of x0 values with random valid genotype values when
        constructing the noisy input xt. Importantly, the loss is still
        computed against the original, true x0 (not the augmented copy),
        as suggested.

        Args:
            x0: Input batch from dataloader of shape [B, C=1, L]
        Returns:
            torch.Tensor: MSE loss.
        """
        # Ensure batch has the correct shape of [B, C=1, L]
        device = x0.device
        x0 = self._prepare_batch(x0)

        # Keep the original x0 for loss calculation
        x0_true = x0

        # Sample random timesteps for each batch element
        # If max_T is provided, restrict sampling to low timesteps [1, max_T]
        if max_T is not None:
            # Continuous sampling in [1, max_T]
            t = 1.0 + (float(max_T) - 1.0) * torch.rand(
                size=(x0.shape[0],), device=device
            )
        else:
            t = tensor_to_device(self.time_sampler.sample(shape=(x0.shape[0],)), device)

        # Generate Gaussian noise
        eps = torch.randn_like(x0, device=device)

        # Optional augmentation: replace a proportion of x0 entries with
        # random genotype values ONLY for the forward diffusion input.
        # This does not alter the target used in the loss.
        aug_prob = float(self.hparams["data"]["augment_prob"])
        if self.training and aug_prob > 0.0:

            # Get valid genotype values from config
            geno_vals = self.hparams["data"]["genotype_values"]  # [0.25, 0.0, 0.5]
            geno_tensor = torch.tensor(geno_vals, device=device, dtype=x0.dtype)

            # Create random mask for positions to replace
            replace_mask = torch.rand_like(x0, device=device) < aug_prob

            # Sample random genotype values for replacement
            rand_idx = torch.randint(
                low=0, high=geno_tensor.numel(), size=x0.shape, device=device
            )
            rand_genos = geno_tensor[rand_idx]

            # Create augmented version: x0_aug
            x0_aug = torch.where(replace_mask, rand_genos, x0)

            # Log the realized augmentation fraction
            self.log(
                "augment_replace_frac",
                replace_mask.float().mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        else:
            x0_aug = x0

        # Forward diffusion: add noise to the (possibly augmented) batch
        xt = self.forward_diffusion.sample(x0_aug, t, eps)

        # ε_θ(xt, t): Model's prediction of the noise added at timestep t
        eps_theta = self.predict_added_noise(xt, t)

        # Elementwise MSE loss
        mse = F.mse_loss(
            eps_theta, (xt - x0_true), reduction="none"
        )  # shape: [B, C=1, L]

        # Optional staircase/real separation
        if getattr(self, "mask", None) is not None:
            mask = self.mask.expand_as(mse)
            loss_stair = (
                mse[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
            )
            loss_real = (
                mse[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)
            )

            # Logging (W&B / Lightning)
            self.log(
                "loss_stair", loss_stair, prog_bar=True, on_step=True, on_epoch=True
            )
            self.log("loss_real", loss_real, prog_bar=True, on_step=True, on_epoch=True)

        # Aggregate to scalar (mean over all elements)
        total_loss = mse.mean()

        # Scale MSE by (1 - ᾱ_t)
        alpha_bar_t = self.forward_diffusion.alpha_bar(t)  # shape: [B, L]
        alpha_bar_t = bcast_right(alpha_bar_t, eps_theta.ndim)  # shape: [B, C=1, L]
        scaled_mse = mse / (1 - alpha_bar_t)

        # Aggregate to scalar (mean over all elements)
        scaled_total_loss = scaled_mse.mean()

        return total_loss  # or use `scaled_total_loss` if you want scaling

    # ===== Inference Methods =====
    def generate_samples(
        self,
        num_samples: int = 10,
        denoise_step: int = 10,
        discretize: bool = False,
        seed: int = 42,
        device=None,
    ) -> torch.Tensor:
        """
        Generate new samples from random noise using the reverse diffusion process.

        This method implements Algorithm 2 from Ho et al., 2020 (DDPM paper):
            For t = T,...,1:
                x_{t-1} ~ p_θ(x_{t-1}|xt)
            where p_θ(x_{t-1}|xt) = N(x_{t-1}; μ_θ(xt, t), β_t I)

        Args:
            num_samples: Number of samples to generate.
            denoise_step: Number of timesteps to skip in reverse diffusion (default: 10).
            discretize: If True, discretize the output to 0, 0.5, and 1.0 values (SNP genotypes).
            seed: Optional seed for reproducible sample generation.
            device: torch.device

        Returns:
            torch.Tensor: Generated samples of shape [num_samples, C, seq_len].
        """
        # Determine device to use
        if device is None:
            device = (
                self.device
                if hasattr(self, "device")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # Use the reverse diffusion process to generate samples
        return self.reverse_diffusion.generate_samples(
            num_samples=num_samples,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=seed,
            device=device,
        )

    def denoise_sample(
        self,
        batch: torch.Tensor,
        denoise_step: int = 10,
        discretize: bool = False,
        seed: int = 42,
        device=None,
    ) -> torch.Tensor:
        """
        Denoise an input batch using the reverse diffusion process.

        This method starts the reverse diffusion process from the provided batch
        (which should be noisy or real data). It does not generate new noise, but
        instead denoises the actual input batch by iteratively applying the reverse
        diffusion steps. This is useful for denoising specific data samples, e.g.,
        for evaluation or restoration tasks.

        Args:
            batch: Noisy input batch to denoise. Shape: [B, C=1, L].
            denoise_step: Number of timesteps to skip in reverse diffusion. If None, uses the instance default.
            discretize: If True, discretize output to SNP values. If None, uses the instance default.
            seed: Random seed for reproducibility (not used in this function, kept for API compatibility).
            device: Device to run the computation on. If None, uses CUDA if available, else CPU.

        Returns:
            Denoised batch of the same shape as input.
        """
        # Determine device to use
        if device is None:
            device = (
                self.device
                if hasattr(self, "device")
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        # Use the reverse diffusion process to denoise the batch
        return self.reverse_diffusion.denoise_sample(
            batch=batch,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=seed,
            device=device,
        )
