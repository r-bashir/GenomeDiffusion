#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Debug script for reverse diffusion process indexing issues.

This script examines the reverse diffusion process step-by-step to identify
potential indexing issues, particularly around t=1000 where parameters might
be accessed at incorrect indices (999 or 1001 instead of 1000).
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# All imports after path modification
# ruff: noqa: E402

from src import DiffusionModel
from src.utils import bcast_right, prepare_batch_shape, set_seed, tensor_to_device
from utils.reverse_utils import generate_timesteps

# Set global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug reverse diffusion indexing issues"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=2, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default="1,2,500,998,999,1000",
        help="Comma-separated list of timesteps to debug",
    )
    return parser.parse_args()


def reverse_diffusion_step(model, x_t, t, step_num=None):
    """
    Debug version of reverse_diffusion_step with detailed logging.

    Args:
        model: The diffusion model
        x_t: Noisy sample at timestep t
        t: Current timestep (tensor or int)
        step_num: Step number for logging

    Returns:
        x_{t-1}: Sample at timestep t-1
    """
    print("\n" + "=" * 60)
    if step_num is not None:
        print(f"DEBUG STEP {step_num}: Reverse diffusion at timestep t={t}")
    else:
        print(f"DEBUG: Reverse diffusion at timestep t={t}")
    print("=" * 60)

    # Ensure tensors are on the correct device
    device = x_t.device
    t = tensor_to_device(t, device)

    # Ensure x_t has the correct shape [B, C, L]
    x_t = prepare_batch_shape(x_t)

    print(f"Input x_t shape: {x_t.shape}")
    print(f"Input x_t device: {x_t.device}")
    print(
        f"Timestep t: {t} (type: {type(t)}, device: {t.device if hasattr(t, 'device') else 'N/A'})"
    )

    # Convert t to tensor if it's an integer
    if isinstance(t, int):
        t = torch.tensor([t] * x_t.size(0), dtype=torch.long, device=device)

    print(f"Timestep t tensor: {t} (shape: {t.shape})")

    # Debug parameter access - check what indices are actually used
    print("\nDEBUG: Parameter access for timestep t=" + str(t[0].item()))

    # Check the forward diffusion parameter arrays
    forward_diff = model.forward_diffusion
    print(f"Forward diffusion tmin: {forward_diff.tmin}, tmax: {forward_diff.tmax}")
    print(f"Beta array (0-indexed) length: {len(forward_diff.betas)}")
    print(f"Alpha array (0-indexed) length: {len(forward_diff.alphas)}")
    print(f"Alpha_bar array (1-indexed) length: {len(forward_diff.alphas_bar)}")

    # Test parameter access with detailed indexing info
    try:
        t_val = t[0].item()

        # Detailed index verification for each parameter array
        print("\n=== INDEX VERIFICATION FOR TIMESTEP t=" + str(t_val) + " ===")

        # Check array lengths and expected indices
        beta_array_len = len(forward_diff.betas)
        alpha_array_len = len(forward_diff.alphas)
        alpha_bar_array_len = len(forward_diff.alphas_bar)

        print("Array lengths:")
        print(
            f"  betas (0-indexed): {beta_array_len} elements [0 to {beta_array_len-1}]"
        )
        print(
            f"  alphas (0-indexed): {alpha_array_len} elements [0 to {alpha_array_len-1}]"
        )
        print(
            f"  alphas_bar (1-indexed): {alpha_bar_array_len} elements [0 to {alpha_bar_array_len-1}]"
        )

        # Calculate expected indices for this timestep
        expected_beta_idx = t_val - 1  # 0-indexed: t=1000 -> idx=999
        expected_alpha_idx = t_val - 1  # 0-indexed: t=1000 -> idx=999
        expected_alpha_bar_idx = t_val  # 1-indexed: t=1000 -> idx=1000

        print("\nExpected indices for t=" + str(t_val) + ":")
        print(f"  beta[{expected_beta_idx}] (0-indexed)")
        print(f"  alpha[{expected_alpha_idx}] (0-indexed)")
        print(f"  alpha_bar[{expected_alpha_bar_idx}] (1-indexed)")

        # Bounds checking
        bounds_ok = True
        if expected_beta_idx < 0 or expected_beta_idx >= beta_array_len:
            print(
                f"🚨 ERROR: Beta index {expected_beta_idx} out of bounds [0, {beta_array_len-1}]"
            )
            bounds_ok = False
        if expected_alpha_idx < 0 or expected_alpha_idx >= alpha_array_len:
            print(
                f"🚨 ERROR: Alpha index {expected_alpha_idx} out of bounds [0, {alpha_array_len-1}]"
            )
            bounds_ok = False
        if expected_alpha_bar_idx < 0 or expected_alpha_bar_idx >= alpha_bar_array_len:
            print(
                f"🚨 ERROR: Alpha_bar index {expected_alpha_bar_idx} out of bounds [0, {alpha_bar_array_len-1}]"
            )
            bounds_ok = False

        if not bounds_ok:
            print("🚨 CRITICAL: Index bounds violation detected!")
            return None

        print("✅ All indices are within bounds")

        # Beta access: instance method handles 0-indexed conversion internally
        print("\nBeta access: t=" + str(t_val))
        beta_t = tensor_to_device(forward_diff.beta(t), device)
        print(f"Beta_t (method): {beta_t[0].item():.8f}")

        # Verify by direct access to confirm we get the same value
        beta_direct = forward_diff.betas[expected_beta_idx]
        print(f"Beta_t (direct array[{expected_beta_idx}]): {beta_direct.item():.8f}")
        if abs(beta_t[0].item() - beta_direct.item()) > 1e-8:
            print("🚨 ERROR: Beta method vs direct access mismatch!")
            print(f"  Method: {beta_t[0].item():.10f}")
            print(f"  Direct: {beta_direct.item():.10f}")
        else:
            print("✅ Beta method and direct access match")

        # Alpha access: instance method handles 0-indexed conversion internally
        print("\nAlpha access: t=" + str(t_val))
        alpha_t = tensor_to_device(forward_diff.alpha(t), device)
        print(f"Alpha_t (method): {alpha_t[0].item():.8f}")

        # Verify by direct access
        alpha_direct = forward_diff.alphas[expected_alpha_idx]
        print(
            f"Alpha_t (direct array[{expected_alpha_idx}]): {alpha_direct.item():.8f}"
        )
        if abs(alpha_t[0].item() - alpha_direct.item()) > 1e-8:
            print("🚨 ERROR: Alpha method vs direct access mismatch!")
            print(f"  Method: {alpha_t[0].item():.10f}")
            print(f"  Direct: {alpha_direct.item():.10f}")
        else:
            print("✅ Alpha method and direct access match")

        # Verify alpha = 1 - beta relationship
        expected_alpha = 1.0 - beta_direct.item()
        print(
            f"Verification: α_t = 1 - β_t = 1 - {beta_direct.item():.8f} = {expected_alpha:.8f}"
        )
        if abs(alpha_t[0].item() - expected_alpha) > 1e-6:
            print("🚨 ERROR: α_t ≠ 1 - β_t relationship violated!")
        else:
            print("✅ α_t = 1 - β_t relationship verified")

        # Alpha_bar access: instance method handles 1-indexed conversion internally
        print("\nAlpha_bar access: t=" + str(t_val))
        alpha_bar_t = tensor_to_device(forward_diff.alpha_bar(t), device)
        print(f"Alpha_bar_t (method): {alpha_bar_t[0].item():.8f}")

        # Verify by direct access
        alpha_bar_direct = forward_diff.alphas_bar[expected_alpha_bar_idx]
        print(
            f"Alpha_bar_t (direct array[{expected_alpha_bar_idx}]): {alpha_bar_direct.item():.8f}"
        )
        if abs(alpha_bar_t[0].item() - alpha_bar_direct.item()) > 1e-8:
            print("🚨 ERROR: Alpha_bar method vs direct access mismatch!")
            print(f"  Method: {alpha_bar_t[0].item():.10f}")
            print(f"  Direct: {alpha_bar_direct.item():.10f}")
        else:
            print("✅ Alpha_bar method and direct access match")

        # Special verification for boundary cases
        if t_val == 1:
            print("\n=== BOUNDARY CASE t=1 VERIFICATION ===")
            print("For t=1: ᾱ_1 should equal α_1")
            print(f"  ᾱ_1 = {alpha_bar_t[0].item():.8f}")
            print(f"  α_1 = {alpha_t[0].item():.8f}")
            if abs(alpha_bar_t[0].item() - alpha_t[0].item()) > 1e-8:
                print("🚨 ERROR: ᾱ_1 ≠ α_1 for boundary case!")
            else:
                print("✅ Boundary case ᾱ_1 = α_1 verified")

        elif t_val == forward_diff.tmax:
            print(f"\n=== BOUNDARY CASE t={forward_diff.tmax} VERIFICATION ===")
            print(f"This is the maximum timestep (t={forward_diff.tmax})")
            print(
                f"  Using beta[{expected_beta_idx}], alpha[{expected_alpha_idx}], alpha_bar[{expected_alpha_bar_idx}]"
            )
            print(f"  ᾱ_{t_val} should be very small: {alpha_bar_t[0].item():.8f}")
            if alpha_bar_t[0].item() > 0.1:
                print(
                    f"⚠️  WARNING: ᾱ_{t_val} = {alpha_bar_t[0].item():.8f} seems large for max timestep"
                )

        # Check for unused elements in alpha_bar array
        if alpha_bar_array_len > forward_diff.tmax + 1:
            unused_elements = alpha_bar_array_len - (forward_diff.tmax + 1)
            print("\n=== ALPHA_BAR ARRAY ANALYSIS ===")
            print(f"Alpha_bar array has {alpha_bar_array_len} elements")
            print(
                f"Used indices: 0 to {forward_diff.tmax} ({forward_diff.tmax + 1} elements)"
            )
            print(
                f"Unused elements: {unused_elements} (indices {forward_diff.tmax + 1} to {alpha_bar_array_len - 1})"
            )

        print(f"\n=== TIMESTEP t={t_val} INDEX VERIFICATION COMPLETE ===")

    except IndexError as e:
        print(f"🚨 INDEX ERROR during parameter access: {e}")
        print("This confirms an indexing issue!")
        return None
    except Exception as e:
        print(f"🚨 OTHER ERROR during parameter access: {e}")
        return None

    # Broadcast parameters to match x_t's dimensions
    ndim = x_t.ndim
    print("\nParameter shapes BEFORE bcast_right (target ndim=%d):" % ndim)
    print(f"Beta_t shape: {beta_t.shape}")
    print(f"Alpha_t shape: {alpha_t.shape}")
    print(f"Alpha_bar_t shape: {alpha_bar_t.shape}")

    beta_t = bcast_right(beta_t, ndim)
    alpha_t = bcast_right(alpha_t, ndim)
    alpha_bar_t = bcast_right(alpha_bar_t, ndim)

    print("\nParameter shapes AFTER bcast_right:")
    print(f"Beta_t shape: {beta_t.shape}")
    print(f"Alpha_t shape: {alpha_t.shape}")
    print(f"Alpha_bar_t shape: {alpha_bar_t.shape}")

    # Numerical stability constant
    eps = 1e-7

    # Predict noise using the noise prediction model
    print("\nPredicting noise with model...")
    epsilon_theta = model.reverse_diffusion.noise_predictor(x_t, t)
    print(f"Predicted noise shape: {epsilon_theta.shape}")
    print(
        f"Predicted noise stats: mean={epsilon_theta.mean().item():.6f}, std={epsilon_theta.std().item():.6f}"
    )

    # Compute mean for p(x_{t-1}|x_t) as in Algorithm 2
    # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
    print("\nComputing reverse diffusion equations:")

    # Step 1: Compute 1/√α_t
    inv_sqrt_alpha_t = torch.rsqrt(alpha_t + eps)  # 1/√α_t
    print(
        f"1/√α_t = 1/√{alpha_t.flatten()[0].item():.8f} = {inv_sqrt_alpha_t.flatten()[0].item():.8f}"
    )

    # Validation: Check if alpha_t is reasonable
    alpha_val = alpha_t.flatten()[0].item()
    if alpha_val <= 0 or alpha_val > 1:
        print(f"⚠️  WARNING: α_t = {alpha_val:.8f} is outside expected range (0, 1]")
    if inv_sqrt_alpha_t.flatten()[0].item() > 10:
        print(
            f"⚠️  WARNING: 1/√α_t = {inv_sqrt_alpha_t.flatten()[0].item():.8f} is very large (α_t too small)"
        )

    # Step 2: Compute √(1-ᾱ_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t + eps)
    print(
        f"√(1-ᾱ_t) = √(1-{alpha_bar_t.flatten()[0].item():.8f}) = {sqrt_one_minus_alpha_bar_t.flatten()[0].item():.8f}"
    )

    # Validation: Check if alpha_bar_t is reasonable
    alpha_bar_val = alpha_bar_t.flatten()[0].item()
    if alpha_bar_val <= 0 or alpha_bar_val > 1:
        print(f"⚠️  WARNING: ᾱ_t = {alpha_bar_val:.8f} is outside expected range (0, 1]")
    if sqrt_one_minus_alpha_bar_t.flatten()[0].item() < 0.001:
        print(
            f"⚠️  WARNING: √(1-ᾱ_t) = {sqrt_one_minus_alpha_bar_t.flatten()[0].item():.8f} is very small (ᾱ_t too close to 1)"
        )

    # Step 3: Compute β_t/√(1-ᾱ_t)
    coef = beta_t / sqrt_one_minus_alpha_bar_t  # β_t/√(1-ᾱ_t)
    print(
        f"β_t/√(1-ᾱ_t) = {beta_t.flatten()[0].item():.8f}/{sqrt_one_minus_alpha_bar_t.flatten()[0].item():.8f} = {coef.flatten()[0].item():.8f}"
    )

    # Validation: Check if coefficient is reasonable
    beta_val = beta_t.flatten()[0].item()
    coef_val = coef.flatten()[0].item()
    if beta_val <= 0 or beta_val > 1:
        print(f"⚠️  WARNING: β_t = {beta_val:.8f} is outside expected range (0, 1]")
    if coef_val > 100:
        print(
            f"⚠️  WARNING: β_t/√(1-ᾱ_t) = {coef_val:.8f} is very large (potential numerical instability)"
        )

    # Step 4: Compute the noise term
    noise_term = coef * epsilon_theta
    print("Noise term: (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t)")
    print(f"  Shape: {noise_term.shape}")
    print(
        f"  Stats: mean={noise_term.mean().item():.6f}, std={noise_term.std().item():.6f}"
    )

    # Validation: Check if noise term is reasonable
    noise_mean = noise_term.mean().item()
    noise_std = noise_term.std().item()
    if abs(noise_mean) > 10:
        print(f"⚠️  WARNING: Noise term mean = {noise_mean:.6f} is very large")
    if noise_std > 10:
        print(f"⚠️  WARNING: Noise term std = {noise_std:.6f} is very large")
    if torch.isnan(noise_term).any():
        print("🚨 ERROR: Noise term contains NaN values!")
    if torch.isinf(noise_term).any():
        print("🚨 ERROR: Noise term contains infinite values!")

    # Step 5: Compute x_t - noise_term
    x_minus_noise = x_t - noise_term
    print("x_t - noise_term:")
    print(f"  Shape: {x_minus_noise.shape}")
    print(
        f"  Stats: mean={x_minus_noise.mean().item():.6f}, std={x_minus_noise.std().item():.6f}"
    )

    # Validation: Check if x_minus_noise is reasonable
    x_minus_mean = x_minus_noise.mean().item()
    x_minus_std = x_minus_noise.std().item()
    if abs(x_minus_mean) > 10:
        print(f"⚠️  WARNING: x_t - noise_term mean = {x_minus_mean:.6f} is very large")
    if x_minus_std > 10:
        print(f"⚠️  WARNING: x_t - noise_term std = {x_minus_std:.6f} is very large")

    # Step 6: Compute the mean μ_θ(x_t, t)
    mean = inv_sqrt_alpha_t * x_minus_noise
    mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
    print("Mean μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t)):")
    print(f"  Shape: {mean.shape}")
    print(f"  Stats: mean={mean.mean().item():.6f}, std={mean.std().item():.6f}")

    # Validation: Check if mean is reasonable
    mean_val = mean.mean().item()
    mean_std = mean.std().item()
    if abs(mean_val) > 10:
        print(f"⚠️  WARNING: Mean μ_θ(x_t, t) = {mean_val:.6f} is very large")
    if mean_std > 10:
        print(f"⚠️  WARNING: Mean μ_θ(x_t, t) std = {mean_std:.6f} is very large")
    if torch.isnan(mean).any():
        print("🚨 ERROR: Mean contains NaN values after nan_to_num!")
    if torch.isinf(mean).any():
        print("🚨 ERROR: Mean contains infinite values after nan_to_num!")

    # Add noise for t > 1, no noise for t = 1
    if t[0].item() > 1:
        # Sample noise z ~ N(0, I)
        z = torch.randn_like(x_t)
        # Compute variance: σ_t^2 = β_t
        sigma_t = torch.sqrt(beta_t + eps)
        x_prev = mean + sigma_t * z
        print("\nAdding noise (t > 1):")
        print(
            f"  σ_t = √β_t = √{beta_t.flatten()[0].item():.8f} = {sigma_t.flatten()[0].item():.8f}"
        )
        print(f"  z ~ N(0, I), shape: {z.shape}")
        print("  x_{t-1} = μ_θ(x_t, t) + σ_t * z")
    else:
        x_prev = mean
        print("\nNo noise added (t = 1):")
        print("  x_0 = μ_θ(x_1, 1)")

    print("\nFinal output x_{t-1}:")
    print(f"  Shape: {x_prev.shape}")
    print(f"  Range: [{x_prev.min().item():.6f}, {x_prev.max().item():.6f}]")
    print(f"  Stats: mean={x_prev.mean().item():.6f}, std={x_prev.std().item():.6f}")

    # Final validation: Check if output is reasonable
    final_mean = x_prev.mean().item()
    final_std = x_prev.std().item()
    final_min = x_prev.min().item()
    final_max = x_prev.max().item()

    if abs(final_mean) > 10:
        print(f"⚠️  WARNING: Final output mean = {final_mean:.6f} is very large")
    if final_std > 10:
        print(f"⚠️  WARNING: Final output std = {final_std:.6f} is very large")
    if final_max - final_min > 100:
        print(
            f"⚠️  WARNING: Final output range = {final_max - final_min:.6f} is very large"
        )
    if torch.isnan(x_prev).any():
        print("🚨 ERROR: Final output contains NaN values!")
        return None
    if torch.isinf(x_prev).any():
        print("🚨 ERROR: Final output contains infinite values!")
        return None

    print("✅ All validation checks passed!")

    return x_prev


def debug_reverse_process(model, x0, timesteps, num_samples=2):
    """
    Debug the reverse diffusion process for specific timesteps.

    Args:
        model: The diffusion model
        x0: Original data samples
        timesteps: List of timesteps to debug
        num_samples: Number of samples to process
    """
    print("\n" + "=" * 80)
    print("DEBUGGING REVERSE DIFFUSION PROCESS")
    print("=" * 80)

    # Take only the requested number of samples
    x0 = x0[:num_samples]
    print(f"Processing {num_samples} samples")
    print(f"Original data shape: {x0.shape}")

    # Start from pure noise (as in generate_samples)
    x = torch.randn_like(x0)
    print(f"Starting from noise with shape: {x.shape}")

    # Debug each timestep
    for i, t in enumerate(sorted(timesteps, reverse=True)):
        print("\n" + "*" * 40)
        print(f"PROCESSING TIMESTEP {t} (step {i+1}/{len(timesteps)})")
        print("*" * 40)

        # Create timestep tensor
        t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)

        # Debug the reverse diffusion step
        x_prev = reverse_diffusion_step(model, x, t_tensor, step_num=i + 1)

        if x_prev is None:
            print(f"ERROR: Failed at timestep {t}")
            break

        x = x_prev

        # Print summary stats
        print("\nSUMMARY for timestep " + str(t) + ":")
        print(f"  Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        print(f"  Input mean: {x.mean().item():.6f}")
        print(f"  Input std: {x.std().item():.6f}")

    return x


def main():
    # Set global seed for reproducibility
    set_seed(42)

    # Parse Arguments
    args = parse_args()

    try:
        # Load the model from checkpoint
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        # Get model config and move to device
        config = model.hparams
        model = model.to(device)
        model.eval()

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)

    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Load Dataset (Test)
    print("\nLoading test dataset...")
    model.setup("test")
    test_loader = model.test_dataloader()

    # Prepare Batch
    print("Preparing a batch of test data...")
    x0 = next(iter(test_loader)).to(device)
    x0 = x0.unsqueeze(1)  # Add channel dimension
    print(f"Input shape: {x0.shape}, dtype: {x0.dtype}, device: {x0.device}")

    # Output directory
    checkpoint_path = Path(args.checkpoint)
    base_dir = checkpoint_path.parent.parent
    output_dir = base_dir / "reverse_diffusion"
    output_dir.mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # --------------------------------------------------------------------------
    #                          Don't change anything above
    # --------------------------------------------------------------------------

    # Generate timesteps for analysis
    tmin, tmax = 1, 1000
    print(f"\nGenerating timesteps between {tmin} to {tmax}.")
    timestep_sets = generate_timesteps(tmin, tmax)
    print(f"Generated timesteps: {timestep_sets}")
    # Debug the reverse process
    final_samples = debug_reverse_process(
        model, x0, timestep_sets["boundary"], args.num_samples
    )

    if final_samples is not None:
        print("\n" + "=" * 80)
        print("DEBUGGING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Final samples shape: {final_samples.shape}")
        print(
            f"Final samples range: [{final_samples.min().item():.4f}, {final_samples.max().item():.4f}]"
        )
        print(f"Final samples mean: {final_samples.mean().item():.6f}")
        print(f"Final samples std: {final_samples.std().item():.6f}")
    else:
        print("\n" + "=" * 80)
        print("DEBUGGING FAILED - INDEXING ISSUE DETECTED")
        print("=" * 80)


if __name__ == "__main__":
    main()
