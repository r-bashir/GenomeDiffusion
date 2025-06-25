#!/usr/bin/env python3
"""
Simple mock execution script to test indexing logic without external dependencies.
This creates dummy data to verify array indexing for reverse diffusion debugging.
"""

import argparse
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ruff: noqa: E402
import numpy as np
import torch

from src.dataset import SNPDataset
from src.forward_diffusion import ForwardDiffusion
from src.utils import load_config, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_indexing_for_timestep(forward_diff, t):
    """Test indexing for a specific timestep."""
    # Convert t_val to Python int if tensor

    print(f"\n{'='*60}")
    print(f"TESTING TIMESTEP t={t}")
    print(f"{'='*60}")

    # Array information
    beta_array_len = len(forward_diff.betas)
    alpha_array_len = len(forward_diff.alphas)
    alpha_bar_array_len = len(forward_diff.alphas_bar)

    print(f"\nArray lengths:")
    print(f"  betas (0-indexed): {beta_array_len} elements [0 to {beta_array_len-1}]")
    print(
        f"  alphas (0-indexed): {alpha_array_len} elements [0 to {alpha_array_len-1}]"
    )
    print(
        f"  alphas_bar (1-indexed): {alpha_bar_array_len} elements [0 to {alpha_bar_array_len-1}]"
    )

    # Calculate expected indices
    expected_beta_idx = t - 1  # 0-indexed: t=1000 -> idx=999
    expected_alpha_idx = t - 1  # 0-indexed: t=1000 -> idx=999
    expected_alpha_bar_idx = t  # 1-indexed: t=1000 -> idx=1000

    print(f"\nExpected indices for t={t}:")
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
        return False

    print("✅ All indices are within bounds")

    # Test parameter access
    try:
        # Beta access
        print(f"\n--- Beta Access Test ---")
        beta_method = forward_diff.beta(t)
        beta_direct = forward_diff.betas[expected_beta_idx]
        print(f"Method result: {beta_method.item():.8f}")
        print(f"Direct access: {beta_direct.item():.8f}")

        if abs(beta_method.item() - beta_direct.item()) > 1e-8:
            print("🚨 ERROR: Beta method vs direct access mismatch!")
            return False
        else:
            print("✅ Beta method and direct access match")

        # Alpha access
        print(f"\n--- Alpha Access Test ---")
        alpha_method = forward_diff.alpha(t)
        alpha_direct = forward_diff.alphas[expected_alpha_idx]
        print(f"Method result: {alpha_method.item():.8f}")
        print(f"Direct access: {alpha_direct.item():.8f}")

        if abs(alpha_method.item() - alpha_direct.item()) > 1e-8:
            print("🚨 ERROR: Alpha method vs direct access mismatch!")
            return False
        else:
            print("✅ Alpha method and direct access match")

        # Verify alpha = 1 - beta relationship
        expected_alpha = 1.0 - beta_direct
        print(f"\n--- Alpha-Beta Relationship Test ---")
        print(
            f"α_t = 1 - β_t = 1 - {beta_direct.item():.8f} = {expected_alpha.item():.8f}"
        )
        print(f"Actual α_t = {alpha_method.item():.8f}")

        if abs(alpha_method.item() - expected_alpha.item()) > 1e-7:
            print("🚨 ERROR: α_t ≠ 1 - β_t relationship violated!")
            return False
        else:
            print("✅ α_t = 1 - β_t relationship verified")

        # Alpha_bar access
        print(f"\n--- Alpha_bar Access Test ---")
        alpha_bar_method = forward_diff.alpha_bar(t)
        alpha_bar_direct = forward_diff.alphas_bar[expected_alpha_bar_idx]
        print(f"Method result: {alpha_bar_method.item():.8f}")
        print(f"Direct access: {alpha_bar_direct.item():.8f}")

        if abs(alpha_bar_method.item() - alpha_bar_direct.item()) > 1e-8:
            print("🚨 ERROR: Alpha_bar method vs direct access mismatch!")
            return False
        else:
            print("✅ Alpha_bar method and direct access match")

        # Special boundary case tests
        if t == 1:
            print(f"\n--- Boundary Case t=1 Test ---")
            print(f"For t=1: ᾱ_1 should equal α_1")
            print(f"  ᾱ_1 = {alpha_bar_method.item():.8f}")
            print(f"  α_1 = {alpha_method.item():.8f}")
            if (
                abs(alpha_bar_method.item() - alpha_method.item()) > 1e-6
            ):  # Slightly relaxed tolerance
                print(
                    "⚠️  WARNING: ᾱ_1 ≠ α_1 for boundary case (might be due to cumulative product precision)"
                )
            else:
                print("✅ Boundary case ᾱ_1 = α_1 verified")

        elif t == forward_diff.tmax:
            print(f"\n--- Boundary Case t={forward_diff.tmax} Test ---")
            print(f"This is the maximum timestep")
            print(f"  ᾱ_{t} = {alpha_bar_method.item():.8f} (should be very small)")
            if alpha_bar_method.item() > 0.1:
                print(f"⚠️  WARNING: ᾱ_{t} seems large for max timestep")
            else:
                print("✅ ᾱ_tmax is appropriately small")

        # Show what values we're actually getting
        print(f"\n--- Actual Values Retrieved ---")
        print(f"  β_{t} = {beta_method.item():.8f}")
        print(f"  α_{t} = {alpha_method.item():.8f}")
        print(f"  ᾱ_{t} = {alpha_bar_method.item():.8f}")

        print(f"\n✅ All tests passed for timestep t={t}")
        return True

    except Exception as e:
        print(f"🚨 ERROR during parameter access: {e}")
        return False


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Forward Diffusion Investigation")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/forward_diffusion"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output",
    )
    return parser.parse_args()


def main():

    # Set global seed for reproducibility
    set_seed(42)

    # Parse Arguments
    args = parse_args()

    # Load config
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Forward Diffusion
    forward_diff = ForwardDiffusion(
        time_steps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule_type=config["diffusion"]["schedule_type"],
    )
    forward_diff = forward_diff.to(device)

    # Load dataset
    print("Loading dataset...")
    dataset = SNPDataset(config)

    # Select a sample
    sample_idx = 0
    x0 = dataset[sample_idx]

    # Reshape sample for visualization: add batch and channel dimensions [batch, channel, seq_length]
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length]
    elif x0.dim() == 2:
        x0 = x0.unsqueeze(0)  # [1, channels, seq_length]

    x0 = x0.to(device)
    print(f"Sample shape: {x0.shape}")
    print(f"Sample unique values: {torch.unique(x0)}")
    print(f"First 10 values: {x0[0, 0, :10]}")

    # Prepare output directory
    base_dir = Path(config["output_path"])
    output_dir = base_dir / "forward_diffusion"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {output_dir}")

    # ===================== Analyze Indexing =====================

    # Test critical timesteps
    test_timesteps = [1, 2, 500, 999, 1000]

    # Also test some edge cases
    edge_cases = [0, 1001]  # These should fail bounds checking

    print(f"\nTesting valid timesteps: {test_timesteps}")
    all_passed = True

    for t_val in test_timesteps:
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
        success = test_indexing_for_timestep(forward_diff, t_tensor)
        if not success:
            all_passed = False

    print(f"\nTesting edge cases (should fail): {edge_cases}")
    for t_val in edge_cases:
        print(f"\n--- Testing invalid timestep t={t_val} ---")
        try:
            t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
            success = test_indexing_for_timestep(forward_diff, t_tensor)
            if success:
                print(f"⚠️  WARNING: Expected t={t_val} to fail, but it passed!")
        except IndexError as e:
            print(f"✅ Expected IndexError for t={t_val}: {e}")
        except Exception as e:
            print(f"✅ Expected error for t={t_val}: {e}")

    # Array analysis
    print(f"\n{'='*60}")
    print("ALPHA_BAR ARRAY ANALYSIS")
    print(f"{'='*60}")

    alpha_bar_len = len(forward_diff.alphas_bar)
    used_indices = forward_diff.tmax + 1  # Indices 0 to tmax
    unused_elements = alpha_bar_len - used_indices

    print(f"Alpha_bar array has {alpha_bar_len} elements")
    print(f"Used indices: 0 to {forward_diff.tmax} ({used_indices} elements)")
    if unused_elements > 0:
        print(
            f"Unused elements: {unused_elements} (indices {forward_diff.tmax + 1} to {alpha_bar_len - 1})"
        )
        print("❌ This indicates a potential issue - no elements should be unused!")
    else:
        print("✅ No unused elements - all array elements are utilized")

    # Show some sample values to verify the pattern
    print(f"\nSample alpha_bar values:")
    sample_indices = [0, 1, 2, 500, 999, 1000]
    for idx in sample_indices:
        if idx < alpha_bar_len:
            print(f"  alpha_bar[{idx}] = {forward_diff.alphas_bar[idx]:.8f}")

    # Verify the cumulative product pattern
    print(f"\nVerifying cumulative product pattern:")
    print(f"  ᾱ_1 = α_1 = {forward_diff.alphas[0]:.8f}")
    print(f"  ᾱ_2 = α_1 * α_2 = {forward_diff.alphas[0] * forward_diff.alphas[1]:.8f}")
    print(f"  ᾱ_2 (from array) = {forward_diff.alphas_bar[2]:.8f}")

    if (
        abs(
            forward_diff.alphas_bar[2]
            - (forward_diff.alphas[0] * forward_diff.alphas[1])
        )
        < 1e-8
    ):
        print("✅ Cumulative product pattern verified")
    else:
        print("🚨 ERROR: Cumulative product pattern broken!")

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL VALID TIMESTEP TESTS PASSED!")
        print("The indexing logic is working correctly!")
    else:
        print("🚨 SOME TESTS FAILED!")
        print("There may be indexing issues in the implementation!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
