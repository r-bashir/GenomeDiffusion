#!/usr/bin/env python
# coding: utf-8

"""Test script for the updated wrapper functions in infer_utils.py.

This script tests the generate_samples() and denoise_sample() wrapper functions
to ensure they produce identical results to the original functions when using
the same parameters (start_timestep=tmax).
"""

from pathlib import Path

import torch

from src.ddpm import DiffusionModel
from src.infer_utils import denoise_sample, generate_samples
from src.utils import set_seed


def test_generate_samples_parity(model, device):
    """Test that wrapper generate_samples() produces identical results to original."""
    print("\n=== Testing generate_samples() Parity ===")

    # Set seed for reproducibility
    set_seed(42)

    # Test parameters
    num_samples = 5
    denoise_step = 1
    discretize = False

    # Generate using original method (from tmax)
    print("1. Generating samples using original model.generate_samples()...")
    with torch.no_grad():
        original_samples = model.generate_samples(
            num_samples=num_samples,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=42,
            device=device,
        )

    # Generate using wrapper with start_timestep=tmax (should be identical)
    print("2. Generating samples using wrapper with start_timestep=tmax...")
    set_seed(42)  # Reset seed to ensure identical randomness
    with torch.no_grad():
        wrapper_samples = generate_samples(
            model=model,
            num_samples=num_samples,
            start_timestep=model.forward_diffusion.tmax,  # Same as original
            denoise_step=denoise_step,
            discretize=discretize,
            seed=42,
        )

    # Compare results
    print("Original samples shape: {}".format(original_samples.shape))
    print("Wrapper samples shape: {}".format(wrapper_samples.shape))

    # Check if results are identical (or very close due to floating point precision)
    mse = torch.mean((original_samples - wrapper_samples) ** 2).item()
    max_diff = torch.max(torch.abs(original_samples - wrapper_samples)).item()

    print("MSE between original and wrapper: {:.10f}".format(mse))
    print("Max absolute difference: {:.10f}".format(max_diff))

    if mse < 1e-6 and max_diff < 1e-6:
        print("✅ PASS: Wrapper produces identical results to original!")
        return True
    else:
        print("❌ FAIL: Wrapper produces different results from original!")
        return False


def test_generate_samples_flexibility(model, device):
    """Test that wrapper can generate from arbitrary timesteps."""
    print("\n=== Testing generate_samples() Flexibility ===")

    # Test different starting timesteps
    timesteps_to_test = [1000, 500, 100, 50]

    for start_t in timesteps_to_test:
        print("\nGenerating from timestep {}...".format(start_t))
        set_seed(42)

        try:
            with torch.no_grad():
                samples = generate_samples(
                    model=model,
                    num_samples=2,
                    start_timestep=start_t,
                    denoise_step=1,
                    discretize=False,
                    seed=42,
                )

            print(
                "✅ Successfully generated samples from t={}, shape: {}".format(
                    start_t, samples.shape
                )
            )

            # Check sample statistics
            mean_val = torch.mean(samples).item()
            std_val = torch.std(samples).item()
            min_val = torch.min(samples).item()
            max_val = torch.max(samples).item()

            print(
                "   Sample stats: mean={:.4f}, std={:.4f}, range=[{:.4f}, {:.4f}]".format(
                    mean_val, std_val, min_val, max_val
                )
            )

        except Exception as e:
            print("❌ Failed to generate from t={}: {}".format(start_t, e))
            return False

    print("✅ PASS: Wrapper successfully generates from arbitrary timesteps!")
    return True


def test_denoise_sample_parity(model, device):
    """Test that wrapper denoise_sample() produces identical results to original."""
    print("\n=== Testing denoise_sample() Parity ===")

    # Create a test batch (simulating noisy data)
    batch_size = 3
    test_batch = torch.randn(batch_size, 1, model._data_shape[1], device=device) * 0.1
    test_batch = torch.clamp(test_batch, 0.0, 0.5)  # Clamp to valid SNP range

    # Test parameters
    denoise_step = 1
    discretize = False

    # Denoise using original method
    print("1. Denoising using original model.denoise_sample()...")
    set_seed(42)
    with torch.no_grad():
        original_denoised = model.denoise_sample(
            batch=test_batch,
            denoise_step=denoise_step,
            discretize=discretize,
            seed=42,
            device=device,
        )

    # Denoise using wrapper with start_timestep=tmax (should be identical)
    print("2. Denoising using wrapper with start_timestep=tmax...")
    set_seed(42)  # Reset seed
    with torch.no_grad():
        wrapper_denoised = denoise_sample(
            model=model,
            batch=test_batch,
            start_timestep=model.forward_diffusion.tmax,  # Same as original
            denoise_step=denoise_step,
            discretize=discretize,
            seed=42,
        )

    # Compare results
    print("Original denoised shape: {}".format(original_denoised.shape))
    print("Wrapper denoised shape: {}".format(wrapper_denoised.shape))

    # Check if results are identical
    mse = torch.mean((original_denoised - wrapper_denoised) ** 2).item()
    max_diff = torch.max(torch.abs(original_denoised - wrapper_denoised)).item()

    print("MSE between original and wrapper: {:.10f}".format(mse))
    print("Max absolute difference: {:.10f}".format(max_diff))

    if mse < 1e-6 and max_diff < 1e-6:
        print("✅ PASS: Wrapper produces identical results to original!")
        return True
    else:
        print("❌ FAIL: Wrapper produces different results from original!")
        return False


def test_denoise_sample_flexibility(model, device):
    """Test that wrapper can denoise from arbitrary timesteps."""
    print("\n=== Testing denoise_sample() Flexibility ===")

    # Create test data
    test_batch = torch.randn(2, 1, model._data_shape[1], device=device) * 0.1
    test_batch = torch.clamp(test_batch, 0.0, 0.5)

    # Test different starting timesteps
    timesteps_to_test = [1000, 500, 200, 50]

    for start_t in timesteps_to_test:
        print("\nDenoising from timestep {}...".format(start_t))
        set_seed(42)

        try:
            with torch.no_grad():
                denoised = denoise_sample(
                    model=model,
                    batch=test_batch,
                    start_timestep=start_t,
                    denoise_step=1,
                    discretize=False,
                    seed=42,
                )

            print(
                "✅ Successfully denoised from t={}, shape: {}".format(
                    start_t, denoised.shape
                )
            )

            # Check sample statistics
            mean_val = torch.mean(denoised).item()
            std_val = torch.std(denoised).item()
            min_val = torch.min(denoised).item()
            max_val = torch.max(denoised).item()

            print(
                "   Denoised stats: mean={:.4f}, std={:.4f}, range=[{:.4f}, {:.4f}]".format(
                    mean_val, std_val, min_val, max_val
                )
            )

        except Exception as e:
            print("❌ Failed to denoise from t={}: {}".format(start_t, e))
            return False

    print("✅ PASS: Wrapper successfully denoises from arbitrary timesteps!")
    return True


def main():
    """Main test function."""
    print("Testing Updated Wrapper Functions in infer_utils.py")
    print("=" * 60)

    # Set global seed
    set_seed(42)

    # Load model
    checkpoint_path = "/home/rbashir/Semester/Thesis/GenomeDiffusion/outputs/lightning_logs/TestDiff/puz14yp1/checkpoints/last.ckpt"

    try:
        print("Loading model from: {}".format(checkpoint_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DiffusionModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=True,
        )
        model = model.to(device)
        model.eval()

        print("✅ Model loaded successfully on {}".format(device))
        print(
            "Model config: tmax={}, tmin={}".format(
                model.forward_diffusion.tmax, model.forward_diffusion.tmin
            )
        )
        print("Data shape: {}".format(model._data_shape))

    except Exception as e:
        print("❌ Failed to load model: {}".format(e))
        return

    # Run tests
    all_tests_passed = True

    # Test generate_samples()
    all_tests_passed &= test_generate_samples_parity(model, device)
    all_tests_passed &= test_generate_samples_flexibility(model, device)

    # Test denoise_sample()
    all_tests_passed &= test_denoise_sample_parity(model, device)
    all_tests_passed &= test_denoise_sample_flexibility(model, device)

    # Final summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Wrapper functions are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
