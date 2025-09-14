#!/usr/bin/env python3
"""
Unit tests for the reverse diffusion process.

This module contains tests for the ReverseDiffusion class, verifying:
- Correct parameter handling
- Timestep indexing consistency
- Proper noise prediction integration
- Mathematical relationships in reverse step
- Edge cases (t=1, t=T)
- Deterministic behavior with fixed seeds

To run these tests:

1. From project root:
   pytest tests/test_reverse_diffusion.py

2. With specific test case:
   pytest tests/test_reverse_diffusion.py -k test_reverse_step_math

3. With verbosity:
   pytest tests/test_reverse_diffusion.py -v

4. With print output:
   pytest tests/test_reverse_diffusion.py -s

Note: Always run from the project root directory to ensure proper imports.
"""

import unittest

import torch

from src.forward_diffusion import ForwardDiffusion
from src.reverse_diffusion import ReverseDiffusion


class MockNoisePredictor(torch.nn.Module):
    """Mock noise predictor for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        """Return zeros as predicted noise."""
        return torch.zeros_like(x)


class TestReverseDiffusion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.diffusion_steps = 1000
        self.forward_diff = ForwardDiffusion(
            time_steps=self.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type="cosine",
        )

        # Create a mock noise predictor that returns zeros
        self.noise_predictor = MockNoisePredictor()

        # Data shape for testing (channels, seq_length)
        self.data_shape = (1, 10)

        # Create reverse diffusion instance
        self.reverse_diff = ReverseDiffusion(
            forward_diffusion=self.forward_diff,
            noise_predictor=self.noise_predictor,
            data_shape=self.data_shape,
        )

    def test_initialization(self):
        """Test that ReverseDiffusion initializes correctly."""
        self.assertEqual(self.reverse_diff.forward_diffusion, self.forward_diff)
        self.assertEqual(self.reverse_diff.noise_predictor, self.noise_predictor)
        self.assertEqual(self.reverse_diff.data_shape, self.data_shape)
        self.assertEqual(self.reverse_diff.denoise_step, 10)  # Default value
        self.assertEqual(self.reverse_diff.discretize, False)  # Default value

    def test_reverse_step_shape(self):
        """Test that reverse_diffusion_step preserves shape."""
        batch_size = 5
        x_t = torch.randn(batch_size, *self.data_shape)
        t = torch.full((batch_size,), 500, dtype=torch.long)

        x_t_minus_1 = self.reverse_diff.reverse_diffusion_step(x_t, t)

        # Check output shape matches input shape
        self.assertEqual(x_t_minus_1.shape, x_t.shape)

    def test_reverse_step_math(self):
        """Test mathematical relationships in reverse diffusion step."""
        batch_size = 3
        x_t = torch.randn(batch_size, *self.data_shape)
        t = torch.full((batch_size,), 500, dtype=torch.long)

        # Since our mock predictor returns zeros, we can predict the output
        # μ_θ(x_t, t) = (1/√α_t) * x_t when ε_θ(x_t, t) = 0

        # Get parameters for timestep t
        alpha_t = self.forward_diff.alpha(t).view(-1, 1, 1)

        # Compute expected mean
        expected_mean = x_t / torch.sqrt(alpha_t)

        # Get actual output (with zero noise prediction)
        x_t_minus_1 = self.reverse_diff.reverse_diffusion_step(x_t, t)

        # For t > 1, there's random noise added, so we can't check exact equality
        # But we can check if the values are in a reasonable range
        diff = torch.abs(x_t_minus_1 - expected_mean)
        self.assertTrue(torch.all(diff < 0.5))  # Reasonable bound for noise

    def test_edge_case_t1(self):
        """Test behavior at t=1 (final denoising step)."""
        batch_size = 3
        x_1 = torch.randn(batch_size, *self.data_shape)
        t = torch.full((batch_size,), 1, dtype=torch.long)

        # At t=1, no noise should be added
        x_0 = self.reverse_diff.reverse_diffusion_step(x_1, t)

        # Get parameters for timestep t=1
        alpha_1 = self.forward_diff.alpha(t).view(-1, 1, 1)

        # Compute expected x_0 (with zero noise prediction)
        expected_x_0 = x_1 / torch.sqrt(alpha_1)

        # Check if output matches expected value (within numerical precision)
        self.assertTrue(torch.allclose(x_0, expected_x_0, rtol=1e-5, atol=1e-5))

    def test_generate_samples(self):
        """Test generate_samples produces correct shape and values."""
        num_samples = 4
        samples = self.reverse_diff.generate_samples(num_samples=num_samples, seed=42)

        # Check shape
        expected_shape = (num_samples, *self.data_shape)
        self.assertEqual(samples.shape, expected_shape)

        # Check values are in valid range for SNP data (0 to 0.5)
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples <= 0.5))

    def test_denoise_sample(self):
        """Test denoise_sample works correctly."""
        batch_size = 3
        batch = torch.randn(batch_size, *self.data_shape)

        denoised = self.reverse_diff.denoise_sample(batch, seed=42)

        # Check shape
        self.assertEqual(denoised.shape, batch.shape)

        # Check values are in valid range for SNP data
        self.assertTrue(torch.all(denoised >= 0))
        self.assertTrue(torch.all(denoised <= 0.5))

    def test_timestep_indexing(self):
        """Test that timestep indexing is consistent between forward and reverse diffusion."""
        # Create a batch with all possible timesteps
        batch_size = 5
        x_t = torch.randn(batch_size, *self.data_shape)

        # Test a few specific timesteps
        for t_val in [1, 10, 100, 500, 999, 1000]:
            t = torch.full((batch_size,), t_val, dtype=torch.long)

            # These should not raise IndexError if indexing is consistent
            try:
                # Get parameters from forward diffusion
                beta_t = self.forward_diff.beta(t)
                alpha_t = self.forward_diff.alpha(t)
                alpha_bar_t = self.forward_diff.alpha_bar(t)
                sigma_t = self.forward_diff.sigma(t)

                # Verify direct array access matches method access
                # This confirms our indexing convention is consistent
                self.assertTrue(
                    torch.allclose(
                        beta_t, self.forward_diff._betas_t[t_val - 1].expand(batch_size)
                    ),
                    f"Beta mismatch at t={t_val}",
                )
                self.assertTrue(
                    torch.allclose(
                        alpha_t,
                        self.forward_diff._alphas_t[t_val - 1].expand(batch_size),
                    ),
                    f"Alpha mismatch at t={t_val}",
                )
                self.assertTrue(
                    torch.allclose(
                        alpha_bar_t,
                        self.forward_diff._alphas_bar_t[t_val].expand(batch_size),
                    ),
                    f"Alpha_bar mismatch at t={t_val}",
                )
                self.assertTrue(
                    torch.allclose(
                        sigma_t, self.forward_diff._sigmas_t[t_val].expand(batch_size)
                    ),
                    f"Sigma mismatch at t={t_val}",
                )

                # Try reverse step with these timesteps
                _ = self.reverse_diff.reverse_diffusion_step(x_t, t)
            except IndexError:
                self.fail(f"IndexError raised for timestep {t_val}")

    def test_deterministic_with_seed(self):
        """Test that generation is deterministic with fixed seed."""
        num_samples = 3

        # Generate samples twice with the same seed
        samples1 = self.reverse_diff.generate_samples(num_samples=num_samples, seed=123)
        samples2 = self.reverse_diff.generate_samples(num_samples=num_samples, seed=123)

        # They should be identical
        self.assertTrue(torch.allclose(samples1, samples2))

        # Generate with a different seed
        samples3 = self.reverse_diff.generate_samples(num_samples=num_samples, seed=456)

        # They should be different
        self.assertFalse(torch.allclose(samples1, samples3))


if __name__ == "__main__":
    unittest.main()
