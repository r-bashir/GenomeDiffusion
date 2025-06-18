#!/usr/bin/env python3
"""
Unit tests for the forward diffusion process.

This module contains tests for the ForwardDiffusion class, verifying:
- Array lengths and shapes
- Value ranges and constraints
- Indexing consistency
- Edge cases (t=0, t=T)
- Mathematical relationships between parameters

To run these tests:

1. From project root with PYTHONPATH:
   pytest tests/test_forward_diffusion.py

2. With specific test case:
   pytest tests/test_forward_diffusion.py -k test_array_lengths

3. With verbosity:
   pytest tests/test_forward_diffusion.py -v

4. With print output:
   pytest tests/test_forward_diffusion.py -s

Note: Always run from the project root directory to ensure proper imports.
"""

import unittest

import numpy as np
import torch

from src.forward_diffusion import ForwardDiffusion


class TestForwardDiffusion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.diffusion_steps = 1000
        self.forward_diff = ForwardDiffusion(
            diffusion_steps=self.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type="cosine",
        )

    def test_array_lengths(self):
        """Test that all arrays have correct lengths."""
        # betas and alphas should have length T
        self.assertEqual(len(self.forward_diff.betas), self.diffusion_steps)
        self.assertEqual(len(self.forward_diff.alphas), self.diffusion_steps)

        # alphas_bar and sigmas should have length T+1 (including t=0)
        self.assertEqual(len(self.forward_diff.alphas_bar), self.diffusion_steps + 1)
        self.assertEqual(len(self.forward_diff.sigmas), self.diffusion_steps + 1)

    def test_value_ranges(self):
        """Test that values are within expected ranges."""
        # betas should be between 0 and 1
        self.assertTrue(torch.all(self.forward_diff.betas >= 0))
        self.assertTrue(torch.all(self.forward_diff.betas <= 1))

        # alphas = 1 - betas, so should be between 0 and 1
        self.assertTrue(torch.all(self.forward_diff.alphas >= 0))
        self.assertTrue(torch.all(self.forward_diff.alphas <= 1))

        # alphas_bar should be monotonically decreasing from 1 to near 0
        self.assertAlmostEqual(self.forward_diff.alphas_bar[0].item(), 1.0, places=5)
        self.assertTrue(
            torch.all(
                self.forward_diff.alphas_bar[1:] <= self.forward_diff.alphas_bar[:-1]
            )
        )

        # sigmas should start at 0 and increase
        self.assertAlmostEqual(self.forward_diff.sigmas[0].item(), 0.0, places=5)
        self.assertTrue(
            torch.all(self.forward_diff.sigmas[1:] >= self.forward_diff.sigmas[:-1])
        )

    def test_indexing_consistency(self):
        """Test that indexing is consistent across all timesteps."""
        # Create batch of all possible timesteps
        t = torch.arange(1, self.diffusion_steps + 1)

        # Test beta indexing
        betas_indexed = self.forward_diff.beta(t)
        self.assertTrue(torch.allclose(betas_indexed, self.forward_diff.betas))

        # Test alpha indexing
        alphas_indexed = self.forward_diff.alpha(t)
        self.assertTrue(torch.allclose(alphas_indexed, self.forward_diff.alphas))

        # Test alpha_bar indexing
        alpha_bar_indexed = self.forward_diff.alpha_bar(t)
        self.assertTrue(
            torch.allclose(alpha_bar_indexed, self.forward_diff.alphas_bar[1:])
        )

        # Test sigma indexing
        sigma_indexed = self.forward_diff.sigma(t)
        self.assertTrue(torch.allclose(sigma_indexed, self.forward_diff.sigmas[1:]))

    def test_edge_cases(self):
        """Test behavior at t=1 and t=1000."""
        # Test t=1
        t_start = torch.tensor([1])
        self.assertEqual(self.forward_diff.beta(t_start).shape, (1,))
        self.assertEqual(self.forward_diff.alpha(t_start).shape, (1,))
        self.assertEqual(self.forward_diff.alpha_bar(t_start).shape, (1,))
        self.assertEqual(self.forward_diff.sigma(t_start).shape, (1,))

        # Test t=1000
        t_end = torch.tensor([1000])
        self.assertEqual(self.forward_diff.beta(t_end).shape, (1,))
        self.assertEqual(self.forward_diff.alpha(t_end).shape, (1,))
        self.assertEqual(self.forward_diff.alpha_bar(t_end).shape, (1,))
        self.assertEqual(self.forward_diff.sigma(t_end).shape, (1,))

    def test_relationships(self):
        """Test mathematical relationships between parameters."""
        # Test alpha = 1 - beta
        self.assertTrue(
            torch.allclose(self.forward_diff.alphas, 1 - self.forward_diff.betas)
        )

        # Test alphas_bar is cumulative product of alphas
        alphas_cumprod = torch.cat(
            [torch.tensor([1.0]), torch.cumprod(self.forward_diff.alphas, dim=0)]
        )
        self.assertTrue(torch.allclose(self.forward_diff.alphas_bar, alphas_cumprod))

        # Test sigma = sqrt(1 - alpha_bar)
        self.assertTrue(
            torch.allclose(
                self.forward_diff.sigmas,
                torch.sqrt(1 - self.forward_diff.alphas_bar),
                rtol=1e-5,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
