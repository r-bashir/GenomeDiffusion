#!/usr/bin/env python
# coding: utf-8

"""
Test script for comprehensive evaluation metrics.

This script demonstrates how to use the comprehensive evaluation framework
to compare real and generated genomic samples with various quality metrics.
"""

from pathlib import Path

import numpy as np
import torch

from src.evaluation_metrics import (
    compute_comprehensive_evaluation,
    print_evaluation_summary,
)
from src.utils import set_seed


def create_test_data():
    """Create synthetic test data to demonstrate evaluation metrics."""
    set_seed(42)

    # Create realistic SNP-like data
    n_samples = 100
    seq_length = 2067

    # Real data: realistic allele frequencies with some structure
    real_af = np.random.beta(
        2, 5, seq_length
    )  # Skewed toward lower frequencies (realistic for MAF)
    real_samples = []

    for i in range(n_samples):
        sample = np.random.binomial(2, real_af) / 2.0  # Convert to 0, 0.5, 1.0 scale
        real_samples.append(sample)

    real_samples = np.array(real_samples)

    # Generated data scenarios
    scenarios = {}

    # Scenario 1: Perfect match (same distribution)
    np.random.seed(43)  # Different seed for variation
    perfect_samples = []
    for i in range(n_samples):
        sample = np.random.binomial(2, real_af) / 2.0
        perfect_samples.append(sample)
    scenarios["perfect"] = np.array(perfect_samples)

    # Scenario 2: Good match (similar but with some noise)
    np.random.seed(44)
    good_af = real_af + np.random.normal(0, 0.05, seq_length)  # Add small noise
    good_af = np.clip(good_af, 0.01, 0.49)  # Keep in valid range
    good_samples = []
    for i in range(n_samples):
        sample = np.random.binomial(2, good_af) / 2.0
        good_samples.append(sample)
    scenarios["good"] = np.array(good_samples)

    # Scenario 3: Poor match (different distribution)
    np.random.seed(45)
    poor_af = np.random.beta(5, 2, seq_length)  # Opposite skew
    poor_samples = []
    for i in range(n_samples):
        sample = np.random.binomial(2, poor_af) / 2.0
        poor_samples.append(sample)
    scenarios["poor"] = np.array(poor_samples)

    # Scenario 4: Random noise
    np.random.seed(46)
    random_samples = np.random.choice([0.0, 0.5, 1.0], size=(n_samples, seq_length))
    scenarios["random"] = random_samples

    return real_samples, scenarios


def test_evaluation_metrics():
    """Test the comprehensive evaluation metrics on different scenarios."""
    print("🧪 Testing Comprehensive Evaluation Metrics")
    print("=" * 60)

    # Create test data
    real_samples, scenarios = create_test_data()

    # Test each scenario
    results = {}

    for scenario_name, gen_samples in scenarios.items():
        print(f"\n🔍 Testing Scenario: {scenario_name.upper()}")
        print("-" * 40)

        # Create output directory for this scenario
        output_dir = Path(f"test_evaluation_output/{scenario_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute comprehensive evaluation
        metrics = compute_comprehensive_evaluation(
            real_samples, gen_samples, output_dir, max_value=0.5
        )

        # Print summary
        print_evaluation_summary(metrics)

        # Store results
        results[scenario_name] = metrics["overall_quality_score"]

        print(f"\n📁 Visualizations saved to: {output_dir}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("📊 SCENARIO COMPARISON SUMMARY")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (scenario, score) in enumerate(sorted_results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{emoji} {i}. {scenario.upper():<10} - Quality Score: {score:.3f}")

    print("\n✅ Evaluation metrics testing completed!")
    print("📁 All results saved to: test_evaluation_output/")

    return results


if __name__ == "__main__":
    test_evaluation_metrics()
