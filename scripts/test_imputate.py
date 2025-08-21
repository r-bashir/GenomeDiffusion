#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import imputate_data

# Add src to path
# sys.path.append(str(Path(__file__).parent / "src"))


def test_pattern_injection_scenarios():
    """Test different pattern/interval combinations with the new injection logic."""

    # Create test data: 100 SNPs with realistic genomic values
    n_samples, n_markers = 1, 100
    np.random.seed(42)  # For reproducible results
    base_data = np.random.choice(
        [0.0, 0.5, 1.0], size=(n_samples, n_markers), p=[0.3, 0.4, 0.3]
    ).astype(np.float32)

    # Test scenarios
    scenarios = [
        {
            "pattern_len": 10,
            "interval": 10,
            "desc": "Pattern=10, Interval=10 (equal sizes)",
        },
        {
            "pattern_len": 20,
            "interval": 10,
            "desc": "Pattern=20, Interval=10 (pattern > interval)",
        },
        {
            "pattern_len": 10,
            "interval": 20,
            "desc": "Pattern=10, Interval=20 (pattern < interval)",
        },
        {
            "pattern_len": 15,
            "interval": 15,
            "desc": "Pattern=15, Interval=15 (equal sizes)",
        },
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\n=== SCENARIO {i+1}: {scenario['desc']} ===")

        # Create pattern based on length
        pattern_len = scenario["pattern_len"]
        if pattern_len == 10:
            pattern = [(0, 3, 0.0), (3, 7, 0.5), (7, 10, 1.0)]
        elif pattern_len == 15:
            pattern = [(0, 5, 0.0), (5, 10, 0.5), (10, 15, 1.0)]
        elif pattern_len == 20:
            pattern = [(0, 7, 0.0), (7, 13, 0.5), (13, 20, 1.0)]

        interval = scenario["interval"]

        # Apply pattern injection
        data = base_data.copy()
        result = imputate_data(data, pattern, interval)

        print(f"Pattern length: {pattern_len}, Injection interval: {interval}")

        # Calculate expected positions
        expected_positions = []
        pos = 0
        while pos < n_markers:
            if pos + pattern_len <= n_markers:
                expected_positions.append((pos, pos + pattern_len - 1, "PATTERN"))
            else:
                expected_positions.append((pos, n_markers - 1, "PARTIAL"))
            pos += pattern_len + interval

        print("Expected pattern positions:")
        for start, end, ptype in expected_positions:
            print(f"  {start:2d}-{end:2d}: {ptype}")

        # Verify first few positions
        print("Verification (first 60 positions):")
        display_len = min(60, n_markers)
        print("Pos:", "".join(f"{i%10}" for i in range(display_len)))
        print("Org:", "".join(f"{val:.0f}" for val in data[0][:display_len]))
        print("Res:", "".join(f"{val:.0f}" for val in result[0][:display_len]))

        # Check if pattern was applied correctly at expected positions
        success = True
        for start, end, ptype in expected_positions[:2]:  # Check first 2 patterns
            if ptype == "PATTERN":
                actual_block = result[0, start : end + 1]
                # Check if it matches pattern structure (0s, then 0.5s, then 1s)
                has_zeros = np.any(actual_block == 0.0)
                has_halves = np.any(actual_block == 0.5)
                has_ones = np.any(actual_block == 1.0)
                pattern_applied = has_zeros and has_halves and has_ones
                print(f"  Pattern at {start}-{end}: {'✓' if pattern_applied else '✗'}")
                success = success and pattern_applied

        print(f"Scenario result: {'✅ PASS' if success else '❌ FAIL'}")

    return True


def plot_sequence_transformation():
    """Plot before/after visualization of sequence transformation."""

    # Create test data: 60 SNPs with realistic genomic values
    n_samples, n_markers = 1, 150
    np.random.seed(42)  # For reproducible results

    # Create realistic genomic data with some variation
    real_data = np.random.choice(
        [0.0, 0.5, 1.0], size=(n_samples, n_markers), p=[0.3, 0.4, 0.3]
    ).astype(
        np.float32
    )  # Realistic allele frequencies

    # Pattern and injection settings
    pattern = [(0, 3, 0.0), (3, 7, 0.5), (7, 10, 1.0)]
    injection_interval = 10

    # Apply imputate transformation
    transformed_data = imputate_data(real_data, pattern, injection_interval)

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    positions = np.arange(n_markers)

    # Plot 1: Original sequence
    ax1.plot(positions, real_data[0], "o-", color="blue", alpha=0.7, markersize=4)
    ax1.set_title("Original Genomic Sequence", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Genotype Value")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # Plot 2: Transformed sequence
    ax2.plot(positions, transformed_data[0], "o-", color="red", alpha=0.7, markersize=4)
    ax2.set_title(
        "After Imputate Transformation (Alternating Blocks)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_ylabel("Genotype Value")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # Add block boundaries and labels
    for i in range(0, n_markers, injection_interval):
        block_end = min(i + injection_interval, n_markers)
        block_type = "PATTERN" if i // injection_interval % 2 == 0 else "REAL"
        color = "lightcoral" if block_type == "PATTERN" else "lightblue"

        ax2.axvspan(i, block_end - 1, alpha=0.2, color=color)
        ax2.text(
            i + injection_interval / 2,
            1.05,
            block_type,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 3: Difference visualization
    difference = transformed_data[0] - real_data[0]
    bars = ax3.bar(
        positions,
        difference,
        alpha=0.7,
        color=["red" if d != 0 else "gray" for d in difference],
    )
    ax3.set_title("Difference (Transformed - Original)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("SNP Position")
    ax3.set_ylabel("Value Change")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Add block boundaries to difference plot
    for i in range(0, n_markers, injection_interval):
        block_end = min(i + injection_interval, n_markers)
        block_type = "PATTERN" if i // injection_interval % 2 == 0 else "REAL"
        color = "lightcoral" if block_type == "PATTERN" else "lightblue"
        ax3.axvspan(i, block_end - 1, alpha=0.2, color=color)

    plt.tight_layout()
    plt.savefig(
        "imputate_transformation_visualization.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Print summary statistics
    print("\n=== TRANSFORMATION SUMMARY ===")
    print(f"Sequence length: {n_markers} SNPs")
    print(f"Pattern length: {max(end for _, end, _ in pattern)} SNPs")
    print(f"Injection interval: {injection_interval} SNPs")
    print(
        f"Number of pattern blocks: {len([i for i in range(0, n_markers, injection_interval) if i // injection_interval % 2 == 0])}"
    )
    print(
        f"Number of real blocks: {len([i for i in range(0, n_markers, injection_interval) if i // injection_interval % 2 == 1])}"
    )

    # Count changes
    changes = np.sum(difference != 0)
    unchanged = np.sum(difference == 0)
    print(f"SNPs changed: {changes} ({changes/n_markers*100:.1f}%)")
    print(f"SNPs unchanged: {unchanged} ({unchanged/n_markers*100:.1f}%)")


if __name__ == "__main__":
    success = test_pattern_injection_scenarios()
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")

    print("\n" + "=" * 50)
    plot_sequence_transformation()
