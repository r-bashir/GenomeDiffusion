#!/usr/bin/env python
# coding: utf-8
# ruff: noqa: E402

"""
Test the functionality of SNP Dataset from src/dataset.py
by running: python scripts/run_dataset.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    SNPDataset,
    handle_missing_values,
    load_data,
    mix_data,
    mix_data_flip_odd,
    normalize_data,
    scale_data,
    setup_logging,
    staircase_data,
)
from src.utils import load_config


# Print data statistics
def plot_sample(sample: torch.Tensor, save_path: Path) -> None:
    """Plot a SNP sample.

    Args:
        sample: A SNP sample.
        save_path: Path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    ax.plot(sample, "o-", color="blue", alpha=0.7, markersize=3)
    ax.set_title("Sample visualization (first 100 SNPs)")
    ax.set_xlabel("SNP Position")
    ax.set_ylabel("Genotype Values")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_first_n_samples(
    dataset: torch.utils.data.Dataset, save_path: Path, n: int = 10
) -> None:
    """Plot the first n samples in a grid to inspect augmentations.

    Creates a 5x2 grid (for n=10) with labels indicating index and parity
    so you can confirm flipped patterns on odd-indexed samples.

    Args:
        dataset: SNPDataset instance
        save_path: Path to save the combined plot
        n: Number of samples to plot (default 10)
    """
    n = min(n, len(dataset))
    if n == 0:
        return

    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2 * rows), squeeze=False)

    for i in range(n):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        sample = dataset[i]
        ax.plot(
            sample,
            "-",
            color=("tab:blue" if i % 2 == 0 else "tab:orange"),
            linewidth=0.8,
            alpha=0.8,
        )
        parity = "even" if i % 2 == 0 else "odd"
        ax.set_title(f"idx {i} ({parity})")
        ax.grid(True, alpha=0.3)
        if r == rows - 1:
            ax.set_xlabel("SNP Position")
        ax.set_ylabel("Genotype")

    # Hide any unused subplots
    total_axes = rows * cols
    for j in range(n, total_axes):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    fig.suptitle("First N Samples (check even vs odd augmentations)", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_even_odd_samples(dataset: torch.utils.data.Dataset, save_dir: Path) -> None:
    """Randomly select and plot one even-indexed and one odd-indexed sample.

    Saves three plots:
    - even_sample.png
    - odd_sample.png
    - even_vs_odd_overlay.png
    """
    n = len(dataset)
    if n == 0:
        return

    even_indices = np.arange(0, n, 2)
    odd_indices = np.arange(1, n, 2)

    even_idx = int(np.random.choice(even_indices)) if even_indices.size > 0 else None
    odd_idx = int(np.random.choice(odd_indices)) if odd_indices.size > 0 else None

    if even_idx is not None:
        plot_sample(dataset[even_idx], save_dir / "even_sample.png")
    if odd_idx is not None:
        plot_sample(dataset[odd_idx], save_dir / "odd_sample.png")

    # Overlay plot if both are available
    if even_idx is not None and odd_idx is not None:
        even_sample = dataset[even_idx]
        odd_sample = dataset[odd_idx]

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(
            even_sample,
            "-",
            color="tab:blue",
            alpha=0.7,
            linewidth=1.0,
            label=f"even idx {even_idx}",
        )
        ax.plot(
            odd_sample,
            "-",
            color="tab:orange",
            alpha=0.7,
            linewidth=1.0,
            label=f"odd idx {odd_idx}",
        )
        ax.set_title("Even vs Odd Sample (overlay)")
        ax.set_xlabel("SNP Position")
        ax.set_ylabel("Genotype Values")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(save_dir / "even_vs_odd_overlay.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def print_data_stats(data: Union[np.ndarray, torch.Tensor], title: str = "") -> None:
    """Print statistics about the data.

    Args:
        data: Data array to analyze
        title: Title for the statistics section
    """
    if title:
        print(f"\n{'='*65}")
        print(f"{title}")
        print(f"{'='*65}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min: {data.min()}, Max: {data.max()}")
    print(f"Mean: {data.mean():.4f}, Std: {data.std():.4f}")

    # For categorical data (0,1,2,9), show value counts
    unique_vals = np.unique(data)
    if set(unique_vals).issubset({0, 1, 2, 9}):
        values, counts = np.unique(data, return_counts=True)
        print("\nValue counts:")
        for v, c in zip(values, counts):
            print(f"  {v}: {c} ({c/data.size*100:.2f}%)")

    # Print first few samples
    # print("\nFirst 3 samples (first 10 markers):")
    # print(data[:3, :10])


def parse_args():
    """Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments

    Raises:
        FileNotFoundError: If config file doesn't exist
        argparse.ArgumentError: For invalid arguments
    """
    parser = argparse.ArgumentParser(description="SNP Data Processing Tool")

    # Required arguments
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file (required)",
    )

    # Optional arguments
    parser.add_argument(
        "--input-path", type=Path, help="Override input path from config"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    return args


def main() -> int:
    """Main entry point for data analysis and preprocessing pipeline.

    This function loads SNP data from a parquet file and applies a series of
    preprocessing steps as specified in the configuration file. The steps include:
    - Loading and transposing data
    - Sequence length filtering
    - Missing value handling
    - Data normalization
    - Data scaling
    - Test pattern application (optional)

    Returns:
        int: 0 on success, 1 on error
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(debug=args.debug)

    # Start data processing
    logger.info("Starting SNP data processing")
    logger.debug(f"Command line arguments: {args}")

    # Set PROJECT_ROOT in environment using the existing PROJECT_ROOT variable
    os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT.absolute())
    logger.info(f"Using PROJECT_ROOT: {os.environ['PROJECT_ROOT']}")

    # Load config with environment variable expansion
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override input path if given via command line
    if args.input_path:
        config["data"]["input_path"] = str(args.input_path)

    logger.info("Configuration loaded successfully")

    # Extract and log data config
    data_config = config.get("data", {})
    logger.debug("Data configuration:")
    for key, value in data_config.items():
        logger.debug(f"  {key}: {value}")

    # Load and transpose data
    input_path = Path(data_config["input_path"])
    logger.info(f"Loading data from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_parquet(input_path).to_numpy().T
    logger.info(f"Loaded data with shape: {data.shape} (samples, markers)")
    print_data_stats(data, "Initial Data")

    # 1. Apply sequence length filtering
    try:
        seq_length = data_config.get("seq_length")
        if seq_length is not None and seq_length < data.shape[1]:
            logger.info(
                f"Applying sequence length filtering with {seq_length} markers..."
            )
            data = data[:, :seq_length]
            plot_sample(data[0], PROJECT_ROOT / "1_seq_length.png")
            print_data_stats(data, "After sequence length filtering")
    except Exception as e:
        logger.error(f"Error during sequence length filtering: {e}")
        raise

    # 2. Handle missing values
    try:
        missing_value = data_config.get("missing_value", 9)
        if missing_value is not None:
            logger.info(f"Handling missing values (marked as {missing_value})...")
            original_missing = np.sum(data == missing_value)
            if original_missing > 0:
                data = handle_missing_values(data, missing_value)
                remaining_missing = np.sum(data == missing_value)
                print_data_stats(data, "After handling missing values")
                logger.info(f"Missing values: {original_missing} → {remaining_missing}")
                plot_sample(data[0], PROJECT_ROOT / "2_missing_values.png")
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise

    # 3. Normalize data
    try:
        if data_config.get("normalize", False):
            logger.info("Normalizing data...")
            data = normalize_data(data)
            print_data_stats(data, "After normalization")
            logger.info("Mapping: 0 → 0.0, 1 → 0.5, 2 → 1.0")
            plot_sample(data[0], PROJECT_ROOT / "3_normalized.png")
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        raise

    # 4. Handle staircase structure (takes precedence)
    try:
        if data_config.get("staircase", False):
            logger.info("Applying staircase structure")

            # Get pattern configuration for augmentation
            staircase_pattern = data_config.get("staircase_pattern", [])

            if not staircase_pattern:
                # Default augmentation pattern if none specified
                staircase_pattern = [[0, 25, 0.0], [25, 75, 0.5], [75, 100, 1.0]]
                logger.info("Using default augmentation pattern: 100 SNPs")

            # Convert to list of tuples
            staircase_pattern = [
                (int(start), int(end), float(value))
                for start, end, value in staircase_pattern
            ]

            logger.info(f"Staircase pattern: {staircase_pattern}")

            data = staircase_data(data, staircase_pattern)
            logger.info(f"Uniques values: {np.unique(data)}")
            plot_sample(data[0], PROJECT_ROOT / "4_staircase.png")

    except Exception as e:
        logger.error(f"Error during staircase structure: {e}")
        raise

    # 5. Handle mixing (only when staircase is disabled)
    try:
        if (not data_config.get("staircase", False)) and data_config.get(
            "mixing", False
        ):
            logger.info("Applying pattern mixing")

            # Get pattern configuration
            mixing_pattern = data_config.get("mixing_pattern", [])
            mixing_interval = data_config.get("mixing_interval", 100)

            # Convert to list of tuples
            mixing_pattern = [
                (int(start), int(end), float(value))
                for start, end, value in mixing_pattern
            ]

            logger.info(f"Pattern: {mixing_pattern}")
            logger.info(f"Mixing interval: {mixing_interval}")

            data = mix_data(data, mixing_pattern, mixing_interval)
            logger.info(f"Uniques values: {np.unique(data)}")
            plot_sample(data[0], PROJECT_ROOT / "5_mixed.png")
    except Exception as e:
        logger.error(f"Error during data mixing: {e}")
        raise

    # 5a. Optionally flip the injected pattern only for odd-indexed samples (post-mixing)
    try:
        if (
            (not data_config.get("staircase", False))
            and data_config.get("mixing", False)
            and data_config.get("flip_mixing", False)
        ):
            logger.info("Applying flipped mixing on odd-indexed samples (post-mixing)")

            # Get pattern configuration
            mixing_pattern = data_config.get("mixing_pattern", [])
            mixing_interval = data_config.get("mixing_interval", 100)

            if not mixing_pattern:
                mixing_pattern = [[0, 10, 0.0], [10, 30, 0.5], [30, 40, 1.0]]
                logger.info(
                    "Using default pattern for flipped mixing: 40 SNPs staircase"
                )

            mixing_pattern = [
                (int(start), int(end), float(value))
                for start, end, value in mixing_pattern
            ]

            logger.info(f"Pattern for flipped mixing (odds only): {mixing_pattern}")
            logger.info(f"Mixing interval: {mixing_interval}")

            data = mix_data_flip_odd(data, mixing_pattern, mixing_interval)
            logger.info(f"Uniques values after flipped mixing: {np.unique(data)}")
            plot_sample(data[1], PROJECT_ROOT / "5a_flipped_mixed.png")
    except Exception as e:
        logger.error(f"Error during flipped mixing: {e}")
        raise

    # 6. Scale data (always last)
    try:
        scale_factor = data_config.get("scale_factor")
        if scale_factor is not None:
            logger.info(f"Scaling data with factor: {scale_factor}")
            data = scale_data(data, scale_factor)
            print_data_stats(data, "After scaling")
            logger.info(f"Scaled range: [{data.min()}, {data.max()}]")

            # Plot sample
            plot_sample(data[0], PROJECT_ROOT / "6_scaled.png")
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        raise

    logger.info("Data preprocessing pipeline completed...")

    # --------------------------------------------------------------------
    # Testing Data Functions and Classes
    # --------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Testing Data Functions and Classes")
    print(f"{'='*65}")

    print("\nData Loading Function:")

    logger.info("Testing Data Loading Function...")
    dataset = load_data(config)
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"First example shape: {dataset[0].shape}")
    logger.debug(f"First example values: {dataset[0][:10]}")

    # Test PyTorch Dataset and DataModule
    try:
        print("\nPyTorch DataSet:")

        logger.info("Testing PyTorch Dataset...")
        dataset = SNPDataset(config)
        logger.info(
            f"Dataset shape [N, L]: {dataset.data.shape}, and dim: {dataset.data.dim()}"
        )

        # Plot first sample
        plot_sample(dataset[0], PROJECT_ROOT / "snp_dataset_sample.png")

        # Visualize randomly chosen even and odd samples
        plot_even_odd_samples(dataset, PROJECT_ROOT)

        # Plot first 10 samples to visually confirm augmentation (even vs odd)
        plot_first_n_samples(dataset, PROJECT_ROOT / "first_10_samples.png", n=10)

        print("\nPyTorch DataLoader:")

        logger.info("Testing PyTorch DataLoader...")
        dataloader = DataLoader(
            dataset, batch_size=config["data"]["batch_size"], shuffle=False
        )
        batch = next(iter(dataloader))  # Shape: [B, L]
        logger.info(f"Batch shape [B, L]: {batch.shape}, and dim: {batch.dim()}")
        plot_sample(batch[0], PROJECT_ROOT / "snp_dataloader_sample.png")
    except Exception as e:
        logger.error(f"Error during dataset testing: {e}")
        raise

    logger.info("Data processing completed successfully")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        print("\nRun with --debug for more details.", file=sys.stderr)
        sys.exit(1)
