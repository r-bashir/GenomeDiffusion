#!/usr/bin/env python
# coding: utf-8

"""
Test the functionality of SNP Dataset from src/dataset.py by running:

$ python scripts/run_dataset.py --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset import (
    SNPDataModule,
    SNPDataset,
    augment_data,
    handle_missing_values,
    load_data,
    normalize_data,
    scale_data,
    setup_logging,
)
from src.utils import load_config


# Print data statistics
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

    # Set PROJECT_ROOT in environment using the existing project_root variable
    os.environ["PROJECT_ROOT"] = str(project_root.absolute())
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

    # 1. Load and transpose data
    input_path = Path(data_config["input_path"])
    logger.info(f"Loading data from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = pd.read_parquet(input_path).to_numpy().T
    logger.info(f"Loaded data with shape: {data.shape} (samples, markers)")
    print_data_stats(data, "Initial Data")

    # 2. Apply sequence length filtering
    try:
        seq_length = data_config.get("seq_length")
        if seq_length is not None and seq_length < data.shape[1]:
            logger.info(
                f"Applying sequence length filtering with {seq_length} markers..."
            )
            data = data[:, :seq_length]
            print_data_stats(data, "After sequence length filtering")
    except Exception as e:
        logger.error(f"Error during sequence length filtering: {e}")
        raise

    # 3. Handle missing values
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
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise

    # 4. Normalize data
    try:
        if data_config.get("normalize", False):
            logger.info("Normalizing data...")
            data = normalize_data(data)
            print_data_stats(data, "After normalization")
            logger.info("Mapping: 0 → 0.0, 1 → 0.5, 2 → 1.0")
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        raise

    # 5. Scale data
    try:
        scale_factor = data_config.get("scale_factor")
        if scale_factor is not None:
            logger.info(f"Scaling data with factor: {scale_factor}")
            data = scale_data(data, scale_factor)
            print_data_stats(data, "After scaling")
            logger.info(f"Scaled range: [{data.min()}, {data.max()}]")
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        raise

    # 6. Handle augmentation (skipping augmentation as requested)
    try:
        if data_config.get("augment", False):
            logger.info("Applying data augmentation (fixed patterns)")
            data = augment_data(data)
            print_data_stats(data, "After augmentation")
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        raise

    logger.info("Data preprocessing pipeline completed...")

    # --------------------------------------------------------------------
    # Testing Data Functions and Classes
    # --------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"Testing Data Functions and Classes")
    print(f"{'='*65}")

    print("\nData Loading Function:")

    logger.info("Testing Data Loading Function...")
    dataset = load_data(config)
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"First example shape: {dataset[0].shape}")
    logger.debug(f"First example values: {dataset[0][:10]}")

    # Test PyTorch Dataset and DataModule
    try:
        print("\nPyTorch Dataset:")

        logger.info("Testing PyTorch Dataset...")
        snp_dataset = SNPDataset(config)
        logger.info(f"Dataset length: {len(snp_dataset)}")
        logger.info(f"First example shape: {snp_dataset[0].shape}")
        logger.debug(f"First example values: {snp_dataset[0][:10]}")

        print("\nPyTorch DataModule:")

        logger.info("Testing Lightning DataModule...")
        data_module = SNPDataModule(config)
        data_module.setup()
        test_loader = data_module.test_dataloader()
        logger.info(f"Test data batches: {len(test_loader)}")

        if len(test_loader) > 0:
            batch = next(iter(test_loader))
            logger.info(f"Batch length: {len(batch)}")
            logger.info(f"First example shape: {batch[0].shape}")
            logger.debug(f"First example values: {batch[0][:10]}")

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
