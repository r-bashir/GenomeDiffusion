#!/usr/bin/env python
# coding: utf-8

"""
SNP Data Loading and Preprocessing Module.

This module provides functionality for loading and preprocessing SNP (Single Nucleotide Polymorphism) data.
It includes classes and functions for data loading, preprocessing, and dataset management for machine learning.

Classes:
    SNPDataset: PyTorch Dataset for SNP data
    SNPDataModule: PyTorch Lightning DataModule for SNP data

Functions:
    load_data: Load and preprocess SNP data from a parquet file
    handle_missing_values: Impute missing values using mode imputation
    normalize_data: Normalize SNP values to [0.0, 0.5, 1.0] range
    scale_data: Scale normalized data by a given factor
    augment_data: Apply test patterns to the data (for debugging)
    print_data_stats: Print statistics about the data
    main: Command-line entry point for data preprocessing

Example:
    # Run as a module
    python -m src.dataset --config config.yaml

    # Run as a script
    python src/dataset.py --config config.yaml
"""

import argparse
import logging
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
import yaml
from scipy import stats


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure console logging and return a logger instance.

    Args:
        debug: If True, set log level to DEBUG, else INFO

    Returns:
        logging.Logger: Configured logger instance with name "DataLogger"
    """
    logger = logging.getLogger("DataLogger")

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


# Initialize the module-level logger
logger = setup_logging()


# Load dataset
def load_data(config: Dict[str, Any]) -> torch.Tensor:
    """Load and preprocess SNP data based on configuration. The data is loaded
    from a parquet file with shape (n_markers, n_samples) and transposed to shape
    (n_samples, n_markers). The data is then processed based on the configuration.

    Preprocessing steps (ordered):
    1. Handle sequence length
    2. Handle missing values
    3. Apply normalization
    4. Apply augmentation (Fixed patterns)
    5. Apply scaling (always last)

    Args:
        config: Configuration dictionary with data and preprocessing parameters

    Returns:
        torch.FloatTensor: Processed data with shape (n_samples, n_markers)
    """
    logger.info("Starting data loading and preprocessing")
    data_config = config.get("data")

    # Load and transpose data
    try:
        input_path = data_config.get("input_path")
        if not input_path:
            raise ValueError("input_path not specified in config")

        logger.info(f"Loading data from {input_path}")
        data = pd.read_parquet(input_path).to_numpy().T
        logger.info(f"Loaded data with shape: {data.shape} (n_samples, n_markers)")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        raise

    # 1. Handle sequence length
    seq_length = data_config.get("seq_length", data.shape[1])
    if seq_length is not None and seq_length < data.shape[1]:
        logger.info(f"Using first {seq_length} markers (out of {data.shape[1]})")
        data = data[:, :seq_length]
    else:
        logger.info(f"Using all {data.shape[1]} markers")

    # 2. Handle missing values
    missing_value = data_config.get("missing_value", 9)
    if missing_value is not None:
        logger.info(f"Handling missing values (marked as {missing_value})")
        data = handle_missing_values(data, missing_value)

    # 3. Handle normalization
    if data_config.get("normalize", False):
        logger.info("Normalizing data to [0.0, 0.5, 1.0] range")
        logger.info("Mapping: 0 → 0.0, 1 → 0.5, 2 → 1.0")
        data = normalize_data(data)
        logger.info(f"Uniques values: {np.unique(data)}")

    # 4. Handle augmentation
    if data_config.get("augment", False):
        logger.info("Applying data augmentation")
        data = augment_data(data)
        logger.info(f"Uniques values: {np.unique(data)}")

    # 5. Handle scaling (always last)
    if data_config.get("scaling", False):
        scale_factor = data_config.get("scale_factor")
        logger.info(f"Scaling data by factor {scale_factor}")
        data = scale_data(data, scale_factor)
        logger.info(f"Uniques values: {np.unique(data)}")

    logger.info("Data preprocessing completed")

    return torch.FloatTensor(data)


# Handle missing values
def handle_missing_values(data: np.ndarray, missing_value: int = 9) -> np.ndarray:
    """Handle missing values (9s) by imputing with the most frequent valid value (0,1,2)
    for each marker. For each marker (column), computes the mode of valid values (0,1,2)
    and uses it to replace any missing values (9s) in that marker. In case of ties
    (multiple modes), the smallest value is used.

    Args:
        data: Input data array of shape (n_samples, n_markers)
        missing_value: Value representing missing data (default: 9)

    Returns:
        Data with missing values imputed
    """
    logger = logging.getLogger(__name__)
    data = data.copy()  # Don't modify input array
    n_missing = 0
    n_markers = data.shape[1]

    for col in range(n_markers):
        marker_values = data[:, col]
        valid_mask = marker_values != missing_value
        n_missing_col = np.sum(~valid_mask)
        n_missing += n_missing_col

        if n_missing_col == 0:
            continue  # No missing values in this column

        if np.any(valid_mask):
            valid_values = marker_values[valid_mask]
            # Compute mode of valid values
            values, counts = np.unique(valid_values, return_counts=True)
            if len(values) > 0:  # If we have valid values
                # Get all values with maximum count (handles multiple modes)
                max_count = counts.max()
                modes = values[counts == max_count]
                # Use the smallest mode value in case of ties
                mode_value = modes.min() if len(modes) > 0 else 0

                # Replace missing values with mode
                missing_mask = ~valid_mask
                data[missing_mask, col] = mode_value
                logger.debug(
                    f"Marker {col}: imputed {n_missing_col} missing values with {mode_value}"
                )
        else:
            logger.warning(
                f"Marker {col}: No valid values found, using 0 for all values"
            )
            data[:, col] = 0  # If all values are missing, set entire column to 0

    if n_missing > 0:
        logger.info(f"Imputed {n_missing} missing values in total")

    return data


# Normalize data
def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data by mapping original SNP/marker values to
    new SNP/marker values. We map 0 → 0.0, 1 → 0.5, and 2 → 1.0"""

    # Create output array with proper type
    result = np.empty_like(data, dtype=np.float32)

    # Vectorized normalization
    result[data == 0] = 0.0
    result[data == 1] = 0.5
    result[data == 2] = 1.0

    return result


# Scale data
def scale_data(
    data: Union[np.ndarray, torch.Tensor], factor: float = 0.5
) -> Union[np.ndarray, torch.Tensor]:
    """Scale data by a given factor to shift peaks for trimodal distribution.
    After normalization, this maps peaks from [0.0, 0.5, 1.0] to [0.0, 0.25, 0.5].
    Args:
        data (np.ndarray or torch.Tensor): Normalized data
        factor (float): Scaling factor
    Returns:
        Scaled data (same type as input)
    """
    data = data * factor
    return data


# Augment data
def augment_data(data: np.ndarray) -> np.ndarray:
    """Apply test patterns to the data for debugging.

    Applies the following fixed patterns to the data:
    - First 25 markers: 0.0
    - Next 50 markers: 0.5
    - Next 25 markers: 1.0

    Args:
        data: Input data array of shape (n_samples, n_markers)

    Returns:
        Copy of input data with test patterns applied
    """
    data = data.copy()
    n_markers = data.shape[1]

    # Define pattern ranges [start, end, value]
    patterns = [
        (0, 25, 0.0),  # First 25 markers set to 0.0
        (25, 75, 0.5),  # Next 50 markers set to 0.5
        (75, 100, 1.0),  # Next 25 markers set to 1.0
    ]

    # Apply each pattern
    for start, end, value in patterns:
        if start < n_markers:  # Only apply if start is within bounds
            end = min(end, n_markers)  # Don't exceed array bounds
            data[:, start:end] = value

    return data


class SNPDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for loading and accessing SNP data.

    Handles loading and preprocessing of SNP data according to the provided config.
    The actual data loading and preprocessing is handled by the `load_data` function.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Full configuration dictionary. The 'data' key should contain
                   data loading and preprocessing parameters.
        """
        self.config = config
        self.data = load_data(config)
        self.validate_data()
        logger.info(f"SNPDataset samples: {len(self)}")
        logger.info(f"SNPDataset shape: {self.data.shape}")

    def validate_data(self) -> None:
        """Validate that data was loaded correctly."""
        if self.data is None:
            raise ValueError("Data loading failed: returned None")
        if len(self.data) == 0:
            raise ValueError("Loaded data is empty")

    def __len__(self) -> int:
        """Return number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            torch.Tensor: The sample at the given index with shape (n_markers,)

        Raises:
            IndexError: If the index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self) - 1}]")
        return self.data[idx]


class SNPDataModule(pl.LightningDataModule):
    """LightningDataModule for handling SNP data loading and splitting.

    Handles train/val/test splits and creates appropriate DataLoaders.
    The data is only loaded and split when setup() is called.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Full configuration dictionary
        """
        super().__init__()
        self.config = config
        self.batch_size = self.config["data"].get("batch_size", 64)
        self.num_workers = self.config["data"].get("num_workers", 4)
        self.datasplit = self.config["data"].get("datasplit", [1700, 200, 167])

        # Track if we've loaded the data
        self._data_loaded = False
        self.full_dataset = None

        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _load_and_split_data(self) -> None:
        """Load the full dataset and perform the train/val/test split."""
        if self._data_loaded:
            return

        logger.info("Loading and splitting dataset...")

        # Load full dataset
        self.full_dataset = SNPDataset(self.config)

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = (
            torch.utils.data.random_split(
                self.full_dataset,
                self.datasplit,
                generator=torch.Generator().manual_seed(42),
            )
        )

        self._data_loaded = True

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the data for the current stage.

        This method is called on every GPU in distributed training. It handles:
        - Loading data only when needed based on the stage
        - Setting up the appropriate datasets

        Args:
            stage: Either 'fit' (train+val), 'validate' (val), 'test' (test), or None (all)
        """
        # For our case, we'll load everything in one go since we're using the same
        # dataset. But we'll respect the stage parameter for future flexibility.
        if stage == "fit" and self.train_dataset is None:
            self._load_and_split_data()
        elif stage == "validate" and self.val_dataset is None:
            self._load_and_split_data()
        elif stage == "test" and self.test_dataset is None:
            self._load_and_split_data()
        elif stage is None and not self._data_loaded:
            self._load_and_split_data()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the training dataloader."""
        if self.train_dataset is None:
            self.setup("fit")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,  # Helps with batch norm
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the validation dataloader."""
        if self.val_dataset is None:
            self.setup("validate")

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create and return the test dataloader."""
        if self.test_dataset is None:
            self.setup("test")

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
