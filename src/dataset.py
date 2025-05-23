#!/usr/bin/env python
# coding: utf-8

"""This script implements the SNPDataset and SNPDataModule classes."""

import argparse
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy import stats


# load dataset
def load_data_old(input_path=None, seq_length=None):
    """
    Load data and process it.

    Args:
        input_path: Path to the parquet file containing SNP data
        seq_length: Number of SNP markers to include (None for all markers)

    Returns:
        torch.FloatTensor: Processed data with shape (n_samples, seq_length)
    """
    # Read data
    try:
        # The parquet data is stored in shape (n_markers, n_samples).
        # We need to transpose it to get shape as (n_samples, n_markers)
        data = pd.read_parquet(input_path).to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    # Handle NaNs
    data = handle_missing_values(data)

    # Normalize Data
    data = normalize_data(data)

    # Augment Data (replace 50% data with 0.5)
    # data = data_augmentation(data)

    # Transpose to get (n_samples, n_markers)
    data = data.T

    # Slice to seq_length if specified
    if seq_length is not None:
        print(f"Loading data with {seq_length} markers (out of {data.shape[1]})")
        data = data[:, :seq_length]
    else:
        print(f"Loading full dataset with {data.shape[1]} markers")

    # Scale Data by a Factor
    # data = scale_data(data, factor=0.5)

    # Convert to Float Tensor
    data = torch.FloatTensor(data)  # [n_samples, seq_len]
    return data


def load_data(input_path=None, seq_length=None):
    """
    Load data and process it.

    Args:
        input_path: Path to the parquet file containing SNP data
        seq_length: Number of SNP markers to include (None for all markers)

    Returns:
        torch.FloatTensor: Processed data with shape (n_samples, seq_length)
    """
    # Read data
    try:
        # The parquet data is stored in shape (n_markers, n_samples).
        # We need to transpose it to get shape as (n_samples, n_markers)
        data = pd.read_parquet(input_path).to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    # Handle NaNs
    data = handle_missing_values(data)

    # Normalize Data
    data = normalize_data(data)

    # Augment Data
    # data = augment_data(data)

    # Transpose to get (n_samples, n_markers)
    data = data.T

    # Slice to seq_length if specified
    if seq_length is not None:
        print(f"Loading data with {seq_length} markers (out of {data.shape[1]})")
        data = data[:, :seq_length]
    else:
        print(f"Loading full dataset with {data.shape[1]} markers")

    # Create position-dependent pattern in the first 100 SNPs
    if seq_length is not None and seq_length >= 100:
        # Set first 25 SNPs to 0.0
        data[:, :25] = 0.0
        # Set next 50 SNPs to 0.5
        data[:, 25:75] = 0.5
        # Set next 25 SNPs to 1.0
        data[:, 75:100] = 1.0

    # Scale Data by a Factor
    data = scale_data(data, factor=0.5)

    # Convert to Float Tensor
    return torch.FloatTensor(data)


def handle_missing_values(data):
    """Handles missing values in the dataset per SNP marker.

    At this point in the data pipeline, the data has shape (n_markers, n_samples),
    where each row represents a single marker across all samples.
    For each marker, we find the most frequent non-missing value and replace
    any missing values (9) with that value.
    """

    # Loop over each SNP/marker (rows in the current orientation)
    for i in range(data.shape[0]):
        row = data[i]  # All samples for this marker
        valid_values = row[row != 9]  # Filter out missing values
        if len(valid_values) > 0:
            # Find most common value for this marker
            mode_value = stats.mode(valid_values, keepdims=True)[0][0]
            # Replace missing values with the mode
            row[row == 9] = mode_value

    return data


def augment_data(data):
    """Augment data by changing the values of certain markers"""

    # Replace 50% of data values with 0.5
    num_elements = data.size
    num_to_replace = num_elements // 2
    indices = np.unravel_index(
        np.random.choice(num_elements, num_to_replace, replace=False), data.shape
    )
    data[indices] = 0.5

    return data


def normalize_data(data):
    """Normalize data by mapping original SNP/marker values to new
    SNP/marker values. We map 0 → 0.0, 1 → 0.5, and 2 → 1.0"""

    # Normalize by mapping
    data[data == 0] = 0.0
    data[data == 1] = 0.5
    data[data == 2] = 1.0

    return data


def scale_data(data, factor=0.5):
    """
    Scale SNP/marker data by 0.5 to shift peaks for trimodal distribution.
    After normalization, this maps peaks from [0.0, 0.5, 1.0] to [0.0, 0.25, 0.5].
    Args:
        data (np.ndarray or torch.Tensor): Normalized data (values in [0.0, 0.5, 1.0])
    Returns:
        Scaled data (same type as input)
    """
    return data * factor


# SNPDataset
class SNPDataset(torch.utils.data.Dataset):
    def __init__(self, input_path=None, seq_length=None):

        self.input_path = input_path
        self.seq_length = seq_length
        self.data = load_data(self.input_path, seq_length=self.seq_length)
        self.validate_data()

    def validate_data(self):
        """Validates the loaded data for integrity."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("Loaded data is empty or None.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


# SNPDataModule
class SNPDataModule(pl.LightningDataModule):
    def __init__(self, input_path, batch_size=256, num_workers=1, seq_length=None):
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.workers = num_workers
        self.fractions = [0.8, 0.1, 0.1]
        self.seq_length = seq_length

    # Setup Data
    def setup(self, stage=None, fractions=[0.8, 0.1, 0.1]):
        """Prepare the dataset"""
        if sum(fractions) != 1.0:
            raise ValueError("Fractions must sum to 1.")

        full_dataset = load_data(self.input_path, seq_length=self.seq_length)
        n = len(full_dataset)

        # Calculate dataset sizes
        n_train = int(fractions[0] * n)
        n_val = int(fractions[1] * n)
        n_test = n - n_train - n_val  # Ensure all data is used

        # Split dataset
        self.trainset, self.valset, self.testset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    # Data Loaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )


class SNPDataModule_v2(pl.LightningDataModule):
    def __init__(
        self,
        input_path,
        batch_size=256,
        num_workers=1,
        impute_strategy="mode",
        seq_length=None,
    ):
        """
        Args:
            input_path (str): Path to SNP dataset.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for DataLoader.
            impute_strategy (str): Strategy for handling missing values (9). Options: ["mode", "mean", None].
        """
        super().__init__()
        self.path = input_path
        self.batch_size = batch_size
        self.workers = num_workers
        self.impute_strategy = impute_strategy
        self.seq_length = seq_length
        self.data_split = [128686, 16086, 16086]  # 80% train, 10% val, 10% test

    def preprocess(self, dataset):
        """Handles missing values (9s) by either mode or mean imputation."""
        data = dataset.tensors[
            0
        ]  # Extract SNP data (assuming dataset is TensorDataset)

        mask = (data != 9).float()  # Mask: 1 for valid, 0 for missing

        if self.impute_strategy == "mode":
            mode_values = self.compute_mode(data)
            data = torch.where(mask.bool(), data, mode_values)
        elif self.impute_strategy == "mean":
            mean_values = torch.nanmean(
                torch.where(mask.bool(), data, torch.nan), dim=0
            )
            data = torch.where(mask.bool(), data, mean_values)

        return torch.utils.data.TensorDataset(
            data, dataset.tensors[1]
        )  # Reconstruct dataset

    def compute_mode(self, data):
        """Computes mode per SNP column, ignoring missing values (9s)."""
        valid_data = data[data != 9]
        mode_values = torch.mode(valid_data, dim=0)[0]  # Get most common value per SNP
        return mode_values

    def setup(self, stage=None):
        """Loads and preprocesses dataset."""
        full_dataset = load_data(
            self.path, seq_length=self.seq_length
        )  # Load with specified sequence length

        # Split into train/val/test
        self.trainset, self.valset, self.testset = torch.utils.data.random_split(
            full_dataset,
            self.data_split,
            generator=torch.Generator().manual_seed(
                42
            ),  # Fixed seed for reproducibility
        )

        # Apply preprocessing
        self.trainset = self.preprocess(self.trainset)
        self.valset = self.preprocess(self.valset)
        self.testset = self.preprocess(self.testset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )


def check_path_exists(path):
    """Check if a path exists."""
    return os.path.exists(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    # Initial marker frequencies (whole dataset)
    print("\nInitial marker frequencies (whole dataset)...")
    raw_data = pd.read_parquet(args.input_path).to_numpy().T
    unique, counts = np.unique(raw_data, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    for value in sorted(freq_dict.keys()):
        percentage = (freq_dict[value] / raw_data.size) * 100
        print(f"Value {value}: {percentage:.2f}%")

    # Handle missing values
    print("\nHandling missing values...")
    raw_data = handle_missing_values(raw_data)

    print("\nFinal marker frequencies (whole dataset)...")
    unique, counts = np.unique(raw_data, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    for value in sorted(freq_dict.keys()):
        percentage = (freq_dict[value] / raw_data.size) * 100
        print(f"Value {value}: {percentage:.2f}%")

    # Handling normalization
    print("\nHandling normalization...")
    raw_data[raw_data == 0] = 0.0
    raw_data[raw_data == 1] = 0.5
    raw_data[raw_data == 2] = 1.0

    print("\nNormalized marker frequencies (whole dataset)...")
    unique, counts = np.unique(raw_data, return_counts=True)
    freq_dict = dict(zip(unique, counts))
    for value in sorted(freq_dict.keys()):
        percentage = (freq_dict[value] / raw_data.size) * 100
        print(f"Value {value}: {percentage:.2f}%")

    # Test Python Dataset
    print("\nTesting Python Dataset:")
    dataset = load_data(
        input_path=args.input_path, seq_length=1000
    )  # Test with 1000 markers
    print(f"Dataset length: {len(dataset)}")
    print(f"First example: {dataset[0].shape}")

    # Test PyTorch Dataset
    print("\nTesting PyTorch Dataset:")
    snp_dataset = SNPDataset(args.input_path, seq_length=1000)  # Test with 1000 markers
    print(f"Dataset length: {len(snp_dataset)}")
    print(f"First example: {snp_dataset[0].shape}")

    # Test Lightning DataModule
    print("\nTesting Lightning DataModule:")
    data_module = SNPDataModule(
        args.input_path, batch_size=256, num_workers=1, seq_length=1000
    )  # Test with 1000 markers
    data_module.setup(fractions=[0.8, 0.1, 0.1])
    print(f"Train batches: {len(data_module.train_dataloader())}")
    batch = next(iter(data_module.train_dataloader()))
    print(f"Batch length: {len(batch)}")
    print(f"First example: {batch[0].shape}")


if __name__ == "__main__":
    main()
