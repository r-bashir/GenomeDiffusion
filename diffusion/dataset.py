#!/usr/bin/env python
# coding: utf-8

"""This script implements the SNPDataset and SNPDataModule classes."""

import argparse
import os
import numpy as np
from scipy import stats
import pandas as pd
import pytorch_lightning as pl
import torch


# load dataset
def load_data(input_path=None):
    # read data
    try:
        data = pd.read_parquet(input_path).to_numpy()
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

    # normalize data: map (0 → 0.0, 1 → 0.5, 2 → 1.0)
    data = np.where(data == 0, 0.0, data)  # Map 0 to 0.0
    data = np.where(data == 1, 0.5, data)  # Map 1 to 0.5
    data = np.where(data == 2, 1.0, data)  # Map 2 to 1.0
    
    # replace missing values with most frequent value
    data = np.where(data == 9, 0.0, data)  # Map 9 to 0.0

    return torch.FloatTensor(data)


# SNPDataset
class SNPDataset(torch.utils.data.Dataset):
    def __init__(self, input_path=None):

        self.input_path = input_path
        self.data = load_data(self.input_path)
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
    def __init__(self, input_path, batch_size=256, num_workers=1):
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.workers = num_workers
        self.fractions = [0.8, 0.1, 0.1]

    # Setup Data
    def setup(self, stage=None, fractions=[0.8, 0.1, 0.1]):
        """Prepare the dataset"""
        if sum(fractions) != 1.0:
            raise ValueError("Fractions must sum to 1.")

        full_dataset = load_data(self.input_path)
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
        self, input_path, batch_size=256, num_workers=1, impute_strategy="mode"
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
        full_dataset = load_data(self.path)  # Assuming this loads a TensorDataset

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

def main(args):
    """Main function to test SNPDataset and SNPDataModule."""
    
    # Test Loading
    print("\nExamining data:")
    dataset = load_data(input_path=args.input_path)
       
    # Analyze unique values in the data
    unique_values = torch.unique(dataset)
    print(f"\nUnique values in data: {unique_values.tolist()}")
    
    # Count occurrences of each value
    value_counts = {}
    for value in [0.0, 0.5, 1.0, 9.0]:
        count = (dataset == value).sum().item()
        percentage = (count / dataset.numel()) * 100
        value_counts[value] = (count, percentage)
    
    print("\nValue distribution (percentage):")
    for value, (count, percentage) in value_counts.items():
        print(f"{value:.1f}: {count} occurrences ({percentage:.2f}%)")

    print(f"Dataset length: {len(dataset)}")
    print(f"First example: {dataset[0].shape}")
        
    # Test SNPDataset
    print("\nTesting SNPDataset Class:")
    snp_dataset = SNPDataset(args.input_path)
    print(f"Dataset length: {len(snp_dataset)}")
    print(f"First example: {snp_dataset[0].shape}")

    # Test SNPDataModule
    print("\nTesting SNPDataModule Class:")
    data_module = SNPDataModule(args.input_path, batch_size=256, num_workers=1)
    data_module.setup(fractions=[0.8, 0.1, 0.1])
    print(f"Train batches: {len(data_module.train_dataloader())}")
    batch = next(iter(data_module.train_dataloader()))
    print(f"Batch length: {len(batch)}")
    print(f"First example: {batch[0].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/HO_data_filtered/HumanOrigins2067_filtered.parquet",
        help="Path to the input SNP dataset file.",
    )
    args = parser.parse_args()

    if check_path_exists(args.input_path):
        print(f"The path {args.input_path} exists")
    else:
        print(f"The path {args.input_path} does not exist")

    main(args)
