import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import pytorch_lightning as pl


# load dataset
def load_data(input_path=""):

    # read data
    data = pd.read_parquet(input_path).to_numpy()

    # normalize data: map (0 → 0.0, 1 → 0.5, 2 → 1.0)
    data = np.where(data == 0, 0.0, data)  # Map 0 to 0.0
    data = np.where(data == 1, 0.5, data)  # Map 1 to 0.5
    data = np.where(data == 2, 1.0, data)  # Map 2 to 1.0

    return torch.FloatTensor(data)


# SNP Dataset
class SNPDataset(Dataset):
    def __init__(self, input_path):
        self.data = load_data(input_path)

    def __len__(self):
        return self.data.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.data[idx]  # Return one sample (row)


# SNP Data Module
class SNPDataModule(pl.LightningDataModule):
    def __init__(self, input_path, batch_size=256, num_workers=1):
        super().__init__()
        self.path = input_path
        self.batch_size = batch_size
        self.workers = num_workers
        self.data_split = [128686, 16086, 16086]  # 80%, 10% and 10%

    # Setup Data
    def setup(self, stage=None):
        """Prepare the dataset"""
        full_dataset = load_data(self.path)
        self.trainset, self.valset, self.testset = random_split(
            full_dataset,
            self.data_split,
            generator=torch.Generator().manual_seed(
                42
            ),  # Fixed seed for reproducibility
        )

    # Data Loaders
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )  # , pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )  # , pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )  # , pin_memory=True, persistent_workers=True)


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
        self.trainset, self.valset, self.testset = random_split(
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
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
