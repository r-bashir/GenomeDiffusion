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
        self.data_split = [128686, 16086, 16086] # 80%, 10% and 10%

    # Setup Data
    def setup(self, stage=None):
        """Prepare the dataset"""
        full_dataset = load_data(self.path)
        self.trainset, self.valset, self.testset = random_split(
            full_dataset,
            self.data_split,
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        )

    # Data Loaders
    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers
            )  # , pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers
            )  # , pin_memory=True, persistent_workers=True)
        
    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers
            )  # , pin_memory=True, persistent_workers=True)