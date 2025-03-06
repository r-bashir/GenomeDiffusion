#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader, random_split

from data_loading import load_data


# TODO: Create a base LightingModule with necessary hooks, this
# module will be inherited by DiffusionNetwork in model.py script.
class NetworkBase(pl.LightningModule):
    def __init__(self, input_path, hparams):
        super().__init__()

        self.path = hparams["input_path"]
        self.split = [10, 1, 1]
        self.batch = 64
        self.workers = 4
        self.data_split = [128686, 16086, 16086]  # 80%, 10% and 10%
        self.trainset, self.valset, self.testset = None, None, None

        # Save hyperparameters
        self.save_hyperparameters(hparams)

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

    # Configure Optimizer & Scheduler
    def configure_optimizers(self):
        """Configure the Optimizer and Scheduler"""
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=0.3,
                    gamma=10,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    # Trainig Step
    def training_step(self, batch, batch_idx):
        # YOUR CODE HERE:
        pass

    # Validation Step
    def validation_step(self, batch, batch_idx):
        # YOUR CODE HERE:
        pass

    # Test Step
    def test_step(self, batch, batch_idx):
        # YOUR CODE HERE:
        pass

    # Optimizer Step
    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Settings before Optimizer Step"""

        # warm up lr
        if self.hparams.get("warmup", 0) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))
