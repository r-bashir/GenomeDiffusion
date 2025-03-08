#!/usr/bin/env python
# coding: utf-8

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import load_data


class NetworkBase(pl.LightningModule):
    """Base network class with common training functionality.

    This class provides a foundation for PyTorch Lightning models with:
    - Data loading and splitting
    - Optimizer configuration with AdamW and learning rate scheduling
    - Learning rate warmup and minimum learning rate
    - Loss computation and training hooks

    Inheriting classes must implement:
    - forward_step: Performs a forward pass through the model
    - sample: Generates samples from the model
    """

    def __init__(self, hparams: dict):
        """Initialize the base network.

        Args:
            hparams: Dictionary containing model hyperparameters including:
                    - input_path: Path to input data
                    - data: Data configuration (batch_size, num_workers, etc.)
                    - optimizer: Optimizer settings
                    - training: Training configuration
        """
        super().__init__()

        # Data settings
        self.path = hparams["input_path"]
        self.batch_size = hparams["data"].get("batch_size", 64)
        self.num_workers = hparams["data"].get("num_workers", 4)
        self.fractions = hparams["data"].get("fractions", [0.8, 0.1, 0.1])

        # Training state
        self.trainset = None
        self.valset = None
        self.testset = None

        # Save hyperparameters
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        """Prepare the dataset"""
        if sum(self.fractions) != 1.0:
            raise ValueError("Fractions must sum to 1.")

        full_dataset = load_data(self.path)
        n = len(full_dataset)

        # Calculate dataset sizes
        n_train = int(self.fractions[0] * n)
        n_val = int(self.fractions[1] * n)
        n_test = n - n_train - n_val  # Ensure all data is used

        self.trainset, self.valset, self.testset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["optimizer"]["lr"],
                weight_decay=self.hparams["optimizer"].get("weight_decay", 0.01),
                betas=self.hparams["optimizer"].get("betas", (0.9, 0.999)),
                eps=self.hparams["optimizer"].get("eps", 1e-8),
                amsgrad=self.hparams["optimizer"].get("amsgrad", True),
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer[0],
                    T_max=self.hparams["training"]["num_epochs"],
                    eta_min=self.hparams["optimizer"].get("min_lr", 1e-6),
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.
        To be implemented by child classes.

        Args:
            batch: Input batch from dataloader

        Returns:
            torch.Tensor: Model outputs
        """
        raise NotImplementedError

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch of data.
        To be implemented by child classes.

        Args:
            batch: Input batch from dataloader

        Returns:
            torch.Tensor: Loss value
        """
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute and return the training loss.

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of the batch

        Returns:
            torch.Tensor: The loss value
        """
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute and return the validation loss.

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of the batch

        Returns:
            torch.Tensor: The loss value
        """
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute and return the test loss.

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of the batch

        Returns:
            torch.Tensor: The loss value
        """
        loss = self.compute_loss(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Apply learning rate warmup and minimum learning rate."""
        warmup_epochs = self.hparams["training"].get("warmup_epochs", 0)

        # Warmup period
        if warmup_epochs and (self.trainer.current_epoch < warmup_epochs):
            lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["optimizer"]["lr"]

        # Enforce minimum learning rate
        min_lr = self.hparams["optimizer"].get("min_lr", 0.0)
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], min_lr)
