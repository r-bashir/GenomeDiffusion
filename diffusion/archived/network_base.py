#!/usr/bin/env python
# coding: utf-8

"""Base network module for PyTorch Lightning integration."""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class NetworkBase(pl.LightningModule):
    """Base class for all network modules with PyTorch Lightning integration.

    This class provides common functionality for training, validation, testing,
    and sample generation. It is designed to be inherited by specific model
    implementations like DiffusionModel.

    Inheriting classes must implement:
    - forward_step: Performs a forward pass through the model
    - compute_loss: Computes the loss for a batch of data
    - generate_samples: Generates samples from the model
    """

    def __init__(self, config: Dict):
        """Initialize the base network.

        Args:
            config: Dictionary containing model and training configuration.
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize dataset
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing.

        This method is called by PyTorch Lightning before training/validation/testing.
        It handles dataset initialization and splitting.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if self.dataset is None:
            # Load dataset - override this in subclasses if needed
            self.dataset = self._create_dataset()

            # Split dataset
            train_size = int(len(self.dataset) * self.config["data"]["split"][0])
            val_size = int(len(self.dataset) * self.config["data"]["split"][1])
            test_size = len(self.dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42),
            )

            print(
                f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}"
            )

    def _create_dataset(self) -> Dataset:
        """Create and return the dataset.

        Override this method in subclasses to create the specific dataset.

        Returns:
            Dataset: The created dataset.
        """
        raise NotImplementedError("Subclasses must implement _create_dataset")

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Prepare a batch for model input.

        Override this method in subclasses if specific batch preparation is needed.

        Args:
            batch: Input batch from dataloader.

        Returns:
            torch.Tensor: Prepared batch.
        """
        return batch

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Override this method in subclasses to implement the specific forward pass.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch.

        Override this method in subclasses to implement the specific loss computation.

        Args:
            batch: Input batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch from dataloader.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Prepare input
        x = self._prepare_batch(batch)

        # Compute loss
        loss = self.compute_loss(x)

        # Log training loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Input batch from dataloader.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Prepare input
        x = self._prepare_batch(batch)

        # Compute loss
        loss = self.compute_loss(x)

        # Log validation loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Input batch from dataloader.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Prepare input
        x = self._prepare_batch(batch)

        # Compute loss
        loss = self.compute_loss(x)

        # Log test loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List[Optimizer], List[Dict]]: Optimizer and scheduler configuration.
        """
        opt_config = self.config["optimizer"]

        # Create optimizer
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.01),
                betas=tuple(opt_config.get("betas", (0.9, 0.999))),
                eps=opt_config.get("eps", 1e-8),
                amsgrad=opt_config.get("amsgrad", False),
            )
        ]

        # Create scheduler
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer[0],
                    T_max=self.config["training"]["num_epochs"],
                    eta_min=opt_config.get("min_lr", 1e-6),
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]

        return optimizer, scheduler

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Apply learning rate warmup and minimum learning rate.

        Args:
            optimizer: The optimizer being used.
        """
        warmup_epochs = self.config["training"].get("warmup_epochs", 0)

        # Warmup period
        if warmup_epochs and (self.trainer.current_epoch < warmup_epochs):
            lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.config["optimizer"]["lr"]

        # Enforce minimum learning rate
        min_lr = self.config["optimizer"].get("min_lr", 0.0)
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], min_lr)

    def forward_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.
        To be implemented by child classes.

        Args:
            batch: Input batch from dataloader

        Returns:
            torch.Tensor: Model outputs
        """
        raise NotImplementedError("Subclasses must implement forward_step")

    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the model.

        Override this method in subclasses to implement specific sample generation.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        raise NotImplementedError("Subclasses must implement generate_samples")
