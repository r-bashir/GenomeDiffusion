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
    - forward: Performs a forward pass through the model
    - compute_loss: Computes the loss for a batch of data
    - generate_samples: Generates samples from the model
    """

    def __init__(self, hparams: Dict):
        """Initialize the network base.

        Args:
            hparams: Dictionary containing model hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._datasplit = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing.

        This method is called by PyTorch Lightning before training/validation/testing.
        It handles dataset initialization and splitting.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if not hasattr(self, "dataset") or self.dataset is None:
            # Load dataset - override this in subclasses if needed
            self.dataset = self._create_dataset()
            self._datasplit = self.hparams["data"]["datasplit"]
            self._train_dataset, self._val_dataset, self._test_dataset = random_split(
                self.dataset,
                self._datasplit,
                generator=torch.Generator().manual_seed(42),
            )

    def _create_dataset(self) -> Dataset:
        """Create and return the dataset.

        Returns:
            Dataset: The created dataset.
        """
        if "input_path" not in self.hparams:
            raise ValueError("input_path must be specified in hparams")
        from diffusion import SNPDataset  # Import here to avoid circular imports

        return SNPDataset(self.hparams["input_path"])

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Prepare a batch for model input.

        Ensures the batch has the correct shape [B, C, seq_len].
        For non-sequence data, override this method.

        Args:
            batch: Input batch from dataloader.

        Returns:
            torch.Tensor: Prepared batch with shape [B, C, seq_len].
        """
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(1)  # Convert to (batch_size, 1, seq_len)
        return batch

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=True,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = self._prepare_batch(batch)
        loss = self.compute_loss(x)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def _shared_evaluation(
        self, batch: torch.Tensor, stage: str, log: bool = True
    ) -> Dict:
        """Shared evaluation logic for validation and test.

        Args:
            batch: Input batch from dataloader
            stage: Stage name ('val' or 'test')
            log: Whether to log metrics

        Returns:
            Dict containing loss and model outputs
        """
        batch = self._prepare_batch(batch)
        loss = self.compute_loss(batch)

        if log:
            self.log(
                f"{stage}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {
            "loss": loss,
            "model_output": batch,  # Original input for comparison
            "predicted": self.forward(batch),  # Model's prediction
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self._shared_evaluation(batch, "val")
        return outputs["loss"]  # Only return loss for validation

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict:
        """Test step for evaluation.

        Args:
            batch: Input batch from test dataloader
            batch_idx: Index of the current batch

        Returns:
            Dict containing test loss and model outputs
        """
        return self._shared_evaluation(batch, "test")  # Return full dict for test

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams["optimizer"]["lr"]),
            weight_decay=float(self.hparams["optimizer"]["weight_decay"]),
            betas=tuple(float(x) for x in self.hparams["optimizer"]["betas"]),
            eps=float(self.hparams["optimizer"]["eps"]),
            amsgrad=bool(self.hparams["optimizer"]["amsgrad"]),
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["scheduler"]["T_max"],
                eta_min=float(self.hparams["scheduler"]["eta_min"]),
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Apply learning rate warmup and minimum learning rate.

        Args:
            optimizer: The optimizer being used.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch.

        Args:
            batch: Input batch.

        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")

    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        raise NotImplementedError("Subclasses must implement generate_samples")
