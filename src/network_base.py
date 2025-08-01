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
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(hparams)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._datasplit = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing.

        This method is called by PyTorch Lightning before training/validation/testing.
        It handles dataset initialization and splitting.

        Data shape:
            - After dataset creation: [N, L] (N = total samples, L = sequence length)
            - After splitting: train/val/test datasets, each with [n, L]

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if not hasattr(self, "dataset") or self.dataset is None:
            # Load dataset
            if "data" not in self.hparams or "input_path" not in self.hparams["data"]:
                raise ValueError("input_path must be specified in hparams['data']")
            seq_length = self.hparams["data"].get("seq_length", None)
            print(f"Creating dataset with sequence length: {seq_length}")
            print(f"Loading data from: {self.hparams['data']['input_path']}")

            # Create and split dataset
            from src import SNPDataset

            self.dataset = SNPDataset(self.hparams)
            self._datasplit = self.hparams["data"]["datasplit"]
            self._train_dataset, self._val_dataset, self._test_dataset = random_split(
                self.dataset,
                self._datasplit,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader with shape of [B, L]."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=True,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader with shape of [B, L]."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader with shape of [B, L]."""
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams["data"]["batch_size"],
            shuffle=False,
            num_workers=self.hparams["data"]["num_workers"],
            pin_memory=True,
        )

    def _prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Prepare a batch for model input.

        Converts input batch from shape [B, L] (from DataLoader) to [B, 1, L] for 1D SNP data.
        This ensures the model always receives [B, C, L] with C=1.

        Args:
            batch: Input batch from dataloader, shape [B, L].

        Returns:
            torch.Tensor: Prepared batch with shape [B, 1, L].
        """
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(1)  # Convert to (B, 1, L)
        return batch

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for model optimization.

        Args:
            batch: Input batch from dataloader, shape [B, L]. Will be converted to [B, 1, L].
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Computed loss for backpropagation
        """
        batch = self._prepare_batch(batch)
        loss = self.compute_loss(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step for model evaluation.

        Args:
            batch: Input batch from validation dataloader, shape [B, L]. Will be converted to [B, 1, L].
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Validation loss
        """
        return self._shared_evaluation(batch, "val")["loss"]

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Test step for model evaluation.

        Args:
            batch: Input batch from test dataloader, shape [B, L]. Will be converted to [B, 1, L].
            batch_idx: Index of the current batch

        Returns:
            dict: Dictionary containing test loss, input targets, and reconstructions
                - 'loss': scalar loss
                - 'target': ground truth batch, shape [B, 1, L]
                - 'reconstruction': model output, shape [B, 1, L]
        """
        return self._shared_evaluation(batch, "test")

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Prediction step for model inference.

        Args:
            batch: Input batch from prediction dataloader, shape [B, L]. Will be converted to [B, 1, L].
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader (for multiple dataloaders)

        Returns:
            dict: Dictionary containing model reconstructions, shape [B, 1, L]
        """
        return self._shared_evaluation(batch, "predict")

    def _shared_evaluation(self, batch: torch.Tensor, stage: str) -> dict:
        """Shared evaluation logic for validation, test, and prediction stages.

        Args:
            batch: Input batch from dataloader, shape [B, L] (converted to [B, 1, L]).
            stage: Evaluation stage ('val', 'test', or 'predict')

        Returns:
            dict: Dictionary with stage-appropriate outputs:
                - val: {'loss': scalar loss}
                - test: {'loss': scalar loss, 'target': [B, 1, L], 'reconstruction': [B, 1, L]}
                - predict: {'reconstruction': [B, 1, L]}
        """
        batch = self._prepare_batch(batch)
        loss = self.compute_loss(batch)
        if stage == "val":
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            return {"loss": loss}
        elif stage == "test":
            return {
                "loss": loss,
                "target": batch,
                "reconstruction": self.denoise_batch(batch),
            }
        elif stage == "predict":
            return {"reconstruction": self.denoise_batch(batch)}
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler (CosineAnnealingLR or ReduceLROnPlateau).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams["optimizer"]["lr"]),
            weight_decay=float(self.hparams["optimizer"]["weight_decay"]),
            betas=tuple(float(x) for x in self.hparams["optimizer"]["betas"]),
            eps=float(self.hparams["optimizer"]["eps"]),
            amsgrad=bool(self.hparams["optimizer"]["amsgrad"]),
        )

        # Get scheduler type from config (default to 'cosine')
        scheduler_type = self.hparams["scheduler"]["type"]
        scheduler = None
        scheduler_dict = None

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.hparams["training"]["epochs"]),
                eta_min=float(self.hparams["scheduler"]["eta_min"]),
            )

        elif scheduler_type == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(self.hparams["scheduler"]["mode"]),
                factor=float(self.hparams["scheduler"]["factor"]),
                patience=int(self.hparams["scheduler"]["patience"]),
                threshold=float(self.hparams["scheduler"]["threshold"]),
                min_lr=float(self.hparams["scheduler"]["min_lr"]),
            )

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        scheduler_dict = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Apply learning rate warmup and minimum learning rate (if enabled in config).

        Args:
            optimizer: The optimizer being used.
        """
        warmup_epochs = int(self.hparams["training"].get("warmup_epochs", 10))

        # Warmup period
        if warmup_epochs and (self.trainer.current_epoch < warmup_epochs):
            lr_scale = min(1.0, float(self.trainer.current_epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * float(self.hparams["optimizer"]["lr"])

        # Enforce minimum learning rate
        min_lr = float(self.hparams["optimizer"].get("min_lr", 0.0))
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], min_lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement forward")

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch."""
        raise NotImplementedError("Subclasses must implement compute_loss")

    def generate_samples(self, num_samples: int = 10) -> torch.Tensor:
        """Generate samples from the model."""
        raise NotImplementedError("Subclasses must implement generate_samples")
