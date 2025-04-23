#!/usr/bin/env python
# coding: utf-8

"""Main script for training the SNP diffusion model.

Examples:
    # Train a new model using configuration from config.yaml
    python train.py --config config.yaml

    # Resume training from a checkpoint
    python train.py --config config.yaml --resume path/to/checkpoint.ckpt
"""

import argparse
import os
import pathlib
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger

from diffusion import DiffusionModel
from diffusion.evaluation_callback import EvaluationCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for SNP data generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_version_from_checkpoint(checkpoint_path: Optional[str]) -> Optional[int]:
    """Extract version number from checkpoint path if resuming.

    Args:
        checkpoint_path: Path to checkpoint for resuming training

    Returns:
        Version number if found, None otherwise
    """
    if not checkpoint_path:
        return None

    # Try to extract version from checkpoint path (e.g., output/lightning_logs/version_0/checkpoints/...)
    path = pathlib.Path(checkpoint_path)
    for parent in path.parents:
        if parent.name.startswith("version_"):
            version = int(parent.name.split("_")[1])
            print(f"Resuming logging to version directory: {parent}")
            return version
    return None


def setup_logger(
    config: Dict, resume_from_checkpoint: Optional[str]
) -> Union[TensorBoardLogger, WandbLogger, CSVLogger]:
    """Setup logger based on configuration.

    Args:
        config: Configuration dictionary
        resume_from_checkpoint: Path to checkpoint for resuming training

    Returns:
        Logger instance (TensorBoardLogger, WandbLogger, or CSVLogger)
    """
    # Get base directory for logs
    base_dir = config.get("output_path", "output")
    logs_dir = os.path.join(base_dir, "lightning_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Get version from checkpoint if resuming
    version = get_version_from_checkpoint(resume_from_checkpoint)

    # Common logger parameters
    logger_params = {"name": "", "save_dir": logs_dir, "version": version}

    # Select logger type from config
    logger_type = config["training"]["logger"]

    if logger_type == "wandb":
        try:
            # Try loading API key from environment variable first
            import wandb

            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
            elif not wandb.api.api_key:
                print("Wandb API key not found. Attempting to log in...")
                wandb.login()

            # Create WandbLogger with consistent parameters
            wandb_logger = WandbLogger(
                **logger_params,
                project=config.get("project_name", "GenomeDiffusion"),
                config=config,
                resume="allow" if resume_from_checkpoint else None,
            )
            return wandb_logger

        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {str(e)}")
            print("Falling back to TensorBoard logger")
            logger_type = "tb"  # Fall back to TensorBoard

    elif logger_type == "tb":
        return TensorBoardLogger(**logger_params)

    elif logger_type == "csv":
        return CSVLogger(**logger_params)

    else:
        raise ValueError(
            f"Logger '{logger_type}' not recognized. Use 'wandb', 'tb', or 'csv'."
        )


def setup_callbacks(config: Dict) -> List:
    """Setup training callbacks.

    Returns a list of callbacks for model training:
    - ModelCheckpoint: Saves best and last model checkpoints
    - LearningRateMonitor: Tracks learning rate changes
    - EarlyStopping: Optional, stops training if no improvement
    """
    callbacks = [
        # Save best and last checkpoints
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=config["training"].get("save_top_k", 3),
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,  # Keep filenames clean
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
    ]

    # Add early stopping if enabled in config
    if config["training"].get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["training"].get("patience", 10),
                mode="min",
            )
        )

    return callbacks


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Ensure output directory exists
    os.makedirs(config["output_path"], exist_ok=True)

    # Initialize model
    try:
        print("Initializing model...")
        model = DiffusionModel(config)
        print(f"Model initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

    # Set up loggers
    logger = setup_logger(config, args.resume)

    # Set up callbacks
    callbacks = setup_callbacks(config)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        default_root_dir=config["output_path"],
        log_every_n_steps=config["training"].get("log_every_n_steps", 50),
        enable_checkpointing=True,  # overridden by checkpoint callback
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train model
    try:
        print("Starting training...")
        trainer.fit(
            model,
            ckpt_path=args.resume,
        )
        print("Training completed successfully")

        print("\nTo evaluate the model, execute:")
        print(
            f"python evaluate.py --checkpoint {trainer.checkpoint_callback.best_model_path}"
        )
        print("\nTo run inference and generate samples, execute:")
        print(
            f"python inference.py --checkpoint {trainer.checkpoint_callback.best_model_path}"
        )
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
