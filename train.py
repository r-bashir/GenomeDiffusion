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
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from src import DiffusionModel
from src.utils import load_config, set_seed, setup_logging


def get_version_from_checkpoint(checkpoint_path: Optional[str]) -> Optional[str]:
    """Extract version string from checkpoint path if resuming.

    Args:
        checkpoint_path: Path to checkpoint for resuming training

    Returns:
        Version string if found, None otherwise
    """
    if not checkpoint_path:
        return None

    # Extract version from checkpoint path, assuming path structure:
    # as `lightning_logs/<project_name>/<version>/checkpoints/last.ckpt`
    path = pathlib.Path(checkpoint_path)
    for parent in path.parents:
        # Check if this is a version directory under lightning_logs/<project_name>/
        if (
            parent.parent
            and parent.parent.parent
            and parent.parent.parent.name == "lightning_logs"
            and (parent / "checkpoints").exists()
        ):

            version = parent.name

            # Determine if it's a standard numeric version or WandB random string
            if parent.name.startswith("version_"):
                print(f"Resuming logging to standard version directory: {parent}")
                # Return just the numeric part for standard versions
                return parent.name.split("_")[1]
            else:
                print(f"Resuming logging to WandB version directory: {parent}")
                # Return the full random string for WandB versions
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
    base_dir = pathlib.Path(config.get("output_path", "outputs"))
    project_logs_dir = base_dir / "lightning_logs"
    project_logs_dir.mkdir(parents=True, exist_ok=True)

    # Get project name
    project_name = config.get("project_name", "GenDiff")

    # Get version from checkpoint if resuming
    version = get_version_from_checkpoint(resume_from_checkpoint)

    # Common params (only for tb and csv)
    logger_params = {
        "save_dir": project_logs_dir,
        "name": project_name,  # project directory name for tb/csv
        "version": version,
    }

    # Select logger type from config
    logger_type = config["training"]["logger"]

    if logger_type == "wandb":
        try:
            import wandb

            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key and not wandb.api.api_key:
                raise ValueError(
                    "WANDB_API_KEY not found in environment variables and no API key configured.\n"
                    "Please either:\n"
                    "1. Set WANDB_API_KEY environment variable, or\n"
                    "2. Change logger type in config.yaml to 'tb' or 'csv'"
                )

            wandb_logger = WandbLogger(
                save_dir=project_logs_dir,
                version=version,
                project=project_name,  # WandB project name
                config=config,
                resume="allow" if resume_from_checkpoint else None,
            )
            return wandb_logger

        except ValueError as e:
            # Re-raise ValueError (missing API key) as-is
            raise e
        except Exception as e:
            # Handle other WandB initialization errors
            raise RuntimeError(f"WandB initialization failed: {str(e)}")

    elif logger_type == "tb":
        return TensorBoardLogger(**logger_params)

    elif logger_type == "csv":
        return CSVLogger(**logger_params)

    else:
        raise ValueError(
            f"Logger '{logger_type}' not recognized. Use 'wandb', 'tb', or 'csv'."
        )


def setup_callbacks(config: Dict) -> List[pl.Callback]:
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

    # Early stopping if enabled in config
    if config["training"].get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["training"].get("patience", 10),
                mode="min",
            )
        )

    return callbacks


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


def main():
    """Main training function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(name="train")
    logger.info("Starting `train.py` script.")

    # Set a seed for reproducibility
    set_seed(42)

    # Only set PROJECT_ROOT if it's not already defined in the environment
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    # Load configuration
    config = load_config(args.config)

    # Print key paths for verification
    logger.info(f"Using PROJECT_ROOT: {os.environ.get('PROJECT_ROOT')}")
    logger.info(f"Input path: {config['data']['input_path']}")
    logger.info(f"Output path: {config['output_path']}")

    # Ensure output directory exists
    os.makedirs(config["output_path"], exist_ok=True)

    # Initialize model
    try:
        logger.info("Initializing model...")
        model = DiffusionModel(config)
        logger.info("Model initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

    # Set up loggers
    train_logger = setup_logger(config, args.resume)

    # Set up callbacks
    callbacks = setup_callbacks(config)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=train_logger,
        default_root_dir=config["output_path"],
        log_every_n_steps=config["training"].get("log_every_n_steps", 50),
        enable_checkpointing=True,  # overridden by checkpoint callback
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    logger.info("Trainer initialized successfully...\n")

    # Train model
    try:
        logger.info("Training is started...\n")
        trainer.fit(
            model,
            ckpt_path=args.resume,
        )
        logger.info("Training is finished...\n")

        # Best checkpoint path
        best_checkpoint_path = trainer.checkpoint_callback.best_model_path

        print("To run inference locally, execute:")
        print(f"$ python inference.py --checkpoint {best_checkpoint_path}")

        print("\nTo run inference on cluster, execute:")
        print(f"$ ./inference.sh {best_checkpoint_path}")

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


if __name__ == "__main__":
    main()
