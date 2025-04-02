#!/usr/bin/env python
# coding: utf-8

"""Main steering script for training SNP diffusion model."""
# python train.py --config config.yaml
# python train.py --resume_from_checkpoint output/lightning_logs/version_x/checkpoints/last.ckpt
# python train.py --config config.yaml --generate_samples

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
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger

from diffusion import DiffusionModel, inference_callback
from diffusion.inference_callback import InferenceCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config.yaml output_path)",
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate samples after training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
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
    logger_params = {
        "name": "",
        "save_dir": logs_dir,
        "version": version
    }
    
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
        raise ValueError(f"Logger '{logger_type}' not recognized. Use 'wandb', 'tb', or 'csv'.")


def setup_callbacks(config: Dict) -> List:
    """Setup training callbacks."""
    # Get base directory for logs
    base_dir = config.get("output_path", "output")
    logs_dir = os.path.join(base_dir, "metric_logs")
    os.makedirs(logs_dir, exist_ok=True)

    callbacks = [
        # Save best and last checkpoints
        ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("patience", 10),
            mode="min",
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
    ]

    # Add inference callback for test metrics if enabled
    if config["training"].get("compute_test_metrics", True):
        metric_logs_dir = os.path.join(config["output_path"], "metric_logs")
        os.makedirs(metric_logs_dir, exist_ok=True)
        callbacks.append(InferenceCallback(output_dir=metric_logs_dir))

    return callbacks


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        config["output_path"] = args.output_dir

    # Initialize model
    model = DiffusionModel(config)

    # Set up logger
    logger = setup_logger(config, args.resume_from_checkpoint)

    # Set up callbacks
    callbacks = setup_callbacks(config)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=logger,
        callbacks=callbacks,
        default_root_dir=config["output_path"],
    )

    # Train model
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model)

    # Run test phase if enabled in config
    if config["training"].get("compute_test_metrics", True):
        print("Running test phase to compute metrics...")
        trainer.test(model)
    else:
        print("Skipping test phase (compute_test_metrics is False in config)")

    # Generate and save samples if requested
    if args.generate_samples:
        print("Generating samples...")
        samples = model.generate_samples(
            num_samples=config["training"].get("num_samples", 10)
        )

        # Get metric logs directory
        base_dir = config.get("output_path", "output")
        metric_logs_dir = os.path.join(base_dir, "metric_logs")
        os.makedirs(metric_logs_dir, exist_ok=True)

        # Save samples in metric_logs directory
        samples_path = os.path.join(metric_logs_dir, "generated_samples.pt")
        torch.save(samples, samples_path)
        print(f"Generated samples shape: {samples.shape}")
        print(f"Samples saved to {samples_path}")

        # Also save a copy in logger's directory if available
        if isinstance(logger, (pl.loggers.TensorBoardLogger, pl.loggers.CSVLogger, pl.loggers.WandbLogger)):
            try:
                logger_samples_path = os.path.join(logger.log_dir, "generated_samples.pt")
                os.makedirs(os.path.dirname(logger_samples_path), exist_ok=True)
                torch.save(samples, logger_samples_path)
                print(f"Additional copy saved to {logger_samples_path}")
            except (TypeError, AttributeError):
                print("Could not save to logger directory")


if __name__ == "__main__":
    args = parse_args()
    main(args)
