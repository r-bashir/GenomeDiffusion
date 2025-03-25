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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from diffusion_model import DiffusionModel


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
    config: Dict, output_dir: Optional[str], resume_from_checkpoint: Optional[str]
) -> Union[TensorBoardLogger, WandbLogger]:
    """Setup logger based on configuration.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path (overrides config output_path)
        resume_from_checkpoint: Path to checkpoint for resuming training

    Returns:
        Logger instance
    """
    # Use output_dir if provided, otherwise use config output_path
    base_dir = (
        output_dir if output_dir is not None else config.get("output_path", "output")
    )

    # Create the lightning_logs directory in the output path
    logs_dir = os.path.join(base_dir, "lightning_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Extract version number from checkpoint path if resuming
    version = get_version_from_checkpoint(resume_from_checkpoint)

    logger_type = config["training"]["logger"]
    if logger_type == "wandb":
        try:
            import wandb

            # Get WANDB API key from environment
            wandb_key = os.environ.get("WANDB_API_KEY")
            if not wandb_key:
                print("Warning: WANDB_API_KEY environment variable not set, falling back to TensorBoard")
                return TensorBoardLogger(save_dir=logs_dir, name="", version=version)
            
            # Initialize wandb with API key
            wandb.login(key=wandb_key)
            
            # Create a proper WandbLogger instance
            wandb_logger = WandbLogger(
                project=config.get("project_name", "GenomeDiffusion"),
                save_dir=logs_dir,
                config=config,
                version=version,
                resume="allow" if resume_from_checkpoint else None,
            )
            return wandb_logger
        except Exception as e:
            print(f"Warning: Failed to initialize wandb ({str(e)}), falling back to TensorBoard")
            return TensorBoardLogger(save_dir=logs_dir, name="", version=version)
    elif logger_type == "tb":
        # Use the lightning_logs directory for TensorBoard
        tb_logger = TensorBoardLogger(save_dir=logs_dir, name="", version=version)
        return tb_logger
    else:
        raise ValueError(f"Logger '{logger_type}' not recognized. Use 'wandb' or 'tb'.")


def setup_callbacks(config: Dict) -> List:
    """Setup training callbacks."""

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
    return callbacks


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)

    # Determine output directory (command line arg overrides config)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else config.get("output_path", "output")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = DiffusionModel(config)

    # Setup logger and callbacks
    logger = setup_logger(config, output_dir, args.resume_from_checkpoint)
    callbacks = setup_callbacks(config)

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["training"]["num_epochs"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        logger=logger,
        callbacks=callbacks,
        precision="16-mixed", 
        accumulate_grad_batches=config["training"].get("grad_accum", 2),
        val_check_interval=config["training"].get("val_check_interval", 0.5),
        strategy="auto",
    )

    # Train model
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model)

    # Test model if validation performance is good
    test_threshold = config["training"].get("test_threshold", float("inf"))
    val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
    if val_loss < test_threshold:
        trainer.test(model)

    # Generate and save samples if requested
    if args.generate_samples:
        samples = model.generate_samples(
            num_samples=config["training"].get("num_samples", 10)
        )

        # Save samples in the same version directory as the logs
        if hasattr(logger, "log_dir"):
            # For TensorBoard or WandbLogger
            samples_path = os.path.join(logger.log_dir, "generated_samples.pt")
        else:
            # Fallback to output directory
            samples_path = os.path.join(output_dir, "generated_samples.pt")

        torch.save(samples, samples_path)
        print(f"Generated samples shape: {samples.shape}")
        print(f"Samples saved to {samples_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
