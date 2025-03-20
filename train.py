#!/usr/bin/env python
# coding: utf-8

"""Main steering script for training SNP diffusion model."""
# python train.py --config config.yaml
# python train.py --config config.yaml --generate_samples

import argparse
import os
import math
import pathlib

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

# Import the new DiffusionModel that inherits from NetworkBase
from diffusion_model import DiffusionModel

# Set CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save outputs"
    )
    parser.add_argument(
        "--generate_samples",
        action="store_true",
        help="Generate samples after training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def set_logger(config):
    if config['training']['logger'] == 'wandb':
        import wandb
        wandb.init(project=config.get('project_name', 'GenomeDiffusion'), config=config)
        return wandb
    elif config['training']['logger'] == 'tb':
        tb_logger = TensorBoardLogger(config['training']['tb_log_dir'])
        return tb_logger
    elif config['training']['logger'] == 'csv':
        csv_logger = CSVLogger(save_dir=config.get('csv_log_path', 'logs'))
        return csv_logger
    else:
        raise ValueError("Logger not recognized. Use 'wandb', 'tb', or 'csv'.")


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks."""
    callbacks = [
        # Save best models
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            save_top_k=config["training"]["save_top_k"],
            mode="min",
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
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model - now using the DiffusionModel that inherits from NetworkBase
    model = DiffusionModel(config)
    
    # Setup logger and callbacks
    logger = set_logger(config)
    callbacks = setup_callbacks(config)
    
    # Setup trainer
    trainer_args = {
        'accelerator': 'cuda' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'max_epochs': config['training']['num_epochs'],
        'gradient_clip_val': config['training']['gradient_clip_val'],
        'logger': logger,
        'callbacks': callbacks,
        'precision': config['training'].get('precision', 16),  # Default to mixed precision
        'accumulate_grad_batches': config['training'].get('grad_accum', 2),
        'val_check_interval': config['training'].get('val_check_interval', 0.5),
        'strategy': 'auto',
    }
    
    trainer = pl.Trainer(**trainer_args)
    
    # Train model
    trainer.fit(model)
    
    # Test model if validation performance is good
    if trainer.callback_metrics.get("val_loss", float("inf")) < config["training"].get(
        "test_threshold", float("inf")
    ):
        trainer.test(model)
    
    # Generate and save samples
    if args.generate_samples:
        samples = model.generate_samples(num_samples=config["training"].get("num_samples", 10))
        torch.save(samples, os.path.join(args.output_dir, "generated_samples.pt"))
        print(f"Generated samples shape: {samples.shape}")
        print(f"Samples saved to {args.output_dir}/generated_samples.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
