#!/usr/bin/env python
# coding: utf-8

"""Script for evaluating trained SNP diffusion model."""

import argparse
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import yaml

from diffusion.diffusion_model import DiffusionModel
from diffusion.inference_callback import InferenceCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test SNP diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for test results (default: checkpoint directory)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (default: from config)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main testing function."""
    # Load configuration
    config = load_config(args.config)

    # Override output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override number of samples if specified
    if args.num_samples:
        config["training"]["num_samples"] = args.num_samples

    # Initialize model from checkpoint
    model = DiffusionModel.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",  # Will be automatically moved to GPU if available
        hparams=config,
    )

    # Set up inference callback
    inference_callback = InferenceCallback(output_dir=str(output_dir))

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",  # Will automatically detect and use GPU if available
        devices="auto",      # Use all available devices
        precision="bf16-mixed",  # Use bfloat16 mixed precision
        callbacks=[inference_callback],
        default_root_dir=str(output_dir),
    )

    # Run testing
    trainer.test(model)


if __name__ == "__main__":
    args = parse_args()
    main(args)
