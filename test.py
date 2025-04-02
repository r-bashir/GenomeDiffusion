#!/usr/bin/env python
# coding: utf-8

"""Script for evaluating trained SNP diffusion model."""

import argparse
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
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
        "--generate_samples",
        action="store_true",
        help="Generate samples after testing",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main testing function."""
    # Validate input files
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Load configuration
    config = load_config(args.config)

    # Get the wandb run directory (one level up from checkpoint)
    checkpoint_path = Path(args.checkpoint)
    if 'checkpoints' in str(checkpoint_path):
        output_dir = checkpoint_path.parent.parent  # Go up two levels: from checkpoint file to run directory
    else:
        output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Test outputs will be saved to: {output_dir}")

    # Initialize model from checkpoint
    try:
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location="cpu",  # Will be automatically moved to GPU if available
            strict=True,  # Ensure all weights are loaded correctly
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

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
    try:
        trainer.test(model)
    except Exception as e:
        raise RuntimeError(f"Testing failed: {e}")

    # Generate samples if requested
    if args.generate_samples:
        print("Generating samples...")
        try:
            # Use default number of samples from config or fallback to 10
            samples = model.generate_samples(num_samples=config["training"].get("num_samples", 10))
            
            # Save samples
            samples_path = output_dir / "generated_samples.pt"
            torch.save(samples, samples_path)
            print(f"Generated samples shape: {samples.shape}")
            print(f"Samples saved to {samples_path}")
            
        except Exception as e:
            print(f"Warning: Sample generation failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
