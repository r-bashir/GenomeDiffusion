#!/usr/bin/env python
# coding: utf-8

"""Script for evaluating trained SNP diffusion models.

Examples:
    # Evaluate a trained model using its checkpoint
    python test.py --config config.yaml --checkpoint path/to/checkpoint.ckpt

    # Evaluate best model from a training run
    python test.py --config config.yaml --checkpoint path/to/run/checkpoints/best.ckpt

Evaluation results are saved in the 'evaluation' directory, including:
- ROC curves
- Confusion matrices
- Test metrics (MSE, MAE)
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import yaml

from diffusion.diffusion_model import DiffusionModel
from diffusion.evaluation_callback import EvaluationCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SNP diffusion model on test dataset"
    )
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
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main evaluation function."""
    # Validate input files
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Load configuration
    config = load_config(args.config)

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    if "checkpoints" in str(checkpoint_path):
        base_dir = (
            checkpoint_path.parent.parent
        )  # Go up two levels: from checkpoint file to run directory
    else:
        base_dir = checkpoint_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluation directory for model assessment results
    output_dir = base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    # Initialize model from checkpoint
    try:
        print("Loading model from checkpoint...")
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            strict=True,
            config=config,
        )
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully on {next(model.parameters()).device}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Set up evaluation callback for metrics
    eval_callback = EvaluationCallback(output_dir=str(output_dir))

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        callbacks=[eval_callback],
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Run evaluation on test dataset
    try:
        print("Starting model evaluation...")
        trainer.test(model)
        print("Evaluation completed successfully")
        print(f"Test metrics and plots saved to: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
