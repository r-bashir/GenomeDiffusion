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
from pathlib import Path
import pytorch_lightning as pl
import torch
from diffusion.diffusion_model import DiffusionModel
from diffusion.evaluation_callback import EvaluationCallback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SNP diffusion model on test dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    return parser.parse_args()


def main():
    """Main function."""

    # Parse arguments
    args = parse_args()

    try:
        print("\nLoading model from checkpoint...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model from checkpoint
        model = DiffusionModel.load_from_checkpoint(
            args.checkpoint,
            map_location=device,
            strict=True,
        )

        config = model.hparams  # model config used during training
        model = model.to(device)  # move model to device
        model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully from checkpoint on {device}")
        print("Model config loaded from checkpoint:\n")
        print(config)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")

    # Setup output directory
    checkpoint_path = Path(args.checkpoint)
    if "checkpoints" in str(checkpoint_path):
        base_dir = checkpoint_path.parent.parent
    else:
        base_dir = checkpoint_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nEvaluation results will be saved to: {output_dir}")

    # Evaluation callback
    eval_callback = EvaluationCallback(output_dir=str(output_dir))

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        callbacks=[eval_callback],
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Run evaluation on test dataset
    try:
        print("\nStarting model evaluation...")
        trainer.test(model)
        print("Evaluation completed successfully")
        print(f"\nTest metrics and plots saved to: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")


# Entry point
if __name__ == "__main__":
    main()
