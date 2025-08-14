#!/usr/bin/env python
# coding: utf-8

"""Training script integrated with W&B Sweeps for hyperparameter optimization.

This script is designed to work with W&B Sweeps to systematically explore
hyperparameters and improve model training performance.

Usage:
    # Initialize sweep
    wandb sweep sweep_config.yaml

    # Run sweep agent
    wandb agent <sweep_id>
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
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import DiffusionModel
from src.utils import load_config, set_seed


def update_config_with_sweep_params(config: Dict, sweep_params: Dict) -> Dict:
    """Update configuration with W&B sweep parameters.

    Args:
        config: Base configuration dictionary
        sweep_params: Parameters from W&B sweep

    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()

    # Comprehensive parameter mapping from sweep params to config structure
    param_mapping = {
        # Optimizer parameters
        "learning_rate": ("optimizer", "lr"),
        "weight_decay": ("optimizer", "weight_decay"),
        "betas": ("optimizer", "betas"),
        "eps": ("optimizer", "eps"),
        "amsgrad": ("optimizer", "amsgrad"),
        # Scheduler parameters
        "scheduler_type": ("scheduler", "type"),
        "eta_min": ("scheduler", "eta_min"),
        "scheduler_factor": ("scheduler", "factor"),
        "scheduler_patience": ("scheduler", "patience"),
        "min_lr": ("scheduler", "min_lr"),
        "threshold": ("scheduler", "threshold"),
        # Model architecture parameters
        "embedding_dim": ("unet", "embedding_dim"),
        "dim_mults": ("unet", "dim_mults"),
        "channels": ("unet", "channels"),
        "with_time_emb": ("unet", "with_time_emb"),
        "with_pos_emb": ("unet", "with_pos_emb"),
        "norm_groups": ("unet", "norm_groups"),
        "edge_pad": ("unet", "edge_pad"),
        # Data parameters
        "batch_size": ("data", "batch_size"),
        "seq_length": ("data", "seq_length"),
        "scale_factor": ("data", "scale_factor"),
        "num_workers": ("data", "num_workers"),
        "normalize": ("data", "normalize"),
        "scaling": ("data", "scaling"),
        # Training parameters
        "epochs": ("training", "epochs"),
        "warmup_epochs": ("training", "warmup_epochs"),
        "early_stopping": ("training", "early_stopping"),
        "patience": ("training", "patience"),
        "save_top_k": ("training", "save_top_k"),
        "log_every_n_steps": ("training", "log_every_n_steps"),
        # Diffusion parameters
        "timesteps": ("diffusion", "timesteps"),
        "beta_start": ("diffusion", "beta_start"),
        "beta_end": ("diffusion", "beta_end"),
        "schedule_type": ("diffusion", "schedule_type"),
    }

    # Update configuration with mapped parameters
    for param_name, param_value in sweep_params.items():
        if param_name in param_mapping:
            # Use mapping for known parameters
            section, key = param_mapping[param_name]
            if section not in updated_config:
                updated_config[section] = {}
            updated_config[section][key] = param_value
        elif "." in param_name:
            # Handle dot notation parameters
            keys = param_name.split(".")
            current_dict = updated_config
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            current_dict[keys[-1]] = param_value
        else:
            # Direct parameter
            updated_config[param_name] = param_value

    return updated_config


def validate_config(config: Dict) -> Dict:
    """
    Validate and fix common config issues that can cause training failures.
    This is especially important for sweep configs that might have invalid combinations.
    """

    # Convert string parameters to appropriate types (from W&B sweep)
    def convert_numeric_strings(d: Dict) -> Dict:
        """Recursively convert string numbers to appropriate numeric types."""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = convert_numeric_strings(value)
            elif isinstance(value, str):
                # Try to convert to float if it looks like a number
                try:
                    if "." in value or "e" in value.lower():
                        d[key] = float(value)
                    elif value.isdigit():
                        d[key] = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
        return d

    config = convert_numeric_strings(config)

    # Fix learning rate issues
    optimizer_config = config.get("optimizer", {})
    scheduler_config = config.get("scheduler", {})

    # Ensure min_lr is less than lr
    if "lr" in optimizer_config:
        lr = optimizer_config["lr"]

        # Fix cosine scheduler eta_min
        if scheduler_config.get("type") == "cosine":
            eta_min = scheduler_config.get("eta_min", lr * 0.01)
            if eta_min >= lr:
                scheduler_config["eta_min"] = lr * 0.01
                print(
                    f"Warning: Fixed eta_min ({eta_min:.2e}) >= lr ({lr:.2e}), set to {scheduler_config['eta_min']:.2e}"
                )

        # Fix reduce scheduler min_lr
        elif scheduler_config.get("type") == "reduce":
            min_lr = scheduler_config.get("min_lr", lr * 0.001)
            if min_lr >= lr:
                scheduler_config["min_lr"] = lr * 0.001
                print(
                    f"Warning: Fixed min_lr ({min_lr:.2e}) >= lr ({lr:.2e}), set to {scheduler_config['min_lr']:.2e}"
                )

    # Ensure reasonable batch size given memory constraints
    batch_size = config.get("data", {}).get("batch_size", 32)
    embedding_dim = config.get("unet", {}).get("embedding_dim", 32)
    seq_length = config.get("data", {}).get("seq_length", 100)

    # Reduce batch size for larger models/sequences to prevent OOM
    if embedding_dim >= 64 and seq_length >= 200 and batch_size > 16:
        config["data"]["batch_size"] = 16
        print(
            f"Reduced batch size to 16 for large model/sequence (emb={embedding_dim}, seq={seq_length})"
        )
    elif embedding_dim >= 128 and batch_size > 8:
        config["data"]["batch_size"] = 8
        print(f"Reduced batch size to 8 for very large model (emb={embedding_dim})")

    return config


def setup_callbacks(config: Dict) -> List:
    """Setup training callbacks optimized for sweeps."""
    callbacks = [
        # Save only the best checkpoint to save space
        ModelCheckpoint(
            dirpath=f"{config['output_path']}/checkpoints",
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,  # Only save best model
            mode="min",
            save_last=False,  # Don't save last to save space
            auto_insert_metric_name=False,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
        # Always use early stopping for sweeps to save compute
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"].get("patience", 15),
            mode="min",
            min_delta=1e-6,  # Small improvement threshold
        ),
    ]

    return callbacks


def parse_args():
    """Parse command line arguments dynamically."""
    parser = argparse.ArgumentParser(description="Train diffusion model with W&B sweep")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    # Parse known args to handle any sweep parameters dynamically
    args, unknown = parser.parse_known_args()

    # Parse sweep parameters from unknown args
    sweep_params = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            param_name = unknown[i][2:]  # Remove '--'
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                param_value = unknown[i + 1]
                # Try to convert to appropriate type
                try:
                    # Handle negative numbers
                    if (
                        param_value.startswith("-")
                        and param_value[1:]
                        .replace(".", "")
                        .replace("e", "")
                        .replace("-", "")
                        .isdigit()
                    ):
                        sweep_params[param_name] = float(param_value)
                    elif "." in param_value or "e" in param_value.lower():
                        sweep_params[param_name] = float(param_value)
                    elif param_value.isdigit():
                        sweep_params[param_name] = int(param_value)
                    elif param_value.lower() in ["true", "false"]:
                        sweep_params[param_name] = param_value.lower() == "true"
                    else:
                        sweep_params[param_name] = param_value
                except ValueError:
                    sweep_params[param_name] = param_value
                i += 2
            else:
                i += 1
        else:
            i += 1

    # Add sweep_params to args
    args.sweep_params = sweep_params
    return args


def main():
    """Main training function for W&B Sweeps."""

    # Parse arguments
    args = parse_args()
    wandb.init()
    set_seed(42)

    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    # Load base configuration
    config = load_config(args.config)

    # Get sweep parameters from command line args (parsed dynamically)
    sweep_params = args.sweep_params.copy()

    # Also get parameters from wandb.config (fallback)
    for key, value in wandb.config.items():
        if key not in sweep_params:
            sweep_params[key] = value

    if sweep_params:
        print(f"Sweep parameters received: {list(sweep_params.keys())}")
    else:
        print("No sweep parameters received - using base config")

    # Update configuration with sweep parameters
    config = update_config_with_sweep_params(config, sweep_params)
    config = validate_config(config)

    # Log the final configuration
    wandb.config.update(config, allow_val_change=True)

    # Create consistent sweep output structure: ./sweeps/<project_name>/<run_number>/
    run_name = wandb.run.name or wandb.run.id
    project_name = config.get("project_name", "GenomeDiffusion")

    # Use PROJECT_ROOT for consistent paths
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    sweep_output_path = f"{project_root}/sweeps/{project_name}/{run_name}"

    # Create directory structure
    os.makedirs(sweep_output_path, exist_ok=True)
    os.makedirs(f"{sweep_output_path}/checkpoints", exist_ok=True)

    # Update config to use sweep output path
    config["output_path"] = sweep_output_path

    # Print key configuration for debugging
    print(f"\n=== SWEEP RUN: {run_name} ===")
    print(f"Output Path: {sweep_output_path}")
    print(f"Learning Rate: {config.get('optimizer', {}).get('lr', 'N/A')}")
    print(f"Scheduler Type: {config.get('scheduler', {}).get('type', 'N/A')}")
    print(f"Batch Size: {config.get('data', {}).get('batch_size', 'N/A')}")
    print(f"Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
    print(f"Weight Decay: {config.get('optimizer', {}).get('weight_decay', 'N/A')}")

    try:
        # Initialize model
        print("Initializing model...")
        model = DiffusionModel(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Log model info to wandb
        wandb.log(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            }
        )

    except Exception as e:
        print(f"Failed to initialize model: {e}")
        wandb.finish(exit_code=1)
        return

    # Setup logger (W&B) - save to sweep output directory
    logger = WandbLogger(
        project=f"{project_name}-HPO",
        name=run_name,
        save_dir=sweep_output_path,
        log_model=False,  # Don't log model artifacts to save space
        log_graph=False,  # Don't log model graph to save space
    )

    # Configure W&B for better plotting
    if logger.experiment:
        # Define custom metrics for step-wise plotting
        logger.experiment.define_metric(
            "train_loss_step", step_metric="trainer/global_step"
        )
        logger.experiment.define_metric(
            "val_loss_step", step_metric="trainer/global_step"
        )
        logger.experiment.define_metric("train_loss_epoch", step_metric="epoch")
        logger.experiment.define_metric("val_loss_epoch", step_metric="epoch")
        logger.experiment.define_metric("lr-AdamW", step_metric="trainer/global_step")

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Initialize trainer with sweep output directory
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        default_root_dir=sweep_output_path,
        log_every_n_steps=config["training"].get("log_every_n_steps", 10),
        enable_checkpointing=True,
        enable_progress_bar=False,  # Disable for cleaner sweep logs
        enable_model_summary=False,  # Disable for cleaner logs
        precision=16,  # Use mixed precision to save memory
    )

    try:
        # Train model
        print("Starting training...")
        trainer.fit(model)

        # Log final metrics
        if trainer.callback_metrics:
            final_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
            final_train_loss = trainer.callback_metrics.get("train_loss", float("inf"))

            wandb.log(
                {
                    "final/val_loss": final_val_loss,
                    "final/train_loss": final_train_loss,
                    "final/epochs_trained": trainer.current_epoch,
                }
            )

            print(f"Training completed - Final val_loss: {final_val_loss:.6f}")

    except Exception as e:
        print(f"Training failed: {e}")
        wandb.log({"training_failed": True, "error": str(e)})
        wandb.finish(exit_code=1)
        return

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
