#!/usr/bin/env python
# coding: utf-8

"""Training script integrated with W&B Sweeps for hyperparameter optimization.

This script trains the diffusion model and is intended to be launched by a W&B
agent during sweeps. It can also be run directly for local testing. Unknown
CLI arguments are parsed as sweep parameters and mapped into the hierarchical
config (see `update_config_with_sweep_params()`), so you can override nested
keys like `unet.use_attention` or `loss.discrete_penalty_weight` from the
command line.

Usage examples:
    # 1) Run locally with a base config only
    python train_sweep.py --config config.yaml

    # 2) Run locally with manual hyperparameter overrides (dot or flat keys)
    python train_sweep.py \
        --config config.yaml \
        --learning_rate 1e-4 \
        --batch_size 32 \
        --unet.use_attention true \
        --attention_heads 4 \
        --loss.use_discrete_loss true \
        --discrete_penalty_weight 0.2

    # 3) Launch a W&B sweep (sweep.yaml should specify program: train_sweep.py)
    wandb sweep sweep.yaml

    # 4) Run W&B agent to execute multiple runs from the sweep
    wandb agent <entity/project>/<sweep_id>

Requirements:
    - WANDB_API_KEY must be configured (env var or `wandb login`).
    - The base config file (default: config.yaml) should exist and be valid.
    - When used via sweeps, parameters from `wandb.config` are merged with any
      CLI overrides passed through the agent.
"""

import argparse
import os
import pathlib
import sys
from typing import Dict, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import DiffusionModel
from src.utils import load_config, set_seed


def update_config_with_sweep_params(config: Dict, sweep_params: Dict) -> Dict:
    """
    Update configuration with W&B sweep parameters.

    Why mapping is needed:
        W&B sweep parameters are flat (e.g., 'learning_rate', 'attention_type'),
        but config.yaml uses a nested structure (e.g., optimizer.lr, unet.attention_type).
        This mapping ensures that sweep parameters are correctly routed into the config dict.

    Only parameters that are actually being tuned in sweep.yaml should be mapped here.
    Any extra mappings are unnecessary and can be removed for clarity and maintainability.

    Args:
        config: Base configuration dictionary
        sweep_params: Parameters from W&B sweep

    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()

    # Only map sweep.yaml parameters to their config locations
    param_mapping = {
        # Data parameters
        "batch_size": ("data", "batch_size"),
        # UNet architecture parameters
        "embedding_dim": ("unet", "embedding_dim"),
        "dim_mults": ("unet", "dim_mults"),
        "with_time_emb": ("unet", "with_time_emb"),
        "with_pos_emb": ("unet", "with_pos_emb"),
        "edge_pad": ("unet", "edge_pad"),
        "norm_groups": ("unet", "norm_groups"),
        # Attention parameters
        "use_attention": ("unet", "use_attention"),
        "attention_type": ("unet", "attention_type"),
        "attention_heads": ("unet", "attention_heads"),
        "attention_dim_head": ("unet", "attention_dim_head"),
        "attention_window": ("unet", "attention_window"),
        "num_global_tokens": ("unet", "num_global_tokens"),
        # Training parameters
        "epochs": ("training", "epochs"),
        "warmup_epochs": ("training", "warmup_epochs"),
        "accumulate_grad_batches": ("training", "accumulate_grad_batches"),
        # Optimizer parameters
        "learning_rate": ("optimizer", "lr"),
        "min_lr": ("optimizer", "min_lr"),
        "weight_decay": ("optimizer", "weight_decay"),
        "amsgrad": ("optimizer", "amsgrad"),
        # Scheduler parameters
        "scheduler_type": ("scheduler", "type"),
        "eta_min": ("scheduler", "eta_min"),
        "scheduler_factor": ("scheduler", "factor"),
        "scheduler_patience": ("scheduler", "patience"),
        "scheduler_mode": ("scheduler", "mode"),
        "scheduler_threshold": ("scheduler", "threshold"),
        "scheduler_min_lr": ("scheduler", "min_lr"),
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
    Validate critical configuration parameters.
    Raises ValueError if configuration is invalid.
    """
    # Validate required sections exist
    required_sections = [
        "data",
        "unet",
        "training",
        "optimizer",
        "scheduler",
        "diffusion",
    ]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate critical parameters have valid values
    if config["data"].get("seq_length", 0) <= 0:
        raise ValueError("seq_length must be positive")

    if config["data"].get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")

    if config["optimizer"].get("lr", 0) <= 0:
        raise ValueError("learning_rate must be positive")

    # Ensure required UNet parameters exist with defaults
    unet_defaults = {
        "channels": 1,
        "enable_checkpointing": True,
        "use_attention": False,
        "attention_heads": 32,
        "attention_dim_head": 32,
        "attention_window": 512,
        "num_global_tokens": 64,
        "dropout": 0.2,
        "use_scale_shift_norm": True,
    }

    for key, default_value in unet_defaults.items():
        if key not in config["unet"]:
            config["unet"][key] = default_value

    # Ensure required diffusion parameters exist with defaults
    diffusion_defaults = {
        "timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.02,
        "beta_schedule": "cosine",
        "denoise_step": 1,
        "discretize": False,
    }

    for key, default_value in diffusion_defaults.items():
        if key not in config["diffusion"]:
            config["diffusion"][key] = default_value

    return config


# Setup logger, specific for Sweeps
def setup_logger(config: Dict) -> WandbLogger:
    """Setup W&B logger for sweep runs.

    Args:
        config: Configuration dictionary

    Returns:
        WandbLogger instance
    """
    # Get base directory for logs
    base_dir = pathlib.Path(config.get("output_path", "outputs"))
    base_dir = base_dir / "sweeps"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Get logger parameters from config
    logger_params = config.get("logger", {})
    if not logger_params:
        logger_params = {"name": None}  # Use W&B auto-naming

    # Get project name
    project_name = logger_params.pop("project", "HPO")

    try:
        # Use existing wandb run since we're in a sweep
        if wandb.run is None:
            raise RuntimeError(
                "No active wandb run found. This should not happen in a sweep."
            )

        wandb_logger = WandbLogger(
            experiment=wandb.run,  # Use existing run
            save_dir=str(base_dir),
            log_model=True,  # Log model checkpoints to W&B
        )
        return wandb_logger

    except Exception as e:
        print(f"Warning: Failed to initialize wandb logger: {str(e)}")
        raise e


def setup_callbacks(config: Dict) -> List[pl.Callback]:
    """Setup training callbacks optimized for sweeps."""
    callbacks = [
        # Model checkpoint, save only best one
        ModelCheckpoint(
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,  # Only save best model
            mode="min",
            save_last=False,  # Don't save last to save space
            auto_insert_metric_name=False,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval="step"),
        # Always use early stopping for sweeps
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
    set_seed(42)

    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    # Load base configuration
    config = load_config(args.config)

    # Initialize wandb for sweep with error handling
    try:
        wandb.init(config=config)
    except Exception as e:
        print(f"❌ Failed to initialize wandb: {e}")
        print("This trial will be skipped.")
        sys.exit(1)  # Exit gracefully to allow next trial

    # Get sweep parameters from command line args (parsed dynamically)
    sweep_params = args.sweep_params.copy()

    # Also get parameters from wandb.config (fallback)
    for key, value in wandb.config.items():
        if key not in sweep_params:
            sweep_params[key] = value

    # Update configuration with sweep parameters
    config = update_config_with_sweep_params(config, sweep_params)
    config = validate_config(config)

    # Log the final configuration
    wandb.config.update(config, allow_val_change=True)

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

    # Setup logger
    train_logger = setup_logger(config)

    # Configure W&B for better plotting
    if train_logger.experiment:
        # Define custom metrics for step-wise plotting
        train_logger.experiment.define_metric(
            "train_loss_step", step_metric="trainer/global_step"
        )
        train_logger.experiment.define_metric(
            "val_loss_step", step_metric="trainer/global_step"
        )
        train_logger.experiment.define_metric("train_loss_epoch", step_metric="epoch")
        train_logger.experiment.define_metric("val_loss_epoch", step_metric="epoch")
        train_logger.experiment.define_metric(
            "lr-AdamW", step_metric="trainer/global_step"
        )

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Enable tensor cores for better performance on CUDA devices
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=train_logger,
        default_root_dir=config["output_path"],
        log_every_n_steps=config["training"].get("log_every_n_steps", 50),
        precision=config["training"].get("precision", "16-mixed"),
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 4),
        enable_checkpointing=True,  # overridden by checkpoint callback
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    try:
        # Train model
        trainer.fit(model)

        # Log final metrics with error handling
        try:
            if trainer.callback_metrics:
                final_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
                final_train_loss = trainer.callback_metrics.get(
                    "train_loss", float("inf")
                )

                wandb.log(
                    {
                        "final/val_loss": final_val_loss,
                        "final/train_loss": final_train_loss,
                        "final/epochs_trained": trainer.current_epoch,
                    }
                )

                print(f"Training completed - Final val_loss: {final_val_loss:.6f}")
        except Exception as log_error:
            print(f"⚠️ Failed to log final metrics: {log_error}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        try:
            wandb.log({"training_failed": True, "error": str(e)})
        except:
            print("⚠️ Could not log training failure to wandb")

        try:
            wandb.finish(exit_code=1)
        except:
            print("⚠️ Could not finish wandb run properly")

        sys.exit(1)  # Exit gracefully to allow next trial

    # Finish W&B run with error handling
    try:
        wandb.finish()
    except Exception as e:
        print(f"⚠️ Failed to finish wandb run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
