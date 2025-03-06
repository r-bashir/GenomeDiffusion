#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml

from data_loading import load_data
from model import DiffusionModel

# Set CUDA device if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_config(config: Dict) -> None:
    """Validate that all required configuration values are present.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required configuration values are missing
    """
    required_keys = {
        'input_path': str,
        'diffusion': {'num_diffusion_timesteps': int, 'beta_start': float, 'beta_end': float},
        'time_sampler': {'tmin': int, 'tmax': int},
        'unet': {'embedding_dim': int, 'dim_mults': list, 'channels': int},
        'data': {'seq_length': int, 'batch_size': int, 'num_workers': int},
        'training': {'num_epochs': int, 'gradient_clip_val': float},
        'optimizer': {'lr': float}
    }
    
    def check_keys(d: Dict, keys: Dict, path: str = ''):
        for key, value_type in keys.items():
            if key not in d:
                raise ValueError(f'Missing required config value: {path + key}')
            if isinstance(value_type, dict):
                if not isinstance(d[key], dict):
                    raise ValueError(f'Expected dict for {path + key}')
                check_keys(d[key], value_type, f'{path}{key}.')
            elif not isinstance(d[key], value_type):
                raise ValueError(f'Invalid type for {path + key}: expected {value_type}, got {type(d[key])}')
    
    check_keys(config, required_keys)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing model configuration.
        
    Raises:
        ValueError: If required configuration values are missing
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        List of PyTorch Lightning callbacks.
    """
    callbacks = [
        # Save best models
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            save_top_k=config['training']['save_top_k'],
            mode='min'
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('patience', 10),
            mode='min'
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='step')
    ]
    return callbacks


def setup_trainer(config: dict, callbacks: list) -> pl.Trainer:
    """Setup PyTorch Lightning trainer.
    
    Args:
        config: Model configuration dictionary.
        callbacks: List of callbacks.
        
    Returns:
        Configured PyTorch Lightning trainer.
    """
    return pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=config['training']['num_epochs'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        logger=TensorBoardLogger('lightning_logs', name='diffusion_model'),
        callbacks=callbacks,
        precision=config['training'].get('precision', 16),  # Default to mixed precision for memory efficiency
        accumulate_grad_batches=config['training'].get('grad_accum', 2),  # Default to gradient accumulation
        val_check_interval=config['training'].get('val_check_interval', 0.5),  # Validate twice per epoch by default
        strategy='ddp_find_unused_parameters_false' if torch.cuda.is_available() else None  # DDP strategy for multi-GPU
    )


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Use command line input_path if provided, otherwise keep config value
    if args.data_path:
        config['input_path'] = args.data_path
    
    # Initialize model with gradient checkpointing for memory efficiency
    model = DiffusionModel(hparams=config)
    if config['training'].get('gradient_checkpointing', True):
        model.unet.gradient_checkpointing_enable()
    
    # Setup training
    callbacks = setup_callbacks(config)
    trainer = setup_trainer(config, callbacks)
    
    # Train model
    trainer.fit(model)
    
    # Test model if validation performance is good
    if trainer.callback_metrics.get('val_loss', float('inf')) < config['training'].get('test_threshold', float('inf')):
        trainer.test(model)
    
    # Generate and save samples
    if args.generate_samples:
        with torch.no_grad():
            samples = model.sample(sample_size=config['training'].get('num_samples', 10))
            torch.save(samples, Path(args.output_dir) / 'generated_samples.pt')
            print(f"Generated samples shape: {samples.shape}")
            print(f"Samples saved to {args.output_dir}/generated_samples.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SNP Diffusion Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=False,
                        help='Path to SNP dataset (overrides config.yaml input_path)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--generate_samples', action='store_true',
                        help='Generate samples after training')
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
