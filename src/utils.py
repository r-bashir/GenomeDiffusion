#!/usr/bin/env python
# coding: utf-8

"""Convenient utility functions for the diffusion model."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file with environment variable expansion.

    This function loads a YAML configuration file and expands any environment variables
    in the configuration values. It also ensures that PROJECT_ROOT is set.

    Args:
        config_path: Path to the YAML config file. Can be relative or absolute.

    Returns:
        Dict containing the loaded and processed configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    # Convert to Path and resolve to absolute path
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Make sure PROJECT_ROOT is set
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.parent.absolute())

    # Load the config file
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    # Function to recursively expand environment variables in config
    def expand_env_vars(config):
        if isinstance(config, dict):
            return {k: expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            return os.path.expandvars(config)
        return config

    # Expand environment variables in the config
    return expand_env_vars(config)


# Set global seeds
def set_seed(seed=42):
    """Set random seed for reproducibility across all libraries.

    Args:
        seed (int, optional): Random seed to use. Defaults to 42.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set up logging
def setup_logging(name: str = __name__, debug: bool = False) -> logging.Logger:
    """
    Set up and return a logger with standardized formatting for universal use across the codebase.

    Args:
        name (str): Name for the logger, typically use __name__ for module-level logging.
        debug (bool): If True, sets log level to DEBUG; otherwise INFO.

    Returns:
        logging.Logger: Configured logger instance.

    Usage:
        from src.utils import setup_logging
        logger = setup_logging(__name__, debug=True)
        logger.info("Message")
    """

    logger = logging.getLogger(
        name
    )  # Get or create a logger for the given name (usually module)
    # Prevent adding multiple handlers if logger already configured
    if not logger.handlers:
        handler = logging.StreamHandler()  # Output logs to console
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set log level based on debug flag
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


# Broadcasting
def bcast_right(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Broadcast a tensor to the right dimensions.

    This utility function handles broadcasting tensors to the right shape,
    which is particularly useful for diffusion models where we need to
    broadcast scalar or 1D tensors to match the shape of data tensors.

    Args:
        x: Input tensor to broadcast
        ndim: Target number of dimensions

    Returns:
        Tensor with shape expanded to ndim by adding dimensions on the right

    Raises:
        ValueError: If x.ndim > ndim (cannot reduce dimensions)
    """
    if x.ndim > ndim:
        raise ValueError(f"Cannot broadcast a value with {x.ndim} dims to {ndim} dims.")
    elif x.ndim < ndim:
        difference = ndim - x.ndim
        return x.view(*x.shape, *((1,) * difference))
    else:
        return x


# Add missing channel dimension to a batch: [B, L] to [B, C, L]
def prepare_batch_shape(batch: torch.Tensor) -> torch.Tensor:
    """Ensure batch has shape [B, C, L] for consistent processing.

    Args:
        batch: Input tensor that might need reshaping

    Returns:
        Tensor with shape [B, C, L]
    """
    if len(batch.shape) == 2:  # [B, L]
        return batch.unsqueeze(1)  # Convert to [B, 1, L]
    return batch  # Already [B, C, L]


def tensor_to_device(
    tensor: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Move a tensor to the specified device if it's not already there.

    This utility function provides a consistent way to handle device placement
    throughout the codebase. If no device is specified, it uses CUDA if available.

    Args:
        tensor: The tensor to move to the device
        device: Target device. If None, uses CUDA if available, else CPU

    Returns:
        The tensor on the specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tensor.device != device:
        return tensor.to(device)
    return tensor


# Load DiffusionModel from a Checkpoint.
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Loads a DiffusionModel from a checkpoint and moves it to the specified device.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        model: The loaded DiffusionModel (on the correct device, in eval mode)
        config: The config/hparams dictionary from the checkpoint
    """
    from src import DiffusionModel

    model = DiffusionModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    config = model.hparams
    model = model.to(device)
    model.eval()
    return model, config
