#!/usr/bin/env python
# coding: utf-8

"""Utility functions for the diffusion model."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


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
