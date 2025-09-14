#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from .dataset import SNPDataModule, SNPDataset
from .ddpm import DiffusionModel
from .forward_diffusion import ForwardDiffusion
from .reverse_diffusion import ReverseDiffusion
from .sinusoidal_embedding import SinusoidalPositionEmbeddings, SinusoidalTimeEmbeddings
from .time_sampler import UniformContinuousTimeSampler, UniformDiscreteTimeSampler
from .unet import UNet1D
from .utils import (
    bcast_right,
    load_config,
    prepare_batch_shape,
    set_seed,
    tensor_to_device,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"

__version__ = "0.1.0"

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "SRC_DIR",
    "SCRIPTS_DIR",
    "DATA_DIR",
    "__version__",
    # Dataset
    "SNPDataset",
    "SNPDataModule",
    # Models
    "DiffusionModel",
    "ForwardDiffusion",
    "ReverseDiffusion",
    "UNet1D",
    # Time Samplers
    "UniformContinuousTimeSampler",
    "UniformDiscreteTimeSampler",
    # Sinusoidal Embeddings
    "SinusoidalTimeEmbeddings",
    "SinusoidalPositionEmbeddings",
    # Common Utils
    "load_config",
    "set_seed",
    "bcast_right",
    "tensor_to_device",
    "prepare_batch_shape",
]
