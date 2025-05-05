#!/usr/bin/env python
# coding: utf-8

"""Utility functions for the diffusion model."""

import torch
import numpy as np
import random


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
