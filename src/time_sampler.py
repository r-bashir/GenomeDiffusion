#!/usr/bin/env python
# coding: utf-8

import math
from functools import partial
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Discrete Time Sampling
class UniformDiscreteTimeSampler:

    def __init__(self, tmin: int, tmax: int):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randint(low=self._tmin, high=self._tmax, size=shape)


#  Continuous Time Sampling
class UniformContinuousTimeSampler:
    def __init__(self, tmin: float, tmax: float):
        self._tmin = tmin
        self._tmax = tmax

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        return self._tmin + (self._tmax - self._tmin) * torch.rand(size=shape)
