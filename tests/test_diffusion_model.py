#!/usr/bin/env python
# coding: utf-8

"""Unit tests for the diffusion model."""

import torch

from src.ddpm import DiffusionModel


def get_dummy_config():
    return {
        "diffusion": {
            "time_steps": 10,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "linear",
        },
        "time_sampler": {"tmin": 1, "tmax": 10},
        "unet": {
            "embedding_dim": 8,
            "dim_mults": [1, 2],
            "channels": 1,
            "with_time_emb": True,
            "with_pos_emb": True,
            "resnet_block_groups": 1,
        },
        "data": {"seq_length": 32},
    }


def test_forward_pass():
    model = DiffusionModel(get_dummy_config())
    x = torch.randn(2, 1, 32)
    t = torch.randint(1, 11, (2,))
    out = model.forward(x, t)
    assert out.shape == (2, 1, 32)


def test_loss_computation():
    model = DiffusionModel(get_dummy_config())
    batch = torch.randn(2, 1, 32)
    loss = model.compute_loss(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar


def test_sample_generation():
    model = DiffusionModel(get_dummy_config())
    samples = model.generate_samples(num_samples=2, denoise_step=2)
    assert samples.shape == (2, 1, 32)
