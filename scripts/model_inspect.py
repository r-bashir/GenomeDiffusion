#!/usr/bin/env python
"""Script to inspect the DiffusionModel structure after refactor."""
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion_model import DiffusionModel

# Dummy config for inspection (adjust as needed for your real config)
config = {
    "diffusion": {
        "diffusion_steps": 10,
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

model = DiffusionModel(config)

print("\n=== DiffusionModel Structure ===")
print(model)
print("\nSubmodules:")
print("ForwardDiffusion:", model.forward_diffusion)
print("Noise Predictor (UNet/MLP):", model.unet)
print("ReverseDiffusion:", model.reverse_diffusion_process)
