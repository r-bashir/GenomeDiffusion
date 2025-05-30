#!/usr/bin/env python
# coding: utf-8
"""
Script to run detailed noise analysis on a trained diffusion model.
Initial settings, argument parsing, model/data loading, and reproducibility follow run_diffusion.py.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.utils.noise_batch_utils import (
    plot_loss_vs_timestep,
    plot_noise_scales,
    run_noise_analysis,
    save_noise_analysis,
)

# Import model and utils
from src.diffusion_model import DiffusionModel


# ========== Argument Parsing ==========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run noise analysis on a trained diffusion model."
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input data file (e.g., .npy or .pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="noise_analysis_results",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3, help="Number of samples for analysis"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timesteps to analyze (default: preset list)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


# ========== Reproducibility ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== Main ==========
def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Model ----
    print(f"Loading model from {args.model_checkpoint}")
    model = DiffusionModel.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    # ---- Load Data ----
    print(f"Loading data from {args.data_path}")
    if args.data_path.endswith(".npy"):
        x0 = torch.from_numpy(np.load(args.data_path))
    else:
        x0 = torch.load(args.data_path)
    x0 = x0.float()

    # ---- Run Noise Analysis ----
    print("\nRunning noise analysis...")
    results = run_noise_analysis(
        model=model,
        x0=x0,
        num_samples=args.num_samples,
        timesteps=args.timesteps,
        verbose=True,
        output_dir=output_dir,
    )

    # ---- Save and Plot Results ----
    save_noise_analysis(results, output_dir)
    plot_loss_vs_timestep(results, output_dir)
    plot_noise_scales(results, output_dir)
    print(f"\nNoise analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
