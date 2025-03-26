#!/usr/bin/env python
# coding: utf-8

"""This script implements the SNPDataset and SNPDataModule classes."""

import argparse
import os
import yaml
from typing import Dict
import torch
from diffusion.dataset import load_data
import wandb

def check_path_exists(path):
    """Check if a path exists."""
    return os.path.exists(path)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (overrides config.yaml output_path)",
    )
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    """Main function to test SNPDataset and SNPDataModule."""
    
    # Load configuration
    config = load_config(args.config)

    # Determine output directory (command line arg overrides config)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else config.get("output_path", "output")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(project=config["project_name"],
               entity=config["wandb_entity"],
               dir=output_dir)
    
    # Log a test message
    wandb.log({"test": "wandb initialized and authenticated successfully"})

    # Test Loading
    print("\nExamining data:")
    dataset = load_data(input_path=config["input_path"])

    # Analyze unique values in the data
    unique_values = torch.unique(dataset)
    print(f"\nUnique values in data: {unique_values.tolist()}")

    # Count occurrences of each value
    value_counts = {}
    for value in [0.0, 0.5, 1.0, 9.0]:
        count = (dataset == value).sum().item()
        percentage = (count / dataset.numel()) * 100
        value_counts[value] = (count, percentage)

    print("\nValue distribution (percentage):")
    for value, (count, percentage) in value_counts.items():
        print(f"{value:.1f}: {count} occurrences ({percentage:.2f}%)")

    print(f"\nDataset length: {len(dataset)}")
    print(f"First example: {dataset[0].shape}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
