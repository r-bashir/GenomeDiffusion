#!/usr/bin/env python
# coding: utf-8

"""
Simple W&B Sweep management script.
Initializes sweeps and runs agents, automatically saving best hyperparameters.

Usage:
    # Initialize and run sweep
    python run_sweep.py --config sweep.yaml --count 10

    # Continue an existing sweep
    python run_sweep.py --sweep_id <sweep_id> --count 5
"""

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Dict, Optional

import wandb
import yaml


def initialize_sweep(config_path: str, project: str = "HPO") -> str:
    """Initialize a new W&B sweep.

    Args:
        config_path: Path to sweep configuration file
        project: W&B project name

    Returns:
        Sweep ID
    """
    if not pathlib.Path(config_path).exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    print(
        f"🚀 Initializing sweep with {len(sweep_config.get('parameters', {}))} parameters..."
    )
    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"✅ Sweep initialized: {sweep_id}")
    return sweep_id


def save_best_config(sweep_id: str, project: str) -> Dict:
    """Save the best configuration from completed sweep runs.
    Only saves hyperparameters that were tuned in the sweep.

    Args:
        sweep_id: W&B sweep ID
        project: W&B project name

    Returns:
        Best configuration dictionary
    """
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")
    runs = list(sweep.runs)

    if not runs:
        print("No runs found in sweep")
        return None

    # Get tuned parameters from sweep config
    tuned_params = set(sweep.config.get("parameters", {}).keys())

    # Filter completed runs and find best by val_loss
    completed_runs = [r for r in runs if r.state == "finished"]
    if not completed_runs:
        print("No completed runs found")
        return None

    # Try different validation loss keys
    val_keys = ["val_loss", "val_loss_epoch", "final/val_loss"]
    best_run = None
    best_val = float("inf")
    used_key = None

    for run in completed_runs:
        for key in val_keys:
            if key in run.summary:
                val_loss = run.summary[key]
                if isinstance(val_loss, (int, float)) and val_loss < best_val:
                    best_val = val_loss
                    best_run = run
                    used_key = key
                break

    if not best_run:
        print("No valid runs with validation loss found")
        return None

    # Extract only tuned parameters from best config
    best_config = {k: v for k, v in best_run.config.items() if k in tuned_params}

    # Save to file
    output_file = f"best_config_{sweep_id}.yaml"
    with open(output_file, "w") as f:
        f.write(f"# Best tuned hyperparameters from sweep {sweep_id}\n")
        f.write(f"# Best run: {best_run.name}\n")
        f.write(f"# Validation loss ({used_key}): {best_val:.6f}\n\n")
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"\n🏆 Best run: {best_run.name}")
    print(f"   Val loss: {best_val:.6f}")
    print(f"   Tuned parameters: {len(best_config)}")
    print(f"💾 Config saved to: {output_file}")

    return best_config


def main():
    parser = argparse.ArgumentParser(description="Run W&B sweeps and save best config")
    parser.add_argument("--config", type=str, help="Path to sweep config file")
    parser.add_argument("--sweep_id", type=str, help="Existing sweep ID to continue")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of runs for this agent"
    )
    parser.add_argument("--project", type=str, default="HPO", help="W&B project name")
    args = parser.parse_args()

    if not args.config and not args.sweep_id:
        parser.error("Either --config or --sweep_id must be provided")

    # Set project root
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    try:
        # Get entity from API
        api = wandb.Api()
        entity = api.default_entity
        if not entity:
            raise RuntimeError(
                "Could not determine wandb entity. Please login with 'wandb login'"
            )

        # Get sweep ID
        sweep_id = args.sweep_id
        if not sweep_id:
            sweep_id = initialize_sweep(args.config, args.project)

        # Run agent
        print(f"\n🤖 Running sweep agent for {args.count} runs...")
        cmd = ["wandb", "agent", "--count", str(args.count)]
        cmd.append(f"{entity}/{args.project}/{sweep_id}")

        subprocess.run(cmd, check=True)

        # Save best config
        save_best_config(sweep_id.split("/")[-1], args.project)

    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
