#!/usr/bin/env python
# coding: utf-8

"""
Local W&B Sweep Workflow:
    1. python run_sweep.py --init --config sweep.yaml --project <hpo>  # Initialize
    2. python run_sweep.py --agent sweep_id --project <hpo>  # Run agent(s)
    3. python run_sweep.py --analyze sweep_id --project <hpo>  # Analyze

Cluster W&B Sweep Workflow:
    # Single sweep job (init + agent + analyze inside container)
    sbatch sweep.slurm sweep.yaml MyProject

    # Multiple coordinated agents (self-contained: init + agents + analyze)
    sbatch sweep_parallel.slurm sweep.yaml 5 MyProject

    # Job arrays at scale (self-contained: init by task 1 + agents + analyze)
    sbatch --array=1-20 agent.slurm MyProject sweep.yaml

    # Direct container execution (test locally through container)
    bash sweep.slurm sweep.yaml MyProject

"""

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Dict, Optional

import yaml

import wandb


def initialize_sweep(config_path: str, project: str = "HPO") -> str:
    """Initialize a new W&B sweep."""
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


def save_best_config(
    sweep_id: str, project: str, entity: Optional[str] = None
) -> Optional[Dict]:
    """Save the best configuration from completed sweep runs."""
    api = wandb.Api()
    sweep_path = f"{entity}/{project}/{sweep_id}" if entity else f"{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    runs = list(sweep.runs)

    if not runs:
        print("No runs found in sweep")
        return None

    tuned_params = set(sweep.config.get("parameters", {}).keys())

    completed_runs = [r for r in runs if r.state == "finished"]
    if not completed_runs:
        print("No completed runs found")
        return None

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

    best_config = {k: v for k, v in best_run.config.items() if k in tuned_params}

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


def write_sweep_id_file(path: str, sweep_id: str, project: str, entity: str) -> None:
    data = {"sweep_id": sweep_id, "project": project, "entity": entity}
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def main():
    parser = argparse.ArgumentParser(description="Pure W&B sweep management")
    parser.add_argument("--init", action="store_true", help="Initialize a new sweep")
    parser.add_argument(
        "--agent", type=str, metavar="SWEEP_ID", help="Run agent for existing sweep"
    )
    parser.add_argument(
        "--analyze", type=str, metavar="SWEEP_ID", help="Analyze completed sweep"
    )
    parser.add_argument(
        "--config", type=str, help="Path to sweep config file (required for --init)"
    )
    parser.add_argument(
        "--save-sweep-id",
        type=str,
        dest="save_sweep_id",
        help="Path to save the sweep ID YAML",
    )
    parser.add_argument("--project", type=str, default="HPO", help="W&B project name")
    args = parser.parse_args()

    if not (args.init or args.agent or args.analyze):
        parser.error("Must specify one of: --init, --agent, or --analyze")

    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    try:
        api = wandb.Api()
        entity = api.default_entity
        if not entity:
            raise RuntimeError(
                "Could not determine wandb entity. Please login with 'wandb login'"
            )

        if args.init:
            if not args.config:
                parser.error("--init requires --config")
            print("🚀 Initializing sweep...")
            sweep_id = initialize_sweep(args.config, args.project)
            out_path = args.save_sweep_id or "current_sweep.yaml"
            write_sweep_id_file(out_path, sweep_id, args.project, entity)
            print(f"📝 Sweep ID: {sweep_id}")
            print(f"📝 Saved to: {out_path}")
            print(f"\n🤖 To run agents:")
            print(f"   wandb agent {entity}/{args.project}/{sweep_id}")
            print(f"   python run_sweep.py --agent {sweep_id} --project {args.project}")
            return

        if args.agent:
            sweep_id = args.agent
            target = f"{entity}/{args.project}/{sweep_id}"
            print(f"🤖 Starting W&B agent: {target}")

            cmd = ["wandb", "agent", target]
            proc = subprocess.run(cmd, env=os.environ.copy())
            print(f"Agent finished with exit code: {proc.returncode}")
            return

        if args.analyze:
            analyze_id = args.analyze
            print(f"📊 Analyzing sweep: {analyze_id}")
            save_best_config(analyze_id.split("/")[-1], args.project, entity)
            return

    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
