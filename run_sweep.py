#!/usr/bin/env python
# coding: utf-8

"""
Script to initialize and run W&B Sweeps for hyperparameter optimization.

This script provides utilities to:
1. Initialize a new sweep
2. Run sweep agents
3. Monitor sweep progress
4. Analyze sweep results

Usage:
    # Initialize a new sweep
    python run_sweep.py --init

    # Run sweep agent (after initialization)
    python run_sweep.py --agent <sweep_id>

    # Monitor sweep progress
    python run_sweep.py --monitor <sweep_id>
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

import wandb


def initialize_sweep(config_path: str, project: str) -> str:
    """Initialize a new W&B sweep.

    Args:
        config_path: Path to sweep configuration file
        project: W&B project name

    Returns:
        Sweep ID
    """
    print(f"🚀 Initializing W&B Sweep...")
    print(f"Config: {config_path}")
    print(f"Project: {project}")

    # Check if config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Sweep config file not found: {config_path}")

    # Load and validate config
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    print(f"Sweep method: {sweep_config.get('method', 'unknown')}")
    print(f"Metric: {sweep_config.get('metric', {}).get('name', 'unknown')}")
    print(f"Parameters to optimize: {len(sweep_config.get('parameters', {}))}")

    # Initialize sweep
    try:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"✅ Sweep initialized successfully!")
        print(f"Sweep ID: {sweep_id}")
        print(
            f"Sweep URL: https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}"
        )

        # Save sweep ID for future reference
        sweep_info = {
            "sweep_id": sweep_id,
            "project": project,
            "config_path": config_path,
            "url": f"https://wandb.ai/{wandb.api.default_entity}/{project}/sweeps/{sweep_id}",
        }

        with open("current_sweep.yaml", "w") as f:
            yaml.dump(sweep_info, f, default_flow_style=False)

        print(f"📝 Sweep info saved to current_sweep.yaml")

        return sweep_id

    except Exception as e:
        print(f"❌ Failed to initialize sweep: {e}")
        sys.exit(1)


def resolve_sweep_id(sweep_id: str) -> tuple[str, str, str]:
    """Resolve sweep ID to full path and extract components.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)

    Returns:
        Tuple of (full_sweep_id, entity, project)
    """
    if not sweep_id:
        return None, None, None

    if "/" in sweep_id:
        # Already full path
        parts = sweep_id.split("/")
        if len(parts) >= 3:
            entity = parts[0]
            project = parts[1]
            return sweep_id, entity, project
        else:
            return sweep_id, None, None

    # Try to get full path from current_sweep.yaml
    try:
        with open("current_sweep.yaml", "r") as f:
            sweep_info = yaml.safe_load(f)
            if sweep_info and sweep_info.get("sweep_id") == sweep_id:
                # Extract entity/project from URL
                url = sweep_info.get("url", "")
                if "wandb.ai/" in url:
                    path_parts = url.split("wandb.ai/")[1].split("/")
                    if len(path_parts) >= 3:
                        entity = path_parts[0]
                        project = path_parts[1]
                        full_sweep_id = f"{entity}/{project}/{sweep_id}"
                        return full_sweep_id, entity, project
    except FileNotFoundError:
        print("Warning: current_sweep.yaml not found")
    except Exception as e:
        print(f"Warning: Could not read sweep info: {e}")

    return sweep_id, None, None


def run_sweep_agent(sweep_id: str, count: int = None):
    """Run a sweep agent.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)
        count: Number of runs for this agent
    """
    print(f"🤖 Starting W&B Sweep Agent...")

    # Resolve sweep ID to full path
    full_sweep_id, entity, project = resolve_sweep_id(sweep_id)
    if entity and project:
        print(f"Found full sweep path: {full_sweep_id}")
        sweep_id = full_sweep_id

    print(f"Sweep ID: {sweep_id}")
    if count:
        print(f"Run count: {count}")

    # Build command
    cmd = ["wandb", "agent"]
    if count:
        cmd.extend(["--count", str(count)])
    cmd.append(sweep_id)

    print(f"Command: {' '.join(cmd)}")
    print("🏃 Starting agent (Ctrl+C to stop)...")

    try:
        # Run the agent
        subprocess.run(cmd, check=True)
        print("✅ Agent completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Agent stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Agent failed with exit code {e.returncode}")
        sys.exit(1)


def monitor_sweep(sweep_id: str, project: str):
    """Monitor sweep progress.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)
        project: W&B project name (used as fallback)
    """
    print(f"📊 Monitoring Sweep Progress...")

    # Resolve sweep ID to full path
    full_sweep_id, entity, resolved_project = resolve_sweep_id(sweep_id)
    if resolved_project:
        project = resolved_project
        sweep_id = full_sweep_id

    print(f"Sweep ID: {sweep_id}")
    print(f"Project: {project}")

    try:
        # Initialize API
        api = wandb.Api()
        # Handle full sweep path vs project/sweep_id format
        if "/" in sweep_id and len(sweep_id.split("/")) >= 3:
            sweep = api.sweep(sweep_id)
        else:
            sweep = api.sweep(f"{project}/{sweep_id}")

        print(f"Sweep URL: {sweep.url}")
        print(f"Sweep State: {sweep.state}")
        print(f"Best Run: {sweep.best_run()}")

        # Monitor runs
        print("\n📈 Recent Runs:")
        runs = list(sweep.runs)

        if not runs:
            print("No runs found yet...")
            return

        # Sort by creation time
        runs.sort(key=lambda x: x.created_at, reverse=True)

        header = (
            str("Run Name").ljust(20)
            + " "
            + str("State").ljust(12)
            + " "
            + str("Val Loss").ljust(12)
            + " "
            + str("Duration").ljust(10)
        )
        print(header)
        print("-" * len(header))

        # Try multiple possible val loss keys similar to analyze_sweep_results
        candidate_keys = ["val_loss", "val_loss_epoch", "final/val_loss"]

        for run in runs[:10]:  # Show last 10 runs
            # Pick first available key
            detected_key = next((k for k in candidate_keys if k in run.summary), None)
            raw_val = run.summary.get(detected_key, "N/A") if detected_key else "N/A"

            if isinstance(raw_val, (float, int)):
                val_loss_str = f"{float(raw_val):.6f}"
            else:
                val_loss_str = "N/A"

            duration = "N/A"
            if run.state == "finished" and hasattr(run, "_attrs"):
                if "runtime" in run._attrs:
                    duration = f"{run._attrs['runtime']//60:.0f}m"

            row = (
                str(run.name)[:20].ljust(20)
                + " "
                + str(run.state)[:12].ljust(12)
                + " "
                + str(val_loss_str)[:12].ljust(12)
                + " "
                + str(duration)[:10].ljust(10)
            )
            print(row)

        # Show best results
        if sweep.best_run():
            best = sweep.best_run()
            # Determine best val metric using same candidate keys
            best_key = next((k for k in candidate_keys if k in best.summary), None)
            best_raw = best.summary.get(best_key, "N/A") if best_key else "N/A"
            best_val = (
                f"{float(best_raw):.6f}"
                if isinstance(best_raw, (float, int))
                else "N/A"
            )
            print(f"\n🏆 Best Run: {best.name}")
            print(f"   Val Loss: {best_val} (key: {best_key or 'N/A'})")
            print(f"   Config: {dict(best.config)}")

    except Exception as e:
        print(f"❌ Failed to monitor sweep: {e}")


def analyze_sweep_results(sweep_id: str, project: str):
    """Analyze sweep results and provide recommendations.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)
        project: W&B project name (used as fallback)
    """
    print(f"🔍 Analyzing Sweep Results...")

    # Resolve sweep ID to full path
    full_sweep_id, entity, resolved_project = resolve_sweep_id(sweep_id)
    if resolved_project:
        project = resolved_project
        sweep_id = full_sweep_id

    try:
        api = wandb.Api()
        # Handle full sweep path vs project/sweep_id format
        if "/" in sweep_id and len(sweep_id.split("/")) >= 3:
            sweep = api.sweep(sweep_id)
        else:
            sweep = api.sweep(f"{project}/{sweep_id}")
        runs = list(sweep.runs)

        if not runs:
            print("No runs to analyze yet...")
            return

        # Filter completed runs
        completed_runs = [r for r in runs if r.state == "finished"]

        if not completed_runs:
            print("No completed runs to analyze yet...")
            return

        print(f"📊 Analysis of {len(completed_runs)} completed runs:")

        # Sort by val_loss (try different metric names)
        val_loss_keys = ["val_loss", "val_loss_epoch", "final/val_loss"]
        valid_runs = []
        val_loss_key = None

        for key in val_loss_keys:
            valid_runs = [r for r in completed_runs if key in r.summary]
            if valid_runs:
                val_loss_key = key
                break

        if not valid_runs:
            print("No runs with valid val_loss found...")
            print("Available metrics in first run:")
            if completed_runs:
                print(list(completed_runs[0].summary.keys()))
            return

        valid_runs.sort(key=lambda x: x.summary[val_loss_key])

        # Top 5 runs
        print(f"\n🏆 Top 5 Runs:")
        print(f"{'Rank':<5} {'Run Name':<20} {'Val Loss':<12} {'Key Params'}")
        print("-" * 80)

        for i, run in enumerate(valid_runs[:5]):
            val_loss = run.summary[val_loss_key]

            # Extract key parameters (try both simple and nested config keys)
            config = run.config
            lr = config.get("learning_rate", config.get("optimizer.lr", "N/A"))
            emb = config.get("embedding_dim", config.get("unet.embedding_dim", "N/A"))
            bs = config.get("batch_size", config.get("data.batch_size", "N/A"))

            key_params = f"lr={lr}, emb={emb}, bs={bs}"

            print(f"{i+1:<5} {run.name:<20} {val_loss:<12.6f} {key_params}")

        # Parameter analysis
        print(f"\n📈 Parameter Analysis:")

        # Learning rate analysis
        lr_values = []
        for r in valid_runs:
            lr = r.config.get("learning_rate") or r.config.get("optimizer.lr")
            if lr:
                lr_values.append(float(lr))

        if lr_values:
            best_lr = lr_values[0]  # First in sorted list
            print(f"   Best Learning Rate: {best_lr:.1e}")
            print(f"   LR Range: {min(lr_values):.1e} - {max(lr_values):.1e}")

        # Model size analysis
        emb_dims = []
        for r in valid_runs:
            emb = r.config.get("embedding_dim") or r.config.get("unet.embedding_dim")
            if emb:
                emb_dims.append(int(emb))

        if emb_dims:
            best_emb = emb_dims[0]  # First in sorted list
            print(f"   Best Embedding Dim: {best_emb}")
            print(f"   Embedding Range: {min(emb_dims)} - {max(emb_dims)}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        best_run = valid_runs[0]
        best_config = best_run.config

        best_lr = best_config.get("learning_rate") or best_config.get(
            "optimizer.lr", "N/A"
        )
        best_emb = best_config.get("embedding_dim") or best_config.get(
            "unet.embedding_dim", "N/A"
        )
        best_bs = best_config.get("batch_size") or best_config.get(
            "data.batch_size", "N/A"
        )

        print(
            f"   1. Use learning rate around {float(best_lr):.1e}"
            if best_lr != "N/A"
            else "   1. Learning rate: N/A"
        )
        print(f"   2. Use embedding dimension of {best_emb}")
        print(f"   3. Use batch size of {best_bs}")

        scheduler_type = best_config.get("scheduler_type") or best_config.get(
            "scheduler.type", "N/A"
        )
        print(f"   4. Use {scheduler_type} scheduler")

        # Save analysis to file
        analysis_file = f"sweep_analysis_{sweep_id.split('/')[-1]}.txt"
        with open(analysis_file, "w") as f:
            f.write(f"W&B Sweep Analysis - {sweep_id}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Best Run: {valid_runs[0].name}\n")
            f.write(f"Best Val Loss: {valid_runs[0].summary[val_loss_key]:.6f}\n\n")
            f.write(f"Recommended Parameters:\n")
            f.write(f"- Learning Rate: {best_lr}\n")
            f.write(f"- Batch Size: {best_bs}\n")
            f.write(f"- Scheduler: {scheduler_type}\n")

        print(f"\n💾 Analysis saved to: {analysis_file}")

    except Exception as e:
        print(f"❌ Failed to analyze sweep: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="W&B Sweep Management")

    # Mutually exclusive group for different actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true", help="Initialize a new sweep")
    group.add_argument("--agent", type=str, help="Run sweep agent with given sweep ID")
    group.add_argument("--monitor", type=str, help="Monitor sweep progress")
    group.add_argument("--analyze", type=str, help="Analyze sweep results")

    parser.add_argument(
        "--config",
        type=str,
        default="sweep_config.yaml",
        help="Path to sweep configuration file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (overrides config)",
    )
    parser.add_argument("--project", type=str, default="HPO", help="W&B project name")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set up environment
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    try:
        if args.init:
            # Initialize new sweep
            sweep_id = initialize_sweep(args.config, args.project)
            print(f"\n🎯 Next steps:")
            print(f"   1. Run agent: python run_sweep.py --agent {sweep_id}")
            print(f"   2. Monitor: python run_sweep.py --monitor {sweep_id}")

        elif args.agent:
            # Run sweep agent
            run_sweep_agent(args.agent, args.count)

        elif args.monitor:
            # Monitor sweep
            monitor_sweep(args.monitor, args.project)

        elif args.analyze:
            # Analyze results
            analyze_sweep_results(args.analyze, args.project)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
