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
    python run_sweep.py --init --config <config_path> --project <project_name>

    # Run sweep agent (after initialization)
    python run_sweep.py --agent <sweep_id> --count <count>

    # Monitor sweep progress
    python run_sweep.py --monitor <sweep_id> --project <project_name>

    # Analyze sweep results
    python run_sweep.py --analyze <sweep_id> --project <project_name>
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import wandb
import yaml


def initialize_sweep(config_path: str, base_config_path: str = "config.yaml") -> str:
    """Initialize a new W&B sweep.

    Args:
        config_path: Path to sweep configuration file
        base_config_path: Path to base configuration file (for project name)

    Returns:
        Sweep ID
    """
    # Load base config to get project name
    from src.utils import load_config

    base_config = load_config(base_config_path)
    project = base_config.get("project_name", "GenDiffusion")

    print(f"🚀 Initializing W&B Sweep...")
    print(f"Sweep Config: {config_path}")
    print(f"Base Config: {base_config_path}")
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


def monitor_sweep(
    sweep_id: str, project: str = None, base_config_path: str = "config.yaml"
):
    """Monitor sweep progress.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)
        project: W&B project name (used as fallback, if None loads from config)
        base_config_path: Path to base configuration file (for project name)
    """
    # Load project name from config if not provided
    if project is None:
        from src.utils import load_config

        base_config = load_config(base_config_path)
        project = base_config.get("project_name", "GenDiffusion")
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


def analyze_sweep_results(
    sweep_id: str, project: str = None, base_config_path: str = "config.yaml"
):
    """Analyze sweep results and provide recommendations.

    Args:
        sweep_id: W&B sweep ID (can be just ID or full path)
        project: W&B project name (used as fallback, if None loads from config)
        base_config_path: Path to base configuration file (for project name)
    """
    # Load project name from config if not provided
    if project is None:
        from src.utils import load_config

        base_config = load_config(base_config_path)
        project = base_config.get("project_name", "GenDiffusion")
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

        # Sort by val_loss, handling mixed types
        def safe_sort_key(run):
            val = run.summary[val_loss_key]
            try:
                return float(val)
            except (ValueError, TypeError):
                return float("inf")  # Put invalid values at the end

        valid_runs.sort(key=safe_sort_key)

        # Top 5 runs
        print(f"\n🏆 Top 5 Runs:")
        print(f"{'Rank':<5} {'Run Name':<20} {'Val Loss':<12} {'Key Params'}")
        print("-" * 80)

        for i, run in enumerate(valid_runs[:5]):
            val_loss = run.summary[val_loss_key]

            # Extract key parameters (try both simple and nested config keys)
            config = run.config
            lr = config.get("learning_rate", config.get("optimizer.lr", "N/A"))
            wd = config.get("weight_decay", config.get("optimizer.weight_decay", "N/A"))
            sched = config.get("scheduler_type", config.get("scheduler.type", "N/A"))
            bs = config.get("batch_size", config.get("data.batch_size", "N/A"))
            epochs = config.get("epochs", config.get("training.epochs", "N/A"))

            key_params = f"lr={lr}, wd={wd}, sched={sched}, bs={bs}, epochs={epochs}"

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

        # Weight decay analysis
        wd_values = []
        for r in valid_runs:
            wd = r.config.get("weight_decay") or r.config.get("optimizer.weight_decay")
            if wd:
                wd_values.append(float(wd))

        if wd_values:
            best_wd = wd_values[0]  # First in sorted list
            print(f"   Best Weight Decay: {best_wd:.1e}")
            print(f"   WD Range: {min(wd_values):.1e} - {max(wd_values):.1e}")

        # Scheduler analysis
        sched_types = []
        for r in valid_runs:
            sched = r.config.get("scheduler_type") or r.config.get("scheduler.type")
            if sched:
                sched_types.append(sched)

        if sched_types:
            best_sched = sched_types[0]  # First in sorted list
            print(f"   Best Scheduler: {best_sched}")
            unique_scheds = list(set(sched_types))
            print(f"   Schedulers tested: {unique_scheds}")

        # Batch size analysis
        batch_sizes = []
        for r in valid_runs:
            bs = r.config.get("batch_size") or r.config.get("data.batch_size")
            if bs:
                batch_sizes.append(int(bs))

        if batch_sizes:
            best_bs = batch_sizes[0]  # First in sorted list
            print(f"   Best Batch Size: {best_bs}")
            print(f"   Batch Size Range: {min(batch_sizes)} - {max(batch_sizes)}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        best_run = valid_runs[0]
        best_config = best_run.config

        best_lr = best_config.get("learning_rate") or best_config.get(
            "optimizer.lr", "N/A"
        )
        best_wd = best_config.get("weight_decay") or best_config.get(
            "optimizer.weight_decay", "N/A"
        )
        best_sched = best_config.get("scheduler_type") or best_config.get(
            "scheduler.type", "N/A"
        )
        best_bs = best_config.get("batch_size") or best_config.get(
            "data.batch_size", "N/A"
        )
        best_epochs = best_config.get("epochs") or best_config.get(
            "training.epochs", "N/A"
        )

        print(
            f"   1. Use learning rate around {float(best_lr):.1e}"
            if best_lr != "N/A"
            else "   1. Learning rate: N/A"
        )
        print(
            f"   2. Use weight decay around {float(best_wd):.1e}"
            if best_wd != "N/A"
            else "   2. Weight decay: N/A"
        )
        print(f"   3. Use scheduler type: {best_sched}")
        print(f"   4. Use batch size of {best_bs}")
        print(f"   5. Use epochs: {best_epochs}")

        scheduler_type = best_sched
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
        default="sweep.yaml",
        help="Path to sweep configuration file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of runs for this agent (overrides config)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (if not provided, loads from config.yaml)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="config.yaml",
        help="Path to base configuration file (for project name)",
    )

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
            sweep_id = initialize_sweep(
                args.config, getattr(args, "base_config", "config.yaml")
            )
            print(f"\n🎯 Next steps:")
            print(f"   1. Run agent: python run_sweep.py --agent {sweep_id}")
            print(f"   2. Monitor: python run_sweep.py --monitor {sweep_id}")

        elif args.agent:
            # Run sweep agent
            run_sweep_agent(args.agent, args.count)

        elif args.monitor:
            # Monitor sweep
            monitor_sweep(
                args.monitor, args.project, getattr(args, "base_config", "config.yaml")
            )

        elif args.analyze:
            # Analyze results
            analyze_sweep_results(
                args.analyze, args.project, getattr(args, "base_config", "config.yaml")
            )

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
