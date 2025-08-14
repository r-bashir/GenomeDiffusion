#!/bin/bash
#SBATCH --job-name=genomediff-hpo
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# W&B Sweep SLURM Script for GenomeDiffusion HPO
# Usage: sbatch sweep.slurm <sweep_id> [count]

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Error: Sweep ID required"
    echo "Usage: sbatch sweep.slurm <sweep_id> [count]"
    exit 1
fi

SWEEP_ID=$1
COUNT=${2:-10}  # Default to 10 runs if not specified

echo "Starting W&B Sweep Agent"
echo "Sweep ID: $SWEEP_ID"
echo "Run Count: $COUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create logs directory
mkdir -p logs

# Set up environment
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate virtual environment (adjust path as needed)
# source venv/bin/activate

# Set W&B settings for cluster
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config

# Create wandb directories
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR

# Run sweep agent
echo "Starting sweep agent at $(date)"
python run_sweep.py --agent $SWEEP_ID --count $COUNT

echo "Sweep agent completed at $(date)"
