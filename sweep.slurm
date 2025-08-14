#!/bin/bash
#SBATCH -J GenDiff-HPO           # Job name
#SBATCH -t 24:00:00              # Time limit (HH:MM:SS)
#SBATCH -n 1                     # Number of tasks
#SBATCH --cpus-per-task=8        # Allocate CPU cores
#SBATCH --gpus=1                 # Request GPU
#SBATCH -o slurm_logs/%x-%j.out  # Standard output log
#SBATCH -e slurm_logs/%x-%j.err  # Standard error log

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

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Paths and variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Set environment variables
export PROJECT_ROOT=$PROJECT_DIR
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start timer

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Sweep ID: $SWEEP_ID"
echo "Run Count: $COUNT"
# Set W&B settings for cluster
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config

# Run the container with CUDA environment variables
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    $CONTAINER bash -c "cd /workspace && python run_sweep.py --agent $SWEEP_ID --count $COUNT 2>&1 | tee sweep_${SLURM_JOB_ID}.log" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Create wandb directories
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
