#!/bin/bash
#SBATCH -J GenDiff               # Job name
#SBATCH -t 01:00:00              # Time limit (HH:MM:SS)
#SBATCH -n 1                     # Number of tasks
#SBATCH --cpus-per-task=8        # Allocate CPU cores
#SBATCH --gpus=1                 # Request GPU
#SBATCH --reservation=devel      # Use development reservation
#SBATCH -o slurm_logs/%x-%j.out  # Standard output log
#SBATCH -e slurm_logs/%x-%j.err  # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

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

# Run the container with CUDA environment variables
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    $CONTAINER bash -c "cd /workspace && python train.py --config config.yaml 2>&1 | tee train_${SLURM_JOB_ID}.log" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
