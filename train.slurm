#!/bin/bash
#SBATCH -J GenDiff               # Job name
#SBATCH -t 08:00:00              # Time limit (HH:MM:SS)
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --cpus-per-task=16       # 1 GPU –> Allocate 16 CPU tasks
#SBATCH -o slurm_logs/%x-%j.out  # Standard output log
#SBATCH -e slurm_logs/%x-%j.err  # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# Dual-purpose Training Script
# Submit to SLURM
#   sbatch train.slurm [config_file] [checkpoint_path] [resume_strategy]
# Run directly in container
#     bash train.slurm [config_file] [checkpoint_path] [resume_strategy]

# Create logs directory
mkdir -p slurm_logs

# Set environment variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

export PROJECT_ROOT=$PROJECT_DIR
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Fallback for interactive runs (no SLURM_JOB_ID)
JOB_ID=${SLURM_JOB_ID:-$$}

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

# W&B cache/config for cluster stability
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Arguments: [config_file] [checkpoint] [resume_strategy]
CONFIG_FILE=${1:-"config.yaml"}
RAW_CHECKPOINT=${2:-}
RESUME_STRATEGY=${3:-trainer}

# Optional: checkpoint mapping and safe flags
CHECKPOINT_FLAG=""
if [[ -n "$RAW_CHECKPOINT" ]]; then
    # Map checkpoint into container namespace
    if [[ "$RAW_CHECKPOINT" == /* ]]; then
        MAPPED_CKPT="$RAW_CHECKPOINT"
    else
        REL_PATH="${RAW_CHECKPOINT#./}"
        MAPPED_CKPT="/workspace/$REL_PATH"
    fi
    # Safely escape values
    printf -v CKPT_ESC '%q' "$MAPPED_CKPT"
    printf -v STRAT_ESC '%q' "$RESUME_STRATEGY"
    CHECKPOINT_FLAG=" --checkpoint ${CKPT_ESC} --resume-strategy ${STRAT_ESC}"
fi

# Run the container with CUDA environment variables
CMD_INIT="python train.py --config '$CONFIG_FILE'${CHECKPOINT_FLAG}"
echo "Training command: $CMD_INIT"
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    $CONTAINER bash -c "cd /workspace && $CMD_INIT 2>&1 | tee train_${SLURM_JOB_ID}.log" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
