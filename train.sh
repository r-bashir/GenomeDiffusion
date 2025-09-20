#!/bin/bash

# Paths and variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_ROOT=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

export PROJECT_ROOT

# WandB API Key
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start timer

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"

# Optional: To resume from checkpoint
CHECKPOINT_PATH="${1:-}"
RESUME_FLAG=""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    if [[ "$CHECKPOINT_PATH" == /* ]]; then
        MAPPED_CKPT="$CHECKPOINT_PATH"
    else
        REL_PATH="${CHECKPOINT_PATH#./}"
        MAPPED_CKPT="/workspace/$REL_PATH"
    fi
    RESUME_FLAG="--resume \"$MAPPED_CKPT\""
fi

# Run the container
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_ROOT:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    $CONTAINER bash -c "cd /workspace && python train.py --config config.yaml $RESUME_FLAG 2>&1 | tee train.log" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
