#!/bin/bash

# Paths and variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Set default checkpoint path, but allow override from command line
DEFAULT_CHECKPOINT=checkpoints/last.ckpt
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}

# Print checkpoint being used
echo "Using checkpoint: $CHECKPOINT_DIR"

# WandB API Key
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start timer

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"

# Run the container
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    $CONTAINER bash -c "cd /workspace && python inference.py --checkpoint $CHECKPOINT_DIR" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
