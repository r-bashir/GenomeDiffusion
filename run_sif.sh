#!/bin/bash

# Paths
CONTAINER=lightning_25.01-py3.sif
PROJECT_DIR=$PWD

# WandB API Key
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"

apptainer exec --nv \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    $CONTAINER bash -c "cd /workspace && python train.py --config config.yaml" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}
