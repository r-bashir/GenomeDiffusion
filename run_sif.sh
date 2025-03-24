#!/bin/bash

# Paths
# CONTAINER=/proj/gcae_berzelius/users/x_rabba/pytorch_25.01-py3.sif
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenomeDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Run inside Apptainer, ensuring "data/" is mapped correctly
apptainer exec --nv -B $PROJECT_DIR:/workspace -B $DATA_DIR:/workspace/data \
$CONTAINER bash -c "cd /workspace && python run_sif.py --config config.yaml"

