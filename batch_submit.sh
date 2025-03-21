#!/bin/bash
#SBATCH -A berzelius-2025-96
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/pytorch_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenomeDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Run inside Apptainer, ensuring "data/" is mapped correctly
apptainer exec --nv -B $PROJECT_DIR:/workspace -B $DATA_DIR:/workspace/data \
$CONTAINER bash -c "cd /workspace && python train.py --config config.yaml"

