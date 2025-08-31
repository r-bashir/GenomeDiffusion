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

# Run the container
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_ROOT:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    $CONTAINER bash -c "cd /workspace && python -u tests/test_memory_allocation.py 2>&1 | tee memtest.log" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"