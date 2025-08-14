#!/bin/bash
#SBATCH -J HPO                   # Job name
#SBATCH -t 24:00:00              # Time limit (HH:MM:SS)
#SBATCH -n 1                     # Number of tasks
#SBATCH --cpus-per-task=8        # Allocate CPU cores
#SBATCH --gpus=1                 # Request GPU
#SBATCH -o slurm_logs/%x-%j.out  # Standard output log
#SBATCH -e slurm_logs/%x-%j.err  # Standard error log

# W&B Sweep SLURM Script for GenomeDiffusion HPO
# Usage: sbatch sweep.slurm [config_file] [count]

# Set default config file if not provided
CONFIG_FILE=${1:-"sweep_small.yaml"}
COUNT=${2:-10}  # Default to 10 runs if not specified

echo "Using sweep config: $CONFIG_FILE"
echo "Agent run count: $COUNT"

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
echo "Run Count: $COUNT"

# Set W&B settings for cluster
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config

# Create wandb directories
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR

# Run the container with CUDA environment variables
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    $CONTAINER bash -c "cd /workspace && \
    echo 'Initializing sweep with config $CONFIG_FILE...' && \
    python run_sweep.py --init --config $CONFIG_FILE --project HPO && \
    echo 'Parsing sweep ID from current_sweep.yaml...' && \
    SWEEP_ID=\$(python - <<'PY' \
    import yaml \
    with open('current_sweep.yaml') as f: \
        info = yaml.safe_load(f) or {} \
    print(info.get('sweep_id','')) \
    PY\
    ) && \
    if [[ -z \"\$SWEEP_ID\" ]]; then echo '[ERROR] Could not read sweep_id from current_sweep.yaml' >&2; exit 2; fi && \
    echo \"Sweep ID: \$SWEEP_ID\" && \
    echo 'Starting sweep agent...' && \
    python run_sweep.py --agent \"\$SWEEP_ID\" --count $COUNT 2>&1 | tee sweep_${SLURM_JOB_ID}.log && \
    echo 'Sweep agent completed. Analyzing best parameters...' && \
    python run_sweep.py --analyze \"\$SWEEP_ID\" 2>&1 | tee -a sweep_${SLURM_JOB_ID}.log \
    " || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
