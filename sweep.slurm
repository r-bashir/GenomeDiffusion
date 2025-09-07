#!/bin/bash
#SBATCH -J HPO                   # Job name
#SBATCH -t 08:00:00              # Time limit (HH:MM:SS)
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --cpus-per-task=16       # 1 GPU –> Allocate 16 CPU tasks
#SBATCH -o slurm_logs/%x-%j.out  # Standard output log
#SBATCH -e slurm_logs/%x-%j.err  # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# Dual-purpose W&B Sweep Script for HPO
# Submit to SLURM
#   sbatch sweep.slurm [config_file] [hpo_name] [max_jobs]
# Run directly in container
#     bash sweep.slurm [config_file] [hpo_name] [max_jobs]

# Set defaults
CONFIG_FILE=${1:-"sweep.yaml"}
HPO_NAME=${2:-"HPO"}
MAX_JOBS=${3:-""}

echo "Config: $CONFIG_FILE | Project: $HPO_NAME"
if [ -n "$MAX_JOBS" ]; then
  echo "Agent run cap (max_jobs): $MAX_JOBS"
fi

# Create logs dir
mkdir -p slurm_logs

# Set environment variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

export PROJECT_ROOT=$PROJECT_DIR
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Fallback for interactive runs (no SLURM_JOB_ID)
JOB_ID=${SLURM_JOB_ID:-$$}

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Run Count: $COUNT"

# Set W&B settings for cluster
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config

# Optionally cap agent runs
if [ -n "$MAX_JOBS" ]; then
  export WANDB_AGENT_MAX_JOBS="$MAX_JOBS"
fi

# Create wandb directories
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR

echo "🚀 Starting W&B sweep initialization..."
echo "Config: $CONFIG_FILE"
echo "Project: $HPO_NAME"
echo "================================"

# Step 1: Initialize sweep
echo "📋 Initializing sweep..."
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && \
    python run_sweep.py \
        --init \
        --config '$CONFIG_FILE' \
        --project '$HPO_NAME' \
        --save-sweep-id 'sweep_${JOB_ID}.yaml'
        " || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

if [ $? -ne 0 ]; then
    echo "❌ Sweep initialization failed!"
    exit 1
fi

# Extract Sweep ID from Saved File
SWEEP_ID=$(python -c "import yaml; print(yaml.safe_load(open('sweep_${JOB_ID}.yaml'))['sweep_id'])")
echo "📝 Sweep ID: $SWEEP_ID"

# Step 2: Run Sweep Agent (let W&B manage naturally)
echo "🤖 Starting W&B agent..."
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_AGENT_MAX_JOBS=${WANDB_AGENT_MAX_JOBS:-} \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && \
    EXTRA=\"\"; \
    if [ -n \"$WANDB_AGENT_MAX_JOBS\" ]; then EXTRA=\"--max-jobs $WANDB_AGENT_MAX_JOBS\"; fi; \
    python run_sweep.py \
        --agent '$SWEEP_ID' \
        --project '$HPO_NAME' \
        $EXTRA \
        " || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

echo "✅ Agent execution completed!"
echo "📊 Check W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$HPO_NAME/sweeps/$SWEEP_ID"

# Step 3: Final Analysis (extract best configuration)
echo "📈 Running final analysis to save best configuration..."
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && \
    python run_sweep.py \
        --analyze '$SWEEP_ID' \
        --project '$HPO_NAME'
        " || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
