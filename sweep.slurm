#!/bin/bash
#SBATCH -J GenHPO                # Job name
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
#   sbatch sweep.slurm [config_file] [hpo_name] [checkpoint_path] [resume_strategy]
# Run directly in container
#     bash sweep.slurm [config_file] [hpo_name] [checkpoint_path] [resume_strategy]

# Set defaults
CONFIG_FILE=${1:-"sweep_unet.yaml"}
HPO_NAME=${2:-"HPO"}
CHECKPOINT_PATH=${3:-}
RESUME_STRATEGY=${4:-}

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

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Fallback for interactive runs (no SLURM_JOB_ID)
JOB_ID=${SLURM_JOB_ID:-$$}

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | tr '\n' ', ' | sed 's/, $//')"

# W&B cache/config for cluster stability
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "🚀 Starting W&B sweep initialization..."

# Step 1: Initialize Sweep
echo "📋 1. Initializing W&B Sweep..."

# Build optional flags
INIT_FLAGS_STR=""
if [ -n "$CHECKPOINT_PATH" ]; then
  printf -v CKPT_ESC '%q' "$CHECKPOINT_PATH"
  INIT_FLAGS_STR+=" --checkpoint ${CKPT_ESC}"
fi
if [ -n "$RESUME_STRATEGY" ]; then
  printf -v RES_ESC '%q' "$RESUME_STRATEGY"
  INIT_FLAGS_STR+=" --resume-strategy ${RES_ESC}"
fi

CMD_INIT="python run_sweep.py --init --config '$CONFIG_FILE' --project '$HPO_NAME' --save-sweep-id 'sweep_${JOB_ID}.yaml'${INIT_FLAGS_STR}"
echo "Init command: $CMD_INIT"

apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && $CMD_INIT" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

if [ $? -ne 0 ]; then
    echo "❌ Sweep initialization failed!"
    exit 1
fi

# Extract Sweep ID from Saved File
SWEEP_ID=$(apptainer exec \
    --bind $PROJECT_DIR:/workspace \
    $CONTAINER python -c "import yaml; print(yaml.safe_load(open('/workspace/sweep_${JOB_ID}.yaml'))['sweep_id'])")
echo "📝 Sweep ID: $SWEEP_ID"

# Step 2: Run Sweep Agent
echo "🤖 2. Starting W&B Agent..."
CMD_AGENT="python run_sweep.py --agent '$SWEEP_ID' --project '$HPO_NAME'"
echo "Agent command: $CMD_AGENT"

set +e
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && $CMD_AGENT" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

AGENT_EXIT=$?
set -e 2>/dev/null || true

echo "✅ Agent finished with exit code: $AGENT_EXIT"
echo "📊 Check W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$HPO_NAME/sweeps/$SWEEP_ID"

# Step 3: Final Analysis
echo "📈 3. Running final analysis..."
CMD_ANA="python run_sweep.py --analyze '$SWEEP_ID' --project '$HPO_NAME'"
echo "Analyze command: $CMD_ANA"

apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && $CMD_ANA" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
