#!/bin/bash
#SBATCH -J HPO_Parallel            # Job name
#SBATCH --array=1-4                # Launch 4 array jobs (agents)
#SBATCH -t 08:00:00                # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                 # One task per array job
#SBATCH --gpus=1                   # Request 1 GPU per node/task
#SBATCH --cpus-per-task=16         # CPU cores per task
#SBATCH -o slurm_logs/%x-%A_%a.out # Stdout (%A=array job ID, %a=array task ID)
#SBATCH -e slurm_logs/%x-%A_%a.err # Stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# Slurm script to run W&B Sweep with multiple agents on multiple GPUs (parallel)
# Usage: sbatch sweep_parallel.slurm [config_file] [project_name] [checkpoint_path] [resume_strategy]

# Arguments
CONFIG_FILE=${1:-"sweep.yaml"}
HPO_NAME=${2:-"HPO"}
CHECKPOINT_PATH=${3:-}
RESUME_STRATEGY=${4:-}

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Environment
export PROJECT_ROOT=$PROJECT_DIR
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export WANDB_MODE=online
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

mkdir -p slurm_logs $WANDB_CACHE_DIR $WANDB_CONFIG_DIR

# Sweep initialization only by the first array job
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "🚀 Initializing sweep..."
    # Build optional flags as safely escaped string
    INIT_FLAGS_STR=""
    if [ -n "$CHECKPOINT_PATH" ]; then
        printf -v CKPT_ESC '%q' "$CHECKPOINT_PATH"
        INIT_FLAGS_STR+=" --checkpoint ${CKPT_ESC}"
    fi
    if [ -n "$RESUME_STRATEGY" ]; then
        printf -v RES_ESC '%q' "$RESUME_STRATEGY"
        INIT_FLAGS_STR+=" --resume-strategy ${RES_ESC}"
    fi

    CMD_INIT="python run_sweep.py --init --config '$CONFIG_FILE' --project '$HPO_NAME' --save-sweep-id 'sweep_${SLURM_JOB_ID}.yaml'${INIT_FLAGS_STR}"
    echo "Init command: $CMD_INIT"

    apptainer exec --nv \
        --bind $DATA_DIR:/data \
        --bind $PROJECT_DIR:/workspace \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        $CONTAINER bash -c "cd /workspace && $CMD_INIT"
    if [ $? -ne 0 ]; then
        echo "❌ Sweep initialization failed!"
        exit 1
    fi
fi

# Wait for the first job to finish sweep initialization
srun --ntasks=1 --exclusive sleep 5  # small delay; adjust if needed

# Extract sweep ID (inside container)
SWEEP_ID=$(apptainer exec \
    --bind $PROJECT_DIR:/workspace \
    $CONTAINER python -c "import yaml; print(yaml.safe_load(open('/workspace/sweep_${SLURM_JOB_ID}.yaml'))['sweep_id'])")

echo "🤖 Starting W&B agent for array task $SLURM_ARRAY_TASK_ID"
LOG_FILE="agent_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.log"

apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && \
    python run_sweep.py \
        --agent '$SWEEP_ID' \
        --project '$HPO_NAME'" \
    > "$LOG_FILE" 2>&1

echo "✅ Agent $SLURM_ARRAY_TASK_ID completed. Log: $LOG_FILE"

