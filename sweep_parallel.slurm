#!/bin/bash
#SBATCH -J GenHPO                    # Job name
#SBATCH -t 01:00:00                  # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                   # Single Slurm task; we manage agents ourselves
#SBATCH --gpus=8                     # Request 8 GPUs on this node
#SBATCH --cpus-per-task=64           # CPU cores for the whole job (e.g., 8 per GPU)
#SBATCH -o slurm_logs/%x-%j.out      # Stdout
#SBATCH -e slurm_logs/%x-%j.err      # Stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# W&B Sweep Script (Multi-GPU, coordinated agents on one node)
# Submit to SLURM
#   sbatch sweep_multigpu.slurm [config_file] [project_name] [checkpoint_path] [resume_strategy]

# Arguments
CONFIG_FILE=${1:-"sweep.yaml"}
HPO_NAME=${2:-"HPO"}
CHECKPOINT_PATH=${3:-}
RESUME_STRATEGY=${4:-}

# Create logs dir
mkdir -p slurm_logs

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Environment
export PROJECT_ROOT=$PROJECT_DIR
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export WANDB_MODE=online

# W&B cache/config for cluster stability
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
mkdir -p $WANDB_CACHE_DIR $WANDB_CONFIG_DIR

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0

# Fallback for interactive runs (no SLURM_JOB_ID)
JOB_ID=${SLURM_JOB_ID:-$$}

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | tr '\n' ', ' | sed 's/, $//')"

# Determine number of agents solely from Slurm allocation
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching $NUM_GPUS agents (1 per GPU)"

echo "🚀 Starting W&B sweep initialization..."

# Step 1: Initialize Sweep
echo "📋 1. Initializing W&B Sweep..."
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

# Extract Sweep ID from Saved File (inside container to ensure PyYAML is available)
SWEEP_ID=$(apptainer exec \
    --bind $PROJECT_DIR:/workspace \
    $CONTAINER python -c "import yaml; print(yaml.safe_load(open('/workspace/sweep_${JOB_ID}.yaml'))['sweep_id'])")
echo "📝 Sweep ID: $SWEEP_ID"

echo "🤖 2. Starting $NUM_GPUS coordinated W&B agents (1 per GPU) on this node..."
echo "Project: $HPO_NAME"

# Launch one agent per GPU
declare -a PIDS
SUCCESS_COUNT=0
FAIL_COUNT=0

# Kill all background agents on interruption
cleanup() {
  echo "\n🛑 Caught interruption. Terminating all agent processes..."
  for PID in "${PIDS[@]}"; do
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM
for (( i=0; i<NUM_GPUS; i++ )); do
    echo "🤖 Starting agent on GPU $i/$((NUM_GPUS-1))..."
    LOG_FILE="agent_gpu${i}_${SLURM_JOB_ID}.log"

    CMD_AGENT="python run_sweep.py --agent '$SWEEP_ID' --project '$HPO_NAME'"
    echo "Agent[$i] command: $CMD_AGENT"

    apptainer exec --nv \
        --bind $DATA_DIR:/data \
        --bind $PROJECT_DIR:/workspace \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env CUDA_VISIBLE_DEVICES=$i \
        $CONTAINER bash -c "cd /workspace && $CMD_AGENT" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    PIDS[$i]=$PID
    echo "📝 Agent on GPU $i started with PID: $PID (log: $LOG_FILE)"
    sleep 1

done

echo "✅ Launched $NUM_GPUS agents for sweep: $SWEEP_ID"
echo "🔍 Agents coordinate via W&B to explore hyperparameters"
echo "⏳ Waiting for completion..."

# Wait for all agents
for (( i=0; i<NUM_GPUS; i++ )); do
    PID=${PIDS[$i]}
    echo "⏳ Waiting for agent on GPU $i (PID: $PID)..."
    wait $PID
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 0 ]; then
      echo "✅ Agent on GPU $i completed successfully."
      SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
      echo "❌ Agent on GPU $i failed with exit code: $EXIT_CODE"
      FAIL_COUNT=$((FAIL_COUNT+1))
    fi

done

# Dashboard link (WANDB_ENTITY may be empty if not configured)
echo "📊 Check W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$HPO_NAME/sweeps/$SWEEP_ID"
echo "📋 Agents summary: success=$SUCCESS_COUNT, failed=$FAIL_COUNT"

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

# End time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
