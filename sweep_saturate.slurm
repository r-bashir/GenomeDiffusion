#!/bin/bash
#SBATCH -J HPO_Agents              # Job name
#SBATCH -t 08:00:00                # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gpus=1                   # Request 1 GPU
#SBATCH --cpus-per-task=16         # 1 GPU –> Allocate 16 CPU tasks
#SBATCH -o slurm_logs/%x-%j.out    # Standard output log
#SBATCH -e slurm_logs/%x-%j.err    # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# Coordinated W&B Agents SLURM Script for GenomeDiffusion HPO (saturate one GPU)
# Usage: sbatch sweep_saturate.slurm [config_file] [num_agents] [project_name] [checkpoint_or_artifact] [weights|trainer]

# Arguments
CONFIG_FILE=${1:-"sweep.yaml"}
NUM_AGENTS=${2:-3}
HPO_NAME=${3:-"HPO"}
# Optional: checkpoint path or W&B artifact, and resume strategy
CHECKPOINT_PATH=${4:-}
RESUME_STRATEGY=${5:-}

echo "Config: $CONFIG_FILE | Agents: $NUM_AGENTS | Project: $HPO_NAME"

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Set environment variables
export PROJECT_ROOT=$PROJECT_DIR
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start timer

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Using GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of coordinated agents: $NUM_AGENTS"

# Set W&B cache dirs
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
mkdir -p $WANDB_CACHE_DIR $WANDB_CONFIG_DIR

echo "================================"
echo "🚀 Initializing sweep inside container..."
# Build optional flags for initialization
INIT_FLAGS=( )
if [ -n "$CHECKPOINT_PATH" ]; then
  INIT_FLAGS+=("--checkpoint" "$CHECKPOINT_PATH")
fi
if [ -n "$RESUME_STRATEGY" ]; then
  INIT_FLAGS+=("--resume-strategy" "$RESUME_STRATEGY")
fi

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
        --save-sweep-id 'sweep_${SLURM_JOB_ID}.yaml' \
        ${INIT_FLAGS[@]}"

if [ $? -ne 0 ]; then
    echo "❌ Sweep initialization failed!"
    exit 1
fi

# Extract sweep ID (run inside container to guarantee PyYAML is available)
SWEEP_ID=$(apptainer exec \
    --bind $PROJECT_DIR:/workspace \
    $CONTAINER python -c "import yaml; print(yaml.safe_load(open('/workspace/sweep_${SLURM_JOB_ID}.yaml'))['sweep_id'])")

echo " Starting $NUM_AGENTS coordinated W&B agents..."
echo "Sweep ID: $SWEEP_ID"
echo "Project: $HPO_NAME"
echo "================================"

# Array to store PIDs
declare -a PIDS

# Launch multiple agents on the same GPU
for i in $(seq 1 $NUM_AGENTS); do
    echo "🤖 Starting agent $i/$NUM_AGENTS..."
    LOG_FILE="agent_${i}_${SLURM_JOB_ID}.log"
    
    apptainer exec --nv \
        --bind $DATA_DIR:/data \
        --bind $PROJECT_DIR:/workspace \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        $CONTAINER bash -c "cd /workspace && \
        python run_sweep.py \
            --agent '$SWEEP_ID' \
            --project '$HPO_NAME'" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS[$i]=$PID
    echo "📝 Agent $i started with PID: $PID (log: $LOG_FILE)"
    sleep 2
done

echo "✅ All $NUM_AGENTS agents launched for sweep: $SWEEP_ID"
echo "🔍 Agents will coordinate through W&B to explore hyperparameters"
echo "⏳ Waiting for completion..."

# Wait for all agents
for i in $(seq 1 $NUM_AGENTS); do
    PID=${PIDS[$i]}
    echo "⏳ Waiting for agent $i (PID: $PID)..."
    wait $PID
    EXIT_CODE=$?
    echo "✅ Agent $i completed with exit code: $EXIT_CODE"
done

echo "🎉 All coordinated agents completed!"
echo "📊 Check W&B dashboard: https://wandb.ai/$WANDB_ENTITY/$HPO_NAME/sweeps/$SWEEP_ID"

# Final analysis
echo "📈 Running final analysis to save best configuration..."
apptainer exec --nv \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    $CONTAINER bash -c "cd /workspace && \
    python run_sweep.py \
        --analyze '$SWEEP_ID' \
        --project '$HPO_NAME'"

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS

echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"

