#!/bin/bash
#SBATCH -J AnalyzeSweep              # Job name
#SBATCH -t 00:20:00                  # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                   # One task
#SBATCH --cpus-per-task=4            # Few CPUs are enough
#SBATCH -o slurm_logs/%x-%j.out      # Stdout
#SBATCH -e slurm_logs/%x-%j.err      # Stderr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# W&B Sweep Analysis Only (no agents, no training)
# Usage:
#   sbatch analyze.slurm <sweep_id_or_file> <project_name>
# Examples:
#   sbatch analyze.slurm abc123 HPO
#   sbatch analyze.slurm sweep_14176627.yaml HPO

# Arguments
SWEEP_REF=$1         # Either a sweep ID (e.g., abc123) or a path to sweep_*.yaml
HPO_NAME=$2          # W&B project name (e.g., HPO)

if [ -z "$SWEEP_REF" ] || [ -z "$HPO_NAME" ]; then
  echo "Usage: sbatch analyze.slurm <sweep_id_or_file> <project_name>" >&2
  exit 1
fi

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Environment
export PROJECT_ROOT=$PROJECT_DIR
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export WANDB_MODE=online

# W&B cache/config
export WANDB_CACHE_DIR=$PROJECT_ROOT/.wandb_cache
export WANDB_CONFIG_DIR=$PROJECT_ROOT/.wandb_config
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" slurm_logs

# Determine sweep id
if [ -f "$SWEEP_REF" ]; then
  # If a file path is given, read the sweep id from it (inside the container to ensure PyYAML)
  echo "🔎 Reading sweep id from file: $SWEEP_REF"
  SWEEP_ID=$(apptainer exec \
    --bind $PROJECT_DIR:/workspace \
    $CONTAINER python -c "import yaml, os; p=os.path.join('/workspace', os.path.basename('$SWEEP_REF')); print(yaml.safe_load(open(p))['sweep_id'])")
else
  SWEEP_ID="$SWEEP_REF"
fi

if [ -z "$SWEEP_ID" ]; then
  echo "❌ Could not resolve sweep id from: $SWEEP_REF" >&2
  exit 1
fi

# Log start
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
JOB_ID=${SLURM_JOB_ID:-$$}
ANALYZE_LOG="analyze_${JOB_ID}.log"

echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "📝 Sweep ID: $SWEEP_ID | Project: $HPO_NAME"
echo "📈 Running analysis only (no agents)..."

# Build analyze command
CMD_ANA="python run_sweep.py --analyze '$SWEEP_ID' --project '$HPO_NAME'"
echo "Analyze command: $CMD_ANA"

# Run analysis (no GPU required). Disable W&B console capture and redirect output to a stable log file.
set +e
apptainer exec \
    --bind $DATA_DIR:/data \
    --bind $PROJECT_DIR:/workspace \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_CONSOLE=off \
    $CONTAINER bash -c "cd /workspace && $CMD_ANA" \
    > "$ANALYZE_LOG" 2>&1
ANA_EXIT=$?

if [ "$ANA_EXIT" -ne 0 ]; then
  echo "⚠️ First analyze attempt failed with exit code $ANA_EXIT. Retrying once... (see $ANALYZE_LOG)"
  sleep 5
  apptainer exec \
      --bind $DATA_DIR:/data \
      --bind $PROJECT_DIR:/workspace \
      --env WANDB_API_KEY=$WANDB_API_KEY \
      --env WANDB_CONSOLE=off \
      $CONTAINER bash -c "cd /workspace && $CMD_ANA" \
      >> "$ANALYZE_LOG" 2>&1
  ANA_EXIT=$?
fi
set -e 2>/dev/null || true

if [ "$ANA_EXIT" -ne 0 ]; then
  echo "❌ Analyze failed. See $ANALYZE_LOG for details."
  echo "You can re-run manually inside the container:"
  echo "  apptainer exec --bind $DATA_DIR:/data --bind $PROJECT_DIR:/workspace $CONTAINER \"bash -c 'cd /workspace && $CMD_ANA'\""
  exit 1
fi

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "✅ Analysis completed successfully. Best config saved as best_config_<sweep_id>.yaml in /workspace"
echo "Job $SLURM_JOB_ID finished at $END_TIME"
