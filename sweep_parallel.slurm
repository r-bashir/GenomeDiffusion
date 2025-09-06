#!/bin/bash
#SBATCH -J HPO_Parallel           # Job name
#SBATCH -t 08:00:00               # Time limit
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --gpus=1                  # Request 1 GPU
#SBATCH --cpus-per-task=16        # 16 CPUs for parallel processing
#SBATCH -o slurm_logs/%x-%j.out   # Standard output log
#SBATCH -e slurm_logs/%x-%j.err   # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se

# Parallel W&B Sweep Script
# Usage: sbatch sweep_parallel.slurm [config_file] [num_sweeps] [runs_per_sweep]
# Example: sbatch sweep_parallel.slurm sweep.yaml 5 20

CONFIG_FILE=${1:-"sweep.yaml"}
NUM_SWEEPS=${2:-5}
RUNS_PER_SWEEP=${3:-20}
TOTAL_RUNS=$((NUM_SWEEPS * RUNS_PER_SWEEP))

echo "Config: $CONFIG_FILE | Sweeps: $NUM_SWEEPS | Runs/sweep: $RUNS_PER_SWEEP | Total: $TOTAL_RUNS"

mkdir -p slurm_logs

# Paths
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Environment
export WANDB_API_KEY="cd68c5a140d1346421e71ebad92df1921db1cc19"
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0

# Initialize sweeps and start agents
SWEEP_PIDS=()
echo "Starting $NUM_SWEEPS parallel sweep agents..."

for i in $(seq 1 $NUM_SWEEPS); do
    echo "Starting sweep $i/$NUM_SWEEPS..."
    
    # Initialize sweep and run agent in one step
    apptainer exec --nv \
        --bind $DATA_DIR:/data \
        --bind $PROJECT_DIR:/workspace \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        $CONTAINER bash -c "cd /workspace && \
        python run_sweep.py --init --config $CONFIG_FILE --project HPO && \
        SWEEP_ID=\$(python -c \"import yaml; f=open('current_sweep.yaml'); data=yaml.safe_load(f); f.close(); print(data.get('sweep_id', ''))\") && \
        python run_sweep.py --agent \$SWEEP_ID --count $RUNS_PER_SWEEP" \
        > sweep_${i}_${SLURM_JOB_ID}.log 2>&1 &
    
    SWEEP_PIDS+=($!)
    echo "Sweep $i started with PID: $!"
    sleep 5
done

echo "All sweep agents started. PIDs: ${SWEEP_PIDS[*]}"

# Wait for all sweep agents to complete
echo "Waiting for all sweep agents to finish..."
SWEEP_IDS=()
for i in $(seq 1 $NUM_SWEEPS); do
    wait ${SWEEP_PIDS[$((i-1))]}
    
    # Extract sweep ID from log file
    SWEEP_ID=$(grep "Sweep ID:" sweep_${i}_${SLURM_JOB_ID}.log | tail -1 | cut -d' ' -f3)
    if [[ -n "$SWEEP_ID" ]]; then
        SWEEP_IDS+=($SWEEP_ID)
        echo "Sweep $i completed: $SWEEP_ID"
    fi
done
echo "All sweep agents completed"

# Analyze all sweeps to find best hyperparameters
if [ ${#SWEEP_IDS[@]} -gt 0 ]; then
    echo "Analyzing ${#SWEEP_IDS[@]} sweeps to find best hyperparameters..."
    
    for i in "${!SWEEP_IDS[@]}"; do
        sweep_id=${SWEEP_IDS[$i]}
        echo "Analyzing sweep $((i+1)): $sweep_id"
        
        apptainer exec --nv \
            --bind $DATA_DIR:/data \
            --bind $PROJECT_DIR:/workspace \
            --env WANDB_API_KEY=$WANDB_API_KEY \
            $CONTAINER bash -c "cd /workspace && \
            python run_sweep.py --analyze $sweep_id" \
            >> analysis_${SLURM_JOB_ID}.log 2>&1
    done
    
    echo "Analysis complete. Check analysis_${SLURM_JOB_ID}.log for best hyperparameters from all sweeps."
    echo "Sweep IDs analyzed: ${SWEEP_IDS[*]}"
else
    echo "Warning: No sweep IDs found for analysis"
fi

echo "Job completed successfully!"
echo "Logs: sweep_*_${SLURM_JOB_ID}.log | Analysis: analysis_${SLURM_JOB_ID}.log"
