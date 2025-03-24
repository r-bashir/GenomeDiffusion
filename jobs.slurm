#!/bin/bash
#S-BATCH -A berzelius-2025-96     # Account name
#SBATCH -J GenDiff               # Job name
#SBATCH -t 04:30:00              # Time limit (HH:MM:SS)
#S-BATCH --mem=2000               # Memory in MB
#SBATCH -n 1                     # Number of tasks
#SBATCH --cpus-per-task=4        # Allocate CPU cores
#SBATCH --gpus 1                 # Request 1 GPU, for type --gpus=a100:1
#SBATCH --partition=berzelius    # Ensure correct partition
#SBATCH --mail-type=ALL          # Send email on job begin, end, fail, etc.
#SBATCH --mail-user=rabia.bashir.9649@student.uu.se
#SBATCH -o logs/%x-%j.out        # Standard output log
#SBATCH -e logs/%x-%j.err        # Standard error log

# Ensure logs directory exists
mkdir -p logs

# Paths and variables
CONTAINER=/proj/gcae_berzelius/users/x_rabba/lightning_25.01-py3.sif
PROJECT_DIR=/proj/gcae_berzelius/users/x_rabba/GenomeDiffusion
DATA_DIR=/proj/gcae_berzelius/users/shared/HO_data

# Validate paths before execution
if [[ ! -f "$CONTAINER" ]]; then
    echo "Error: Container file $CONTAINER not found!" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Error: Project directory $PROJECT_DIR not found!" >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory $DATA_DIR not found!" >&2
    exit 1
fi

# Log Start Time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
SECONDS=0  # Start timer

echo ""
echo "Job $SLURM_JOB_ID started on $(hostname) at $START_TIME"
echo "Running with ${SLURM_MEM_PER_NODE} MB of memory and ${SLURM_CPUS_PER_TASK} CPUs"
echo ""

# Run inside Apptainer, ensuring data is mapped correctly
apptainer exec --nv -B $PROJECT_DIR:/workspace -B $DATA_DIR:/workspace/data \
$CONTAINER bash -c "cd /workspace && python train.py --config config.yaml" || {
    echo "Error: Apptainer execution failed!" >&2
    exit 1
}

# Log End Time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$SECONDS  # Get elapsed seconds

echo ""
echo "Job $SLURM_JOB_ID finished at $END_TIME"
echo "Total execution time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"

