#!/bin/bash

# Default values
CONFIG="config.yaml"
CHECKPOINT=""
NUM_SAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint argument is required"
    exit 1
fi

# Build command
CMD="python inference.py --config $CONFIG --checkpoint $CHECKPOINT"
if [ ! -z "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# Run inference
$CMD

status=$?
if [ $status -ne 0 ]; then
    echo "Inference failed with status $status"
    exit $status
fi

echo "Inference completed. Results saved in: $(dirname $(dirname $CHECKPOINT))/inference/"
