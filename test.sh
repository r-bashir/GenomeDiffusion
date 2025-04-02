#!/bin/bash

# Default values
CONFIG="config.yaml"
CHECKPOINT=""

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

# Run evaluation
python test.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"

status=$?
if [ $status -ne 0 ]; then
    echo "Evaluation failed with status $status"
    exit $status
fi

echo "Evaluation completed. Results saved in: $(dirname $(dirname $CHECKPOINT))/evaluation/"
