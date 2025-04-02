#!/bin/bash

# Default values
CONFIG="config.yaml"
OUTPUT_DIR="runs/default"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python train.py \
  --config "$CONFIG" \
  --output_dir "$OUTPUT_DIR"

status=$?
if [ $status -ne 0 ]; then
    echo "Training failed with status $status"
    exit $status
fi

echo "Training completed. Model checkpoints saved in: $OUTPUT_DIR/checkpoints/"
