#!/usr/bin/env bash

# Strict mode: exit on error, undefined vars are errors, and fail pipelines on first failure
set -euo pipefail

# Usage and argument parsing
if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/checkpoint.ckpt"
  echo "Example: $0 /home/aakram/Current/GenDiffusion/outputs/lightning_logs/ResDiff1k/a6fqb2ft/checkpoints/416-0.01.ckpt"
  exit 1
fi

CKPT_PATH="$1"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: checkpoint file not found: $CKPT_PATH" >&2
  exit 1
fi

# Derive run directory: two steps back from checkpoint
# e.g., .../a6fqb2ft/checkpoints/<file>.ckpt -> .../a6fqb2ft
RUN_DIR="$(dirname "$(dirname "$CKPT_PATH")")"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Error: derived run directory not found: $RUN_DIR" >&2
  exit 1
fi

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

log_and_run() {
  local name="$1"; shift
  local log_file="$RUN_DIR/${name}.log"
  echo "["$(timestamp)"] >>> Starting ${name} | Log: ${log_file}"
  # tee writes to both console and file; pipefail ensures failures propagate
  ( "$@" ) 2>&1 | tee "$log_file"
  echo "["$(timestamp)"] <<< Completed ${name}"
}

echo "Run directory: $RUN_DIR"
echo "Checkpoint:    $CKPT_PATH"

# Ensure Python runs from repository root (this script resides at repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) Inference (writes outputs to RUN_DIR/inference by script design)
log_and_run inference python inference.py --checkpoint "$CKPT_PATH"

# 2) Comprehensive analysis (expects inference artifacts in RUN_DIR/inference)
log_and_run sample_analysis python sample_analysis.py --checkpoint "$CKPT_PATH"

# 3) DDPM diagnostics
log_and_run run_ddpm python scripts/run_ddpm.py --checkpoint "$CKPT_PATH"

# 4) LD investigation
log_and_run run_ld python scripts/run_ld.py --checkpoint "$CKPT_PATH"

# 5) Reverse diffusion diagnostics
log_and_run run_reverse python scripts/run_reverse.py --checkpoint "$CKPT_PATH"

echo "All tasks completed successfully. Logs saved under: $RUN_DIR"
