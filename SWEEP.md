# GenomeDiffusion: W&B Sweeps Guide

This guide documents how to run hyperparameter sweeps for GenomeDiffusion using Weights & Biases (W&B), both locally and on the cluster. It also covers re-tuning from a checkpoint (weights-only or full trainer resume).

Key scripts:
- `run_sweep.py`: sweep orchestration (init, agent, analyze)
- `train_sweep.py`: training entrypoint used by sweeps (defined in `sweep_unet.yaml` as `program`)
- `sweep_unet.yaml`: sweep configuration and parameter search space
- Cluster launchers: `sweep.slurm`, `sweep_parallel.slurm`

---

## 1) Prerequisites
- W&B login
  ```bash
  wandb login
  ```
- GPU visibility (optional but typical for local runs)
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```
- Confirm `sweep_unet.yaml` has `program: train_sweep.py` (it does), and desired search space.

---

## 2) Resume Strategies (checkpointed re-tuning)
`train_sweep.py` supports two strategies via `training.resume_strategy`:
- `weights` (recommended for HPO): Load model weights from a checkpoint and start fresh optimizer/scheduler.
- `trainer`: Resume entire trainer state (optimizer/scheduler/epoch); use for continuing a single run rather than HPO.

Provide the checkpoint via `training.checkpoint`:
- Local `.ckpt` path
- Directory containing a `.ckpt` file
- W&B model artifact reference: `entity/project/artifact_or_run:alias`

These are injected at sweep initialization via `run_sweep.py --checkpoint` and `--resume-strategy` and delivered to `train_sweep.py` through `wandb.config`.

---

## 3) Local GPU Workflow

### 3.1 Initialize a sweep (inject checkpoint + resume)
- Local checkpoint example:
  ```bash
  python run_sweep.py \
    --init \
    --config sweep_unet.yaml \
    --project HPO \
    --checkpoint /absolute/path/to/best.ckpt \
    --resume-strategy weights
  ```

- W&B artifact checkpoint example:
  ```bash
  python run_sweep.py \
    --init \
    --config sweep_unet.yaml \
    --project HPO \
    --checkpoint your-entity/GenomeDiffusion/model-artifact:best \
    --resume-strategy weights
  ```
This creates a sweep and writes `current_sweep.yaml` with the sweep ID.

### 3.2 Launch agent(s)
```bash
SWEEP_ID=$(python -c "import yaml; print(yaml.safe_load(open('current_sweep.yaml'))['sweep_id'])")
python run_sweep.py --agent "$SWEEP_ID" --project HPO
```

To run multiple local agents (one per terminal / GPU):
```bash
# In another terminal
python run_sweep.py --agent "$SWEEP_ID" --project HPO
```

---

## 4) Cluster Workflow (AppTainer/Singularity)
The SLURM scripts run everything inside the container and already pass the sweep defaults to `run_sweep.py` during initialization.

Paths bound inside the container:
- Project: host `$PROJECT_DIR` → container `/workspace`
- Data: host `$DATA_DIR` → container `/data`

If you pass a local checkpoint path, it must be visible inside `/workspace` in the container. Prefer W&B artifacts for portability.

### 4.1 Single-job sweep (init + one agent + analyze)
Submit with optional 3rd and 4th args for checkpoint and resume strategy:
```bash
# Using a local path inside the project dir bound as /workspace
sbatch sweep.slurm sweep_unet.yaml HPO /workspace/checkpoints/best.ckpt weights

# Using a W&B artifact (preferred)
sbatch sweep.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights
```

### 4.2 Parallel agents (array job; init by task 1)
```bash
# Local path (must exist under /workspace in the container)
sbatch sweep_parallel.slurm sweep_unet.yaml HPO /workspace/checkpoints/best.ckpt weights

# W&B artifact reference (preferred)
sbatch sweep_parallel.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights
```

These scripts will:
- Initialize the sweep via `run_sweep.py --init ... [--checkpoint ... --resume-strategy ...]`
- Start agent(s) via `run_sweep.py --agent <SWEEP_ID> --project <HPO>`
- Optionally analyze the sweep and dump best tuned parameters

### 4.3 Saturate one GPU with concurrent agents
Run multiple agents on the same GPU (use cautiously; ensure enough VRAM and adjust batch sizes if needed):

```bash
sbatch sweep_saturate.slurm sweep_unet.yaml 3 HPO your-entity/GenomeDiffusion/model-artifact:best weights
```

Notes:
- All agents will share the same GPU (`CUDA_VISIBLE_DEVICES=0` inside the script).
- Consider lowering `batch_size` in the sweep space to avoid OOM.
- Prefer W&B artifacts for the checkpoint to avoid container path issues.

---

## 5) What the scripts do
- `run_sweep.py`
  - `--init`: reads `sweep_unet.yaml`, injects defaults (`checkpoint`, `resume_strategy`) into `parameters.*.value`, creates sweep, writes `current_sweep.yaml` (or provided path).
  - `--agent <SWEEP_ID>`: launches a W&B agent that runs `train_sweep.py` per trial.
  - `--analyze <SWEEP_ID>`: collects best run and writes `best_config_<SWEEP_ID>.yaml` with tuned params only.

- `train_sweep.py`
  - Parses sweep params, merges into config, validates.
  - If `training.checkpoint` is set:
    - `resume_strategy=weights`: loads model weights only (fresh optimizer/scheduler).
    - `resume_strategy=trainer`: resumes full trainer via `Trainer.fit(..., ckpt_path=...)`.
  - Supports resolving W&B artifacts and local paths/directories.

- `sweep_unet.yaml`
  - Defines search space and sets `program: train_sweep.py`.

---

## 6) Troubleshooting
- W&B agent does not stop with Ctrl+C
  - Symptom: agent keeps running or leaves zombie processes.
  - Workarounds:
    - Stop the exact agent process (e.g., `ps aux | grep wandb` → kill PID).
    - If inside SLURM: `scancel <jobid>` for the agent job.
    - Ensure no nested subprocesses block signal handling; if needed, close terminals and re-open.

- Checkpoint not found / not resolved
  - If using a path, ensure it exists and is visible inside the container (`/workspace/...`).
  - If using a W&B artifact, verify the reference: `entity/project/artifact:alias` or a model artifact created by your run.

- CUDA device not visible
  - Export `CUDA_VISIBLE_DEVICES=<gpu_id>` and ensure your environment/container enables GPU (`--nv` flag with AppTainer).

- Resume strategy confusion
  - For HPO re-tuning use `weights`.
  - Use `trainer` only to continue a run as-is.

---

## 7) Examples at a glance

Local (artifact-based):
```bash
python run_sweep.py --init --config sweep_unet.yaml --project HPO \
  --checkpoint your-entity/GenomeDiffusion/model-artifact:best \
  --resume-strategy weights
SWEEP_ID=$(python -c "import yaml; print(yaml.safe_load(open('current_sweep.yaml'))['sweep_id'])")
python run_sweep.py --agent "$SWEEP_ID" --project HPO
```

Cluster (single job + artifact):
```bash
sbatch sweep.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights
```

Cluster (parallel agents + artifact):
```bash
sbatch sweep_parallel.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights
```

---

If you need help publishing a checkpoint as a W&B artifact or adapting paths to your cluster, open an issue or ping in the project chat.
