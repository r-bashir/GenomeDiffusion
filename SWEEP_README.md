# W&B Sweeps: Hierarchical Guide

This document explains how to run hyperparameter sweeps for the SNP diffusion model using Weights & Biases. It is aligned with the current scripts and flags in this repository.

## 1) Prerequisites

- WANDB account and API key configured:
  - Export once per shell: `export WANDB_API_KEY=...`
  - Or run `wandb login`
- Base config exists: `config.yaml`
- Sweep config exists: `sweep.yaml`
- Training entrypoint for sweeps: `train_sweep.py`
- Optional cluster scripts: `sweep.slurm`, `sweep_parallel.slurm`

## 2) File Overview

- `sweep.yaml` — W&B sweep configuration (method, metric, parameter space). Should set `program: train_sweep.py`.
- `train_sweep.py` — Training script used by agents. Supports dynamic param overrides via CLI or `wandb.config`.
- `run_sweep.py` — CLI helper to init sweeps, run agents, monitor, and analyze.
- `sweep.slurm` — SLURM script to run a single agent on a cluster.
- `sweep_parallel.slurm` — SLURM script to run multiple agents in parallel (if desired).

## 3) Local Workflow (recommended for smoke tests)

1. Initialize a sweep (creates a sweep on W&B and saves `current_sweep.yaml`):
   ```bash
   python run_sweep.py --init --config sweep.yaml --base-config config.yaml
   ```

2. Run an agent locally (repeat to spawn more agents):
   ```bash
   # Using full path sweep ID
   python run_sweep.py --agent <entity/project>/<sweep_id> --count 4

   # Or use the short ID if `current_sweep.yaml` is present
   python run_sweep.py --agent <sweep_id> --count 4
   ```

3. Monitor progress:
   ```bash
   python run_sweep.py --monitor <sweep_id> --project <project_name>
   ```

4. Analyze results and get recommendations:
   ```bash
   python run_sweep.py --analyze <sweep_id> --project <project_name>
   ```

## 4) Cluster Workflow (SLURM)

1. Submit a single agent job:
   ```bash
   sbatch sweep.slurm <sweep_id> <count>
   # Example
   sbatch sweep.slurm abc123 8
   ```

2. Submit multiple agents in parallel (if `sweep_parallel.slurm` is provided):
   ```bash
   sbatch sweep_parallel.slurm <sweep_id> <agents> <count_per_agent>
   # Example: 3 agents, each running 5 trials
   sbatch sweep_parallel.slurm abc123 3 5
   ```

Notes:
- SLURM scripts generally run the W&B agent which calls `train_sweep.py` per trial.
- Ensure the container/environment includes CUDA, PyTorch, Lightning, and W&B.

## 5) Overriding Parameters Manually

You can run the training script standalone for quick tests or to debug parameter mappings:

```bash
python train_sweep.py \
  --config config.yaml \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --unet.use_attention true \
  --attention_heads 4 \
  --loss.use_discrete_loss true \
  --discrete_penalty_weight 0.2
```

Unknown CLI args are parsed and mapped into the hierarchical config (see `update_config_with_sweep_params()` in `train_sweep.py`). During sweeps, values from `wandb.config` are merged with these CLI overrides.

## 6) What Gets Optimized (examples)

- Optimizer: learning rate, weight decay, betas, eps, amsgrad
- Scheduler: type (`cosine` or `reduce`), eta_min/min_lr, factor, patience
- Data: batch size, seq length, workers, scaling
- UNet: embedding_dim, dim_mults, channels, kernel_sizes, dilations, norm_groups
- Attention: use_attention, attention_heads, attention_dim_head
- Loss: use_discrete_loss, discrete_penalty_weight
- Diffusion: timesteps, beta_start, beta_end, schedule_type

Refer to `sweep.yaml` for the exact search space.

## 7) Monitoring and Metrics

Key metrics surfaced in W&B:
- `val_loss`, `val_loss_epoch`, `final/val_loss`
- `train_loss`, `train_loss_epoch`
- `model/total_params`, `model/trainable_params`, `model/size_mb`

`run_sweep.py --monitor` prints recent runs, states, durations, and best run summary.

## 8) Troubleshooting

- __WANDB_API_KEY missing__: export it or run `wandb login`.
- __Short vs full sweep IDs__: If providing a short ID, we try to resolve entity/project via `current_sweep.yaml`. Otherwise pass `<entity>/<project>/<id>`.
- __OOM errors__: `train_sweep.py` reduces batch size for large models/sequences.
- __Scheduler min lr issues__: `train_sweep.py` auto-fixes inconsistent `eta_min`/`min_lr`.
- __Cluster issues__: check SLURM logs; confirm the environment has required packages and GPU access.

---

Start by initializing the sweep, then run one or more agents (locally or on SLURM). Monitor progress and use the analysis output to update `config.yaml` for your final training runs.
