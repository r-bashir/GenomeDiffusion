# W&B Sweeps for GenomeDiffusion HPO

This setup provides automated hyperparameter optimization to address the training issues you've been experiencing (stagnant loss ~0.02, increasing MSE).

## 🎯 Key Issues Addressed

1. **Learning Rate Problems**: Fixed lr=1e-3 with min_lr=3e-3 (min > max)
2. **Model Architecture**: Testing deeper UNet configurations
3. **Missing Embeddings**: Exploring time/position embeddings
4. **Diffusion Parameters**: Optimizing beta schedules and timesteps
5. **Training Duration**: Testing longer training periods

## 📁 Files Created

- `sweep_config.yaml` - W&B sweep configuration with 50 hyperparameter combinations
- `train_sweep.py` - Modified training script integrated with W&B sweeps
- `run_sweep.py` - Management script for initializing and monitoring sweeps
- `sweep.slurm` - SLURM script for cluster execution

## 🚀 Quick Start

### 1. Initialize Sweep
```bash
python run_sweep.py --init
```
This will output a sweep ID like `username/project/sweep_id`

### 2. Run Sweep Agent (Local)
```bash
python run_sweep.py --agent <sweep_id> --count 10
```

### 3. Run Sweep Agent (Cluster)
```bash
sbatch sweep.slurm <sweep_id> 10
```

### 4. Monitor Progress
```bash
python run_sweep.py --monitor <sweep_id>
```

### 5. Analyze Results
```bash
python run_sweep.py --analyze <sweep_id>
```

## 🔧 Key Parameters Being Optimized

### Critical Learning Parameters
- **Learning Rate**: 1e-5 to 1e-2 (log scale)
- **Weight Decay**: 1e-5 to 1e-2 (log scale)
- **Scheduler Type**: cosine vs reduce
- **Min Learning Rate**: 1e-7 to 1e-4

### Model Architecture
- **Embedding Dimension**: [16, 32, 64, 128]
- **Model Depth**: [[1,2], [1,2,4], [1,2,4,8]]
- **Time Embeddings**: true/false
- **Position Embeddings**: true/false
- **Edge Padding**: [1, 2, 4]

### Diffusion Parameters
- **Beta Schedule**: linear vs cosine
- **Beta Start**: 1e-5 to 1e-3 (log scale)
- **Beta End**: 0.01 to 0.05
- **Timesteps**: [500, 1000, 2000]

### Training Parameters
- **Batch Size**: [16, 32, 64]
- **Epochs**: [100, 200, 300]
- **Sequence Length**: [50, 100, 200]

## 🎯 Expected Improvements

The sweep will systematically explore:

1. **Better Learning Rates**: Finding optimal lr that enables proper convergence
2. **Deeper Models**: Testing if more model capacity helps capture genomic patterns
3. **Proper Embeddings**: Time/position embeddings are crucial for diffusion models
4. **Optimal Diffusion**: Better noise schedules for genomic data
5. **Training Stability**: Longer training with proper early stopping

## 📊 Monitoring

The sweep uses Bayesian optimization to efficiently explore the hyperparameter space. Key metrics tracked:

- `val_loss` (primary optimization target)
- `train_loss`
- `final/val_loss`
- `model/total_params`
- `model/size_mb`

## 🏆 Expected Outcomes

Based on the issues you described, the sweep should find configurations that:

1. **Reduce Training Loss**: From ~0.02 to <0.01
2. **Improve MSE**: Decreasing rather than increasing MSE(x_{t-1}, x_0)
3. **Better Convergence**: Stable training with proper learning curves
4. **Optimal Architecture**: Right balance of model capacity and efficiency

## 🔍 Analysis Features

The analysis script will provide:
- Top performing configurations
- Parameter sensitivity analysis
- Specific recommendations for your next training run
- Comparison of different architectural choices

## 💡 Next Steps After Sweep

1. **Identify Best Config**: Use `--analyze` to find optimal hyperparameters
2. **Update config.yaml**: Apply best parameters to your main config
3. **Run Full Training**: Train final model with optimized settings
4. **Validate Results**: Re-run your `run_ddpm.py` analysis to confirm improvements

## 🛠️ Troubleshooting

- **OOM Errors**: Sweep automatically reduces batch size for large models
- **Failed Runs**: Early stopping prevents wasted compute on poor configs
- **Slow Progress**: Bayesian optimization improves efficiency over time
- **Cluster Issues**: Check SLURM logs in `logs/` directory

Start with the initialization step and let the sweep run overnight for best results!
