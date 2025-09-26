# Diffusion-Deep-Learning-for-Genome-Variant-Data

## _1. Working Locally_

Working locally on PC/Laptop might get killed due to memory unavailability, as the training batch might not fit in memory. So, make things a bit simpler by clipping the sequence length from _`n_markers=160858`_ to _`n_markers=1000`_. We can easily slice the SNP sequence by setting `seq_length=1000` in the `config.yaml` under `data`. The _`config["data"]["seq_length"]`_ will adjust functionality of _`dataset.py`_, _`ddpm.py`_, etc.


To run training and inference, call the relevant scripts from the root project. All training is controlled through _`config.yaml`_:

``` shell
# activate conda environment
conda activate <env-name>
```

```shell
# run training, required --config flag
python train.py --config config.yaml
```

```shell
# resume training from a checkpoint
# use --resume-strategy trainer to fully resume the same run (optimizer/scheduler/epoch)
python train.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --resume-strategy trainer

# or load weights only and start a fresh run (new optimizer/scheduler/version)
python train.py --config config.yaml --checkpoint path/to/checkpoint.ckpt --resume-strategy weights
```

```shell
# run inference, required '--checkpoint' flag
python inference.py --checkpoint path/to/checkpoint.ckpt
```

To perform further analysis, we have several scripts under `scripts/`, all of which should be executed from the project root, similar to `inference.py` above. Note that `config` is directly extracted from the checkpoint:

```shell
# run scripts from the scripts dir
python scripts/run_ddpm.py --checkpoint path/to/checkpoint.ckpt
```



## _2. Working Remotely_

To work remotely, _e.g._ on a cluster, one may need to set special settings as most clusters put restrictions on using their resources. In addition, one may need to run training in a containerised environment. So we have _`train.sh`_ and _`inference.sh`_ that similarly run training and inference as _`train.py`_ and _`inference.py`_ mentioned above. The only difference is that the bash script sets the cluster environment.

In our case, we work on the Berzelius cluster as follows.

### _2.1. Log in to Berzelius_

```shell
# ssh connection (berzelius is set in ~/.ssh/config)
ssh berzelius 
Verification code: ...

# go to project repo
cd proj/GenomeDiffusion
```

### _2.2 Training on Berzelius_

We are using separate scripts for training and inference, which utilise bash scripts that contain all necessary settings to run Python scripts within an AppTainer container on Berzelius, providing GPU support. Simply run the following commands one after the other:

```shell
# allocate GPU
interactive --gpus 1

# run training
./train.sh

# run inference
./inference.sh path/to/checkpoint.ckpt
```

**Note**: after `train.sh` is finished, it will print `path/to/<run number>/checkpoints`. Results from `inference.sh` are saved in `path/to/<run number>/inference/`. All paths are printed at the end of each script when finished.


### _2.3 Load/unload from Berzelius_

```shell
# use SFTP, same as SSH
sftp berzelius 
Verification code:

# once connected, download full run (contains checkpoints, evaluation and inference results)
sftp> get -r path/to/run
```

## 3. Sweeps (HPO)

See `SWEEP.md` for the complete guide to running W&B Sweeps locally and on the cluster, including re-tuning from checkpoints and troubleshooting. Below are quick-start commands.

### Local (GPU)

```shell
# initialize a sweep and inject a checkpoint + resume strategy
python run_sweep.py --init --config sweep_unet.yaml --project HPO \
  --checkpoint your-entity/GenomeDiffusion/model-artifact:best \
  --resume-strategy weights

# start an agent
SWEEP_ID=$(python -c "import yaml; print(yaml.safe_load(open('current_sweep.yaml'))['sweep_id'])")
python run_sweep.py --agent "$SWEEP_ID" --project HPO
```

### Cluster (AppTainer)

```shell
# single job (init + one agent + analyze)
sbatch sweep.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights

# parallel agents (array, one GPU per agent)
sbatch sweep_parallel.slurm sweep_unet.yaml HPO your-entity/GenomeDiffusion/model-artifact:best weights

# saturate one GPU with N concurrent agents
sbatch sweep_saturate.slurm sweep_unet.yaml 3 HPO your-entity/GenomeDiffusion/model-artifact:best weights
```