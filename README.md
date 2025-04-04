# Diffusion-Deep-Learning-for-Genome-Variant-Data




## _1. Login to Berzelius_


```shell
# ssh connection (berzelius is set in ~/.ssh/config)
ssh berzelius 
Verification code: ...

# go to project repo
cd proj/GenomeDiffusion
```

### _1.1 Working on Berzelius_

We are using separate scripts for training, testing and inference using bash scripts that contain all settings to run python scripts using apptainer container on cluster with GPUs. Simply run the following commands one after the other:

```shell
# allocate GPU
interactive --gpus 1

# run training
./train.sh

# run testing/evaluation
./test.sh path/to/checkpoints/last.ckpt

# run inference
./inference.sh path/to/checkpoints/last.ckpt
```

**Note**: after `train.sh` is finished, it will print `path/to/run/checkpoints`. Results from `test.sh` and `inference.sh` are saved in `path/to/run/evaluation/` and `path/to/run/inference/`. All paths are printed at the end of each script when finished.


### _1.2 Load Data from Berzelius_

```shell
# use SFTP, same as SSH
sftp berzelius 
Verification code:

# once connected, download full run (contains checkpoints, evaluation and infernece results)
sftp> get -r path/to/run
```