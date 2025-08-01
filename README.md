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
# resume training if disrupted, required --config and --resume flags
python train.py --config config.yaml --resume path/to/checkpoint.ckpt
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