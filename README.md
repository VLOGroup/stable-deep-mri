![Stable Deep MRI Reconstruction](./assets/sketch.pdf)
# Stable Deep MRI Reconstruction
This repository contains the code of [Stable Deep MRI Reconstruction using Generative Priors](https://arxiv.org/pdf/2210.13834.pdf).

## Usage
The framework assumes the environment variables `EXPERIMENTS_ROOT` (needed for training) and `DATASETS_ROOT` (needed for training and evaluation) to be set.
`EXPERIMENTS_ROOT` is the output base-directory for training can be any directory on the machine.
`DATASETS_ROOT` should contain the `fastmri` dataset, i.e. the directory `$DATASETS_ROOT/fastmri/multicoil_train` (e.g.) should exist.

### Training
To train the model, run `python train.py output_dir`.
The first argument is the experiment output directory, i.e. checkpoints and losses etc. will be saved to `$EXPERIMENTS_ROOT/output_dir/`.

### Evaluation
All evaluation code is found in `evaluate.py`.
The `if __name__ == '__main__':` block lists evaluation functions along with annotations indicating the corresponding table or figure in the paper.
Pretrained models and other data needed for evaluation can be found [here](https://files.icg.tugraz.at/d/573098a94ecc4710b80e/).
