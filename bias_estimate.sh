#!/bin/bash -l

#SBATCH --job-name=estimate_bias_statistics
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source ${HOME}/.bashrc
mamba activate baidu-bert-model

python src/bias_estimation.py
