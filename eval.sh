#!/bin/bash -l

#SBATCH --job-name=evaluate-bert
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1

source ${HOME}/.bashrc
mamba activate baidu-bert-model

python src/evaluate.py
