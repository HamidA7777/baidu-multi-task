#!/bin/bash -l

#SBATCH --job-name=bert-training
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2

source ${HOME}/.bashrc
mamba activate baidu-bert-model

torchrun --nproc_per_node=2 src/train_ddp.py
