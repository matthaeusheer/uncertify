#!/bin/bash
#SBATCH  --output=../data/slurm_logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH --job-name train_vae

source /itet-stor/maheer/net_scratch/conda/etc/profile.d/conda.sh
conda activate cluster_pytorch_glow
mkdir -p ../data/slurm_logs

python -u ../scripts/train_vae.py "$@"
