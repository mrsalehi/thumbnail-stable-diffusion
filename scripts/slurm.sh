#!/bin/bash
#SBATCH --job-name=thumbnail_ftune_20_kids_channels
#SBATCH --output=/home/user/thumbnail-stable-diffusion/slurm_outputs/%x.out
#SBATCH --error=/home/user/thumbnail-stable-diffusion/slurm_outputs/%x.err
#SBATCH --time=96:00:00
#SBATCH --chdir=/home/user/thumbnail-stable-diffusion/src
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION
#SBATCH --mem=128G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=MAIL
#SBATCH --signal=B:TERM@120
#SBATCH --exclude=g3007

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
seed=$RANDOM
offset=12867
export MASTER_PORT=$((seed+offset))
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export HUGGINGFACE_CACHE="/home/user/cache"
export TRANSFORMERS_CACHE="/home/user/cache"
export DIFFUSERS_CACHE="/home/user/cache"
export NCCL_DEBUG=INFO

srun --cpu_bind=v --accel-bind=gn bash ../scripts/run.sh