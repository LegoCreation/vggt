#!/bin/bash
#SBATCH --job-name="VGGT\_finetuning\_all\_losses"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=node21
#SBATCH --gres=gpu:1,VRAM:80G
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/prakt/s0016/slurm\_logs/slurm.log
#SBATCH --error=/usr/prakt/s0016/slurm\_logs/slurm\_err.log
#SBATCH --qos=external

export NCCL_DEBUG=INFO
pwd; hostname; date
nvidia-smi

source ~/miniconda3/bin/activate 
conda activate vggt_base
# salloc --nodes=1 --nodelist=node21 --cpus-per-task=4 --mem=64G --gres=gpu:1,VRAM:60G --time=12:00:00 --mail-type=NONE -n 1 srun --pty bash -i
torchrun --nproc_per_node=1 launch.py --config uco3d_full_training
