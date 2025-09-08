#!/bin/bash
#SBATCH --job-name="VGGT_smart_port"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/usr/prakt/s0017/vggt/slurm.log
#SBATCH --error=/usr/prakt/s0017/vggt/slurm_err.log
#SBATCH --qos=external

export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Kill any lingering processes and wait
pkill -f "torchrun" || true
pkill -f "train.py" || true
sleep 3

pwd; hostname; date
nvidia-smi

source /usr/prakt/s0017/anaconda3/bin/activate
conda activate vggt_nvs

CONFIG_FILE="uco3d_nvs_training.yaml"

# Find a free port dynamically
FREE_PORT=$(python find_free_port.py)
if [ $? -eq 0 ]; then
    echo "Found free port: $FREE_PORT"
    torchrun --nproc_per_node=1 --master_port=$FREE_PORT training/launch.py --config $CONFIG_FILE
else
    echo "No free port found, trying direct python execution"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
    python train.py --config $CONFIG_FILE
fi 