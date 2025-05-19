#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --mem=256gb
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4             # processes per node
#SBATCH --gpus-per-node=4               # GPUs per node
#SBATCH --account=OD-233566       # Project ID from get_project_codes

# Get interactive session:
# salloc --nodes=2 --ntasks=4 --gpus=4 --cpus-per-task=4 --partition=gpu --time=02:00:00 --account=OD-233566

module load cuda/12.8.1
module load nccl
source llama-env/bin/activate

export HF_HOME=/scratch3/$USER/cache

MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((12300 + RANDOM % 50000))

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Node rank: $SLURM_NODEID"

# Note: --nproc_per_node must not greater than GPUs per node
torchrun --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py