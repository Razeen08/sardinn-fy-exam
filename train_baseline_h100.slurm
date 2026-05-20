#!/bin/bash
#SBATCH --job-name=sardinn_h100
#SBATCH --partition=preempt
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/rkabir5/fyexam/logs/baseline_%j.out
#SBATCH --error=/scratch/rkabir5/fyexam/logs/baseline_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sardinn

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python3 /scratch/rkabir5/fyexam/code/run_training.py \
    --variant baseline \
    --mlp_layers 8 \
    --sine_layer 4 \
    --n_epochs 200 \
    --lr 1e-5 \
    --batch_size 10 \
    --data_dir /scratch/rkabir5/fyexam/data/hcp_processed \
    --subjects_file /scratch/rkabir5/fyexam/subjects.txt \
    --checkpoint_dir /scratch/rkabir5/fyexam/checkpoints/baseline \
    --log_path /scratch/rkabir5/fyexam/logs/baseline/log.json

echo "Finished: $(date)"
