#!/bin/bash
#SBATCH --job-name=sardinn_infer
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/rkabir5/fyexam/logs/inference_%j.out
#SBATCH --error=/scratch/rkabir5/fyexam/logs/inference_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sardinn

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 /scratch/rkabir5/fyexam/code/run_inference.py

echo "Finished: $(date)"
