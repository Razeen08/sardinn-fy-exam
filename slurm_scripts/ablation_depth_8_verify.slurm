#!/bin/bash
#SBATCH --job-name=ablation_depth_8_verify
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/rkabir5/fyexam/logs/ablation_depth_8_verify_%j.out
#SBATCH --error=/scratch/rkabir5/fyexam/logs/ablation_depth_8_verify_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sardinn

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 /scratch/rkabir5/fyexam/code/run_training.py \
    --variant ablation_depth_8_verify \
    --mlp_layers 8 \
    --sine_layer 4 \
    --n_epochs 80 \
    --lr 1e-5 \
    --batch_size 10 \
    --data_dir /scratch/rkabir5/fyexam/data/hcp_processed \
    --subjects_file /scratch/rkabir5/fyexam/subjects.txt \
    --checkpoint_dir /scratch/rkabir5/fyexam/checkpoints/ablation_depth_8_verify \
    --log_path /scratch/rkabir5/fyexam/logs/ablation_depth_8_verify/log.json

echo "Finished: $(date)"
