#!/bin/bash
#SBATCH --job-name=abl_inf
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/rkabir5/fyexam/logs/ablation_inference_%j.out
#SBATCH --error=/scratch/rkabir5/fyexam/logs/ablation_inference_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sardinn

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /scratch/rkabir5/fyexam

for variant in ablation_depth_4 ablation_depth_6 ablation_depth_8_verify \
               ablation_depth_10 ablation_depth_12 \
               ablation_sine_none ablation_sine_L2 ablation_sine_L6 ablation_sine_all; do
    echo ""
    echo "=============================="
    echo "Inference: $variant"
    echo "=============================="
    python3 code/run_inference_ablation.py \
        --checkpoint checkpoints/${variant}/best_model.pth \
        --results_dir results/${variant} \
        --variant ${variant}
done

echo ""
echo "All ablation inference complete"
echo "Finished: $(date)"
