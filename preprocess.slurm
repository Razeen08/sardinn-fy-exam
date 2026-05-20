#!/bin/bash
#SBATCH --job-name=hcp_preprocess
#SBATCH --partition=standard
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/rkabir5/fyexam/logs/preprocess_%j.out
#SBATCH --error=/scratch/rkabir5/fyexam/logs/preprocess_%j.err

# Thread settings for Patch2Self parallelism
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

module load python3/3.11.10
module load gcc/13.2.0/b1
module load hdf5/1.14.4-3/b2
source /scratch/rkabir5/fyexam/venv/bin/activate

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python3 /scratch/rkabir5/fyexam/preprocess.py

echo "Finished: $(date)"
