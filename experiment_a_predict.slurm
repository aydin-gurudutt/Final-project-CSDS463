#!/bin/bash
#SBATCH --job-name=ensemble_A
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-3%3
#SBATCH --output=logs/ensemble_A_%A_%a.out
#SBATCH --error=logs/ensemble_A_%A_%a.err

set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"
python -u experiment_a_ensemble.py predict --seed "${SLURM_ARRAY_TASK_ID}"
