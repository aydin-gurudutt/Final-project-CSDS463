#!/bin/bash
#SBATCH --job-name=ensemble_A_summary
#SBATCH --account=ccipd
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ensemble_A_summary_%j.out
#SBATCH --error=logs/ensemble_A_summary_%j.err

set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"
python -u experiment_a_ensemble.py summarize
