#!/bin/bash
#SBATCH --job-name=greedy_B
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=1-9%2
#SBATCH --output=logs/greedy_B_%A_%a.out
#SBATCH --error=logs/greedy_B_%A_%a.err
set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"
case "$SLURM_ARRAY_TASK_ID" in
  1) ARCH=clam; SEED=1 ;;
  2) ARCH=clam; SEED=2 ;;
  3) ARCH=clam; SEED=3 ;;
  4) ARCH=abmil; SEED=1 ;;
  5) ARCH=abmil; SEED=2 ;;
  6) ARCH=abmil; SEED=3 ;;
  7) ARCH=dsmil; SEED=1 ;;
  8) ARCH=dsmil; SEED=2 ;;
  9) ARCH=dsmil; SEED=3 ;;
  *) echo "Invalid array index" >&2; exit 1 ;;
esac
python -u experiment_b_greedy_soup.py --architecture "$ARCH" --seed "$SEED"
