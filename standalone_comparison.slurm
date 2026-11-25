#!/usr/bin/env bash
#SBATCH --job-name=mil_standalone
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/standalone_%j.out
#SBATCH --error=logs/standalone_%j.err
set -euo pipefail
PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"
STAGE="${STAGE:?Specify STAGE=train or STAGE=evaluate}"
mkdir -p logs
case "$STAGE" in
  train)
    ARCH="${ARCH:?Specify ARCH=clam or ARCH=dsmil}"
    SEED="${SEED:?Specify SEED=1, 2, or 3}"
    case "$ARCH:$SEED" in
      clam:1|clam:2|clam:3|dsmil:1|dsmil:2|dsmil:3) ;;
      *) echo "Invalid architecture/seed" >&2; exit 1;;
    esac
    echo "STANDALONE train $ARCH seed=$SEED job=$SLURM_JOB_ID"
    python -u standalone_comparison.py train --architecture "$ARCH" --seed "$SEED"
    ;;
  evaluate)
    echo "STANDALONE evaluate all six models job=$SLURM_JOB_ID"
    python -u standalone_comparison.py evaluate
    ;;
  *) echo "Invalid STAGE: $STAGE" >&2; exit 1;;
esac
echo "Completed: $(date)"
