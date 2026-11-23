#!/bin/bash
#SBATCH --job-name=mil_candidates
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/candidates_%j.out
#SBATCH --error=logs/candidates_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

cd "$PROJECT/karthikraj_baseline"

MODEL="${MODEL:-clam}"
SEED="${SEED:-1}"

case "$MODEL" in
    clam) SCRIPT=train_candidates_clam.py ;;
    dsmil) SCRIPT=train_candidates_dsmil.py ;;
    *) echo "Invalid model: $MODEL"; exit 1 ;;
esac

echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"

python -u "$SCRIPT" \
    --model "$MODEL" \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --nb_models 5 \
    --nb_epochs 5 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --init_seed "$SEED" \
    --gpu_index 0 \
    --candidate_only

echo "Candidate-only job completed"
