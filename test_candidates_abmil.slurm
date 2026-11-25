#!/bin/bash
#SBATCH --job-name=abmil_candidates
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-3%2
#SBATCH --output=logs/abmil_candidates_%A_%a.out
#SBATCH --error=logs/abmil_candidates_%A_%a.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"

MODEL=abmil
SEED="${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}"

echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

python -u train_candidates_clam.py \
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

echo "End time: $(date)"
echo "ABMIL CANDIDATE GENERATION COMPLETED: seed=$SEED"
