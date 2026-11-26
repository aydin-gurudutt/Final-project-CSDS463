#!/bin/bash
#SBATCH --job-name=abmil_weighted
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-6%3
#SBATCH --output=logs/abmil_weighted_%A_%a.out
#SBATCH --error=logs/abmil_weighted_%A_%a.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"

case "${SLURM_ARRAY_TASK_ID:?SLURM array index required}" in
    1) SEED=1; METHOD=auc_weighted ;;
    2) SEED=1; METHOD=stability_weighted ;;
    3) SEED=2; METHOD=auc_weighted ;;
    4) SEED=2; METHOD=stability_weighted ;;
    5) SEED=3; METHOD=auc_weighted ;;
    6) SEED=3; METHOD=stability_weighted ;;
    *) echo "Unsupported array task" >&2; exit 1 ;;
esac

RUN="weights/abmil_weighted_fusions/seed_${SEED}/${METHOD}"
CHECKPOINT="$RUN/merged_initial.pth"
OUTPUT="$RUN/training"

test -s "$CHECKPOINT"
test ! -e "$OUTPUT"

echo "ABMIL weighted continuation; seed=$SEED method=$METHOD job=$SLURM_JOB_ID"
echo "Initialization: $CHECKPOINT"
echo "Start: $(date)"

python -u train_fusion_clam.py \
    --merged_checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT" \
    --seed "$SEED" \
    --model abmil \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --gpu_index 0

test -s "$OUTPUT/best_validation.pth"
test -s "$OUTPUT/history.json"

echo "WEIGHTED ABMIL CONTINUATION COMPLETE: seed=$SEED method=$METHOD"
echo "End: $(date)"
