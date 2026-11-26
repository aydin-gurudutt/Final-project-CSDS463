#!/bin/bash
#SBATCH --job-name=dsmil_weighted
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-6%3
#SBATCH --output=logs/dsmil_weighted_%A_%a.out
#SBATCH --error=logs/dsmil_weighted_%A_%a.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject
module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"

case "${SLURM_ARRAY_TASK_ID:?SLURM array task index required}" in
  1) SEED=1; METHOD=auc_weighted ;;
  2) SEED=1; METHOD=stability_weighted ;;
  3) SEED=2; METHOD=auc_weighted ;;
  4) SEED=2; METHOD=stability_weighted ;;
  5) SEED=3; METHOD=auc_weighted ;;
  6) SEED=3; METHOD=stability_weighted ;;
  *) echo "Invalid array index" >&2; exit 1 ;;
esac

ROOT="weights/dsmil_weighted_fusions/seed_${SEED}/${METHOD}"
CHECKPOINT="$ROOT/merged_initial.pth"
OUTPUT="$ROOT/training"

test -s "$CHECKPOINT"
test ! -e "$OUTPUT"

echo "Architecture: dsmil"
echo "Seed: $SEED"
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Started: $(date)"

python -u train_fusion_dsmil.py \
  --merged_checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT" \
  --seed "$SEED" \
  --model dsmil \
  --dataset camelyon16 \
  --num_classes 1 \
  --feats_size 512 \
  --num_epochs 3 \
  --lr 0.0002 \
  --weight_decay 0.005 \
  --gpu_index 0

test -s "$OUTPUT/best_validation.pth"
test -s "$OUTPUT/history.json"

echo "DSMIL WEIGHTED CONTINUATION COMPLETE seed=$SEED method=$METHOD"
echo "Ended: $(date)"
