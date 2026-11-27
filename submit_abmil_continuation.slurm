#!/bin/bash
#SBATCH --job-name=abmil_cont
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-9%3
#SBATCH --output=logs/abmil_cont_%A_%a.out
#SBATCH --error=logs/abmil_cont_%A_%a.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

cd "$PROJECT/karthikraj_baseline"

TASK="${SLURM_ARRAY_TASK_ID}"

case "$TASK" in
  1) SEED=1; METHOD=standalone ;;
  2) SEED=1; METHOD=baseline ;;
  3) SEED=1; METHOD=stability ;;
  4) SEED=2; METHOD=standalone ;;
  5) SEED=2; METHOD=baseline ;;
  6) SEED=2; METHOD=stability ;;
  7) SEED=3; METHOD=standalone ;;
  8) SEED=3; METHOD=baseline ;;
  9) SEED=3; METHOD=stability ;;
  *) echo "Invalid task ID"; exit 1 ;;
esac

BASE="weights/abmil_fusions/seed_${SEED}"

case "$METHOD" in
  standalone)
    CHECKPOINT="$BASE/standalone/selected_candidate.pth"
    ;;
  baseline)
    CHECKPOINT="$BASE/baseline/merged_initial.pth"
    ;;
  stability)
    CHECKPOINT="$BASE/stability/merged_initial.pth"
    ;;
esac

OUTPUT="$BASE/$METHOD/training"

test -f "$CHECKPOINT"
test ! -e "$OUTPUT"

echo "Architecture: ABMIL"
echo "Seed: $SEED"
echo "Method: $METHOD"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
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

test -f "$OUTPUT/best_validation.pth"
test -f "$OUTPUT/history.json"

echo "ABMIL CONTINUATION COMPLETE"
echo "Seed: $SEED"
echo "Method: $METHOD"
echo "End: $(date)"
