#!/bin/bash
#SBATCH --job-name=clam_fusion
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fusion_%j.out
#SBATCH --error=logs/fusion_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"

LAMBDA="${LAMBDA:?Specify LAMBDA=0.0 or 0.5}"
SEED="${SEED:-1}"

case "$LAMBDA" in
    0.0|0.5) ;;
    *) echo "Unsupported lambda: $LAMBDA"; exit 1 ;;
esac

RUN_DIR=weights/clam/20261122212201489441
MERGED="$RUN_DIR/offline_merges/merged_lambda_${LAMBDA}.pth"
OUTPUT="$RUN_DIR/fusion_results/lambda_${LAMBDA}_seed_${SEED}"

test -f "$MERGED"

if [[ -e "$OUTPUT" ]]; then
    echo "Output directory already exists: $OUTPUT"
    exit 1
fi

echo "Lambda: $LAMBDA"
echo "Training seed: $SEED"
echo "Merged checkpoint: $MERGED"
echo "Output directory: $OUTPUT"
echo "Node: $(hostname)"

python -u train_fusion_clam.py \
    --merged_checkpoint "$MERGED" \
    --output_dir "$OUTPUT" \
    --seed "$SEED" \
    --model clam \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --gpu_index 0

echo "FUSION JOB COMPLETED: lambda=$LAMBDA"
