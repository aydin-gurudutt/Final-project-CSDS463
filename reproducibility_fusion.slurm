#!/bin/bash
#SBATCH --job-name=repro_fusion
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/repro_fusion_%j.out
#SBATCH --error=logs/repro_fusion_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT/karthikraj_baseline"

: "${ARCH:?Missing ARCH}"
: "${SEED:?Missing SEED}"
: "${LAMBDA:?Missing LAMBDA}"

case "${ARCH}:${SEED}" in
  clam:2) RUN=weights/clam/20261123122950019167 ;;
  clam:3) RUN=weights/clam/20261123122950435501 ;;
  dsmil:2) RUN=weights/dsmil/20261123123022171992 ;;
  dsmil:3) RUN=weights/dsmil/20261123123007990510 ;;
  *) echo "Invalid architecture or seed"; exit 1 ;;
esac

case "${ARCH}:${LAMBDA}" in
  clam:0.0|clam:0.5|dsmil:0.0|dsmil:5.0) ;;
  *) echo "Invalid lambda"; exit 1 ;;
esac

MERGED="$RUN/offline_merges/merged_lambda_${LAMBDA}.pth"

# Reuse exactly the same initial merge if candidate
# membership did not change.
if [[ "$LAMBDA" != "0.0" ]]; then
    if [[ "$ARCH:$SEED" == "clam:2" ||
          "$ARCH:$SEED" == "dsmil:2" ||
          "$ARCH:$SEED" == "dsmil:3" ]]; then
        MERGED="$RUN/offline_merges/merged_lambda_0.0.pth"
    fi
fi

OUTPUT="$RUN/fusion_results/lambda_${LAMBDA}_seed_${SEED}"

test -f "$MERGED"

if [[ -e "$OUTPUT" ]]; then
    echo "Refusing to overwrite: $OUTPUT"
    exit 1
fi

echo "Architecture: $ARCH"
echo "Seed: $SEED"
echo "Lambda: $LAMBDA"
echo "Checkpoint: $MERGED"
echo "Output: $OUTPUT"
echo "Node: $(hostname)"

python -u "train_fusion_${ARCH}.py" \
    --merged_checkpoint "$MERGED" \
    --output_dir "$OUTPUT" \
    --seed "$SEED" \
    --model "$ARCH" \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --gpu_index 0

echo "REPRODUCIBILITY FUSION COMPLETE"
