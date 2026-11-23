#!/bin/bash
#SBATCH --job-name=mil_stability
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stability_%j.out
#SBATCH --error=logs/stability_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

cd "$PROJECT/karthikraj_baseline"

MODEL="${MODEL:?Specify MODEL}"
LAMBDA="${STABILITY_LAMBDA:?Specify STABILITY_LAMBDA}"
SEED="${SEED:-1}"

case "$MODEL" in
    clam) SCRIPT=train_stability_clam.py ;;
    dsmil) SCRIPT=train_stability_dsmil.py ;;
    *) echo "Invalid model: $MODEL"; exit 1 ;;
esac

echo "Model: $MODEL"
echo "Lambda: $LAMBDA"
echo "Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

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
    --stability_lambda "$LAMBDA" \
    --gpu_index 0

echo "Stability test completed"
