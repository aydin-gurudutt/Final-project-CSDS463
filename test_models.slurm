#!/bin/bash
#SBATCH --job-name=mil_smoke
#SBATCH --account=ccipd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

PROJECT=/mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject

module load Python/3.11.3-GCCcore-12.3.0
source "$PROJECT/.venv/bin/activate"

cd "$PROJECT/karthikraj_baseline"

MODEL="${MODEL:?Specify MODEL=clam or MODEL=dsmil}"

if [[ "$MODEL" == "clam" ]]; then
    SCRIPT=train_soup3.py
elif [[ "$MODEL" == "dsmil" ]]; then
    SCRIPT=train_soup3_dsmil.py
else
    echo "Unsupported model: $MODEL"
    exit 1
fi

echo "Testing model: $MODEL"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"

python -u "$SCRIPT" \
    --model "$MODEL" \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --nb_models 3 \
    --nb_epochs 5 \
    --num_epochs 3 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --init_seed 1 \
    --gpu_index 0

echo "Smoke test completed: $MODEL"
