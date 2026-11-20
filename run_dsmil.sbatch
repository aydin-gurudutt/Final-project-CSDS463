#!/bin/bash
#SBATCH --job-name=dsmil_full
#SBATCH --account=csds463
#SBATCH --partition=markov_gpu
#SBATCH --exclude=class21t
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/vstor/courses/csds463/kxd457/Final_project/mil_merging/logs/dsmil_%j.out
#SBATCH --error=/mnt/vstor/courses/csds463/kxd457/Final_project/mil_merging/logs/dsmil_%j.err

set -euo pipefail

PROJECT="/mnt/vstor/courses/csds463/kxd457/Final_project"
cd "$PROJECT/mil_merging"

module load Python/3.11.5-GCCcore-13.2.0
source "$PROJECT/.venv/bin/activate"
hash -r

echo "Model: DSMIL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Verify CUDA before starting
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((100, 100), device="cuda")
print("CUDA tensor test passed:", x.shape)
PY

# Full DSMIL fusion training
python -u train_soup3_dsmil.py \
    --model dsmil \
    --dataset camelyon16 \
    --num_classes 1 \
    --feats_size 512 \
    --nb_models 10 \
    --nb_epochs 5 \
    --num_epochs 200 \
    --lr 0.0002 \
    --weight_decay 0.005 \
    --init_seed 1 \
    --gpu_index 0

echo "DSMIL training completed successfully at $(date)"
