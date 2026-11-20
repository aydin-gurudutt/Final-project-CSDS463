#!/bin/bash
#SBATCH --job-name=mil_full
#SBATCH --account=csds463
#SBATCH --partition=markov_gpu
#SBATCH --exclude=class21t
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/vstor/courses/csds463/kxd457/Final_project/mil_merging/logs/slurm_%j.out
#SBATCH --error=/mnt/vstor/courses/csds463/kxd457/Final_project/mil_merging/logs/slurm_%j.err

set -euo pipefail

PROJECT="/mnt/vstor/courses/csds463/kxd457/Final_project"
cd "$PROJECT/mil_merging"

module load Python/3.11.5-GCCcore-13.2.0
source "$PROJECT/.venv/bin/activate"

MODEL="${MODEL:?Submit with MODEL=abmil or MODEL=clam}"

if [[ "$MODEL" != "abmil" && "$MODEL" != "clam" ]]; then
    echo "ERROR: Invalid model: $MODEL"
    exit 1
fi

echo "Model: $MODEL"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Account: $SLURM_JOB_ACCOUNT"
echo "Compute Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Verify GPU
python - <<'PY'
import torch

assert torch.cuda.is_available(), "ERROR: CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))

x = torch.ones((100, 100), device="cuda")
print("CUDA test passed:", x.shape)
PY

# Verify dataset
python - <<'PY'
from pathlib import Path
import pandas as pd

dataset = Path("datasets/camelyon16")

assert dataset.is_dir(), "Dataset directory missing"

for filename, expected in [
    ("camelyon16.csv", 397),
    ("train.csv", 215),
    ("val.csv", 54),
    ("test.csv", 128),
]:
    df = pd.read_csv(dataset / filename)
    assert len(df) == expected, f"Incorrect count in {filename}"
    assert df["path"].map(lambda p: Path(p).is_file()).all(), \
        f"Missing feature files in {filename}"

assert len(list((dataset / "features").glob("*.csv"))) == 397
print("Dataset verification passed")
PY

# Full model-fusion training
python -u train_soup3.py \
    --model "$MODEL" \
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

echo "Completed $MODEL at $(date)"