#!/usr/bin/env bash
# Submit six standalone MIL runs on Pioneer and evaluate only if all succeed.
set -euo pipefail
cd "$(dirname "$0")"
test -f standalone_comparison.py && test -f standalone_comparison.slurm
mkdir -p logs
if [[ -e heldout_evaluation/standalone_auc_results.csv ]]; then
  echo "Standalone held-out evaluation already exists. Refusing to duplicate." >&2
  exit 1
fi
ids=()
for arch in clam dsmil; do
  for seed in 1 2 3; do
    job=$(sbatch --parsable --export=ALL,STAGE=train,ARCH="$arch",SEED="$seed" standalone_comparison.slurm)
    echo "Submitted $arch seed=$seed training job=$job"
    ids+=("$job")
  done
done
dep=$(IFS=:; echo "${ids[*]}")
eval_job=$(sbatch --parsable --dependency="afterok:$dep" --export=ALL,STAGE=evaluate standalone_comparison.slurm)
echo "Submitted held-out evaluation job=$eval_job, dependent on all six training jobs."
echo "After evaluation finishes, summarize with: python3 standalone_comparison.py summarize"
echo "Inspect status using: squeue -u $USER"
