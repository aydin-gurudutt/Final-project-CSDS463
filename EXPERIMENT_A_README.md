# Experiment A — Cross-architecture prediction ensembling (validation only)

## Goal

Compare **CLAM alone**, **CLAM + ABMIL**, and **CLAM + ABMIL + DSMIL** with fixed, equal-weight probability averaging. Each of the three seeds is evaluated twice:

- `standalone`: the validation-selected, continued **single-best-candidate** model for each architecture.
- `baseline`: the validation-selected, continued **original equal-weight merged baseline** for each architecture.

This avoids blending performance effects from the new stability-weighted fusion tests into the primary ensemble comparison. Existing trained checkpoints are reused, not overwritten. The script performs inference on the original **54-slide validation split only**; the 128-slide held-out test split is read only for verifying filename disjointness and is **never inferred on**.

### Primary predeclared comparisons

```text
CLAM                        : p_CLAM
CLAM+ABMIL                  : (p_CLAM + p_ABMIL) / 2
CLAM+ABMIL+DSMIL            : (p_CLAM + p_ABMIL + p_DSMIL) / 3
```

These are raw sigmoid probabilities, not AUC-normalized or tuned weights. CLAM and ABMIL use their bag-logit sigmoid; DSMIL uses its original 50/50 average of instance-max sigmoid and bag sigmoid. All scores share the same slide IDs, labels, and fixed feature-order shuffle `random_state=1`.

## HPC run instructions

SSH to Pioneer, then:

```bash
cd /mnt/vstor/CSE_BME_CCIPD/home/asg162/EBME463_FinalProject/karthikraj_baseline

module load Python/3.11.3-GCCcore-12.3.0
source ../.venv/bin/activate

GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519_github -o IdentitiesOnly=yes" git pull --ff-only origin main

python -m py_compile experiment_a_ensemble.py
mkdir -p logs
sbatch experiment_a_predict.slurm
```

Save the returned array job ID, then:

```bash
sacct -j YOUR_ARRAY_JOB_ID --format=JobID,State,Elapsed,ExitCode
grep -H -E 'PREFLIGHT|PASS|Traceback|Error' logs/ensemble_A_YOUR_ARRAY_JOB_ID_*.out
```

**Do not summarize until seeds 1–3 all report `COMPLETED 0:0` and `PASS`.** The `predict` stage requires 6 previously trained checkpoints per seed: CLAM, ABMIL, DSMIL for both `standalone` and `baseline`. If any is absent or its seed does not match its training history, the job fails before producing output. Each GPU task performs six models' validation inference, writing:

```text
experiment_a_outputs/
  validation_seed_1.csv
  validation_seed_1_manifest.json
  validation_seed_2.csv
  validation_seed_2_manifest.json
  validation_seed_3.csv
  validation_seed_3_manifest.json
```

Each seed CSV contains two families x 54 slides, with raw CLAM, ABMIL, DSMIL scores and slide labels. Checkpoints and recorded validation AUCs are preserved in each seed's manifest.

Once all three GPU tasks pass, run the CPU-only aggregation **on a login node** (the 324 prediction rows are small and do not require GPU):

```bash
python experiment_a_ensemble.py summarize
```

If Pioneer policy requires a batch job for even this small aggregation, submit `sbatch experiment_a_summarize.slurm` instead; it requests no GPU and uses the site's default partition.

Files created by summarization:

```text
experiment_a_outputs/
  validation_auc_results.csv     # 18 per-seed/per-family/per-method comparisons
  validation_summary.csv         # 6 means, sample SDs, paired differences
  paired_differences.csv         # explicit paired seed AUC deltas and CLAM-ABMIL correlation
```

The script fails if a slide is missing, reordered, duplicated, relabeled, or has invalid model scores. It refuses to overwrite any output files. To rerun, first archive the prior files; **do not blindly delete previous experiment results**.

## Interpretive limitations

- The validation set is the same set used for candidate selection and checkpoint selection. This is an **exploratory** comparison, *not* an unbiased estimate of ensemble generalization.
- The held-out test set has already been inspected in prior experiments; do not tune architecture inclusion or ensemble weights using that test.
- Only three matched seeds and 54 validation slides are included. Report per-seed changes and sample SD; do not claim significant improvement without independent evidence.
- Filename-level split independence does not establish patient-level independence.
- Do not use validation performance to retroactively justify choosing the two-model ensemble instead of the prespecified three-model ensemble without noting the exploratory choice.

## Code

- `experiment_a_ensemble.py`: two-stage, run-checked inference and aggregation.
- `experiment_a_predict.slurm`: 3-seed GPU array.
- `experiment_a_summarize.slurm`: optional CPU-only summary batch job.
