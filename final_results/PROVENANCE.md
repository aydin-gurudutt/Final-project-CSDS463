# Extended Project 2 results: definitions and provenance

The previously published `all_results.csv`, `summary.csv`, and `test_auc_results.csv` are unchanged.

**performance_extended.csv** is derived from 12 published seed-level observations in `all_results.csv`. For each architecture and method, it includes test mean, **sample** standard deviation (n=3), minimum test AUC across seeds, and highest validation-selected AUC across seeds. These are derived statistics, not new model evaluations.

**training_job_times.csv** records 18 completed Pioneer SLURM jobs from archived November 23, 2026 terminal logs (all status COMPLETED, exit 0:0). Six runs generated five candidates each, and 12 runs continued baseline/stability-aware fusion. The table provides architecture, seed, job ID, stage, and elapsed seconds.

**timing_summary.csv** aggregates those elapsed durations by seed. Candidate generation is shared by both fusion methods, and must not be double-counted in the combined experimental GPU budget. The candidate+continuation elapsed time excludes offline merging, queue waiting, environment setup and held-out inference. The joint final evaluator (job 4307791, elapsed 00:05:30) evaluated all 12 models together and should not be assigned to one method.

**three_way_summary.csv** is intentionally absent until the matched standalone comparator has actually been trained and evaluated on Pioneer. Do not fill it with historical ABMIL results: they derive from another training/evaluation configuration.

## Running the missing non-fused comparator

The original proposal included a three-arm comparison: standalone MIL vs. baseline fusion vs. stability-aware fusion. The completed data only compare the latter two. To complete the missing arm using the existing five candidate checkpoint sets, from Pioneer in the project folder:

```bash
mkdir -p logs
for arch in clam dsmil; do
  for seed in 1 2 3; do
    sbatch --export=ALL,STAGE=train,ARCH="$arch",SEED="$seed" standalone_comparison.slurm
  done
done
```

After all six jobs finish successfully:

```bash
sbatch --export=ALL,STAGE=evaluate standalone_comparison.slurm
# After the evaluation job completes successfully:
python3 standalone_comparison.py summarize
```

This selects the single best weighted-mean-early-validation-AUC candidate (no parameter merging), continues it for the same two effective epochs as fusion, and evaluates the held-out test split on the same 128 slides. Those runs will generate new model checkpoints and AUC observations. **Their results are not present until the jobs actually run.**

The original ABMIL single-run AUC remains historical context; ABMIL was not modified or re-evaluated by the stability-aware CLAM/DSMIL extension.
