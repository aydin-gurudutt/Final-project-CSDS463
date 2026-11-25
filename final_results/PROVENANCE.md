# Project 2 experimental results: provenance

All reported AUCs are **held-out slide-level ROC AUC**, using the original 128-slide CAMELYON16 test split. Three random seeds were used for each of CLAM and DSMIL under three methods: **standalone** (one candidate, no merge), **baseline fusion** (top-three candidates), and **stability-aware fusion** (top-three candidates with early-validation stability penalty).

## Result tables

- [`final_results/all_results.csv`](all_results.csv): The 12 original baseline/stability-aware fusion test and validation AUCs; no original values were changed.
- [`final_results/summary.csv`](summary.csv) and [`final_results/performance_extended.csv`](performance_extended.csv): Fusion performance mean, sample SD, minimum test AUC, and highest validation-selected AUC across three seeds.
- [`heldout_evaluation/standalone_auc_results.csv`](../heldout_evaluation/standalone_auc_results.csv): Six **measured** standalone AUCs for CLAM/DSMIL, seeds 1–3. All test sample counts are 128. Each row includes the corresponding untracked checkpoint path.
- [`final_results/three_way_summary.csv`](three_way_summary.csv): Six-row comparison (2 architectures × 3 methods), including mean, sample SD, minimum held-out test AUC, and highest validation-selected AUC.

The standalone best validation AUC in the six-row table comes from each run's `history.json`; those private checkpoint histories are not tracked on GitHub, so only range and structural checks can be performed on that field without Pioneer.

## Training-time provenance

- [`final_results/training_job_times.csv`](training_job_times.csv) lists 18 completed candidate generation/fused continuation SLURM jobs from the archived November 23, 2026, Pioneer logs, each with exit code 0:0.
- [`final_results/timing_summary.csv`](timing_summary.csv) aggregates candidate generation and fusion continuation. Candidate training was **shared** between both fusion methods; do not double-count it as two separate training runs.
- [`final_results/standalone_job_times.csv`](standalone_job_times.csv) records the completed November 25, 2026 standalone training jobs 4313829–4313834 and held-out evaluation job 4313835, all exit code 0:0. The evaluation job ran all six models together; do not assign its full elapsed time to any individual model or compare it directly with a single fusion continuation job.

Job `Elapsed` represents recorded cluster wall time, **not** GPU-hours, a normalized benchmark, or a complete end-to-end experiment cost. These summaries exclude queue waiting, initial data preparation, separately executed offline checkpoint merging, and other unrecorded overhead.

## Reproduction and verification

Source: `standalone_comparison.py` selects each architecture's single best weighted-mean early-validation-AUC candidate from five candidates, trains two more effective epochs using the corresponding fusion continuation function without merging weights, and evaluates using the same test-split and AUC scoring logic as the fused models. No ABMIL model was modified or measured within this matched experiment.

All seven new SLURM jobs completed successfully, and the summary was generated on Pioneer. Re-running `bash submit_standalone_all.sh` is **not necessary** and must not overwrite the completed checkpoints or published CSVs.

To verify all published statistics **without a GPU**, run:

```bash
python3 verify_published_results.py
python3 verify_project2_additions.py
python3 verify_three_way.py
```

These consistency checks cannot independently reproduce model training, guarantee test/validation design independence, audit patient-level disjointness, or validate withheld checkpoint histories. No significance test or broad reliability benefit is claimed from three seeds.
