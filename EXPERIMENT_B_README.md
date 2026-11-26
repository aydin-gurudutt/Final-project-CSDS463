# Experiment B — Greedy Model Soup

**Prepared, not yet submitted.**

For each CLAM, ABMIL and DSMIL architecture and three seeds (nine jobs), evaluate all five existing candidate checkpoints on the 54-slide validation set, initialize with highest validation AUC, and greedily add whichever remaining checkpoint strictly increases validation AUC after equal-weight arithmetic parameter averaging. Stop if none improves.

Model weights are averaged **only within the same architecture and seed**. This is arithmetic model soup, not TIES, and does not train new models. Existing checkpoints and Experiment A outputs are unchanged. No test data is used.

Run from project directory on Pioneer:

```bash
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519_github -o IdentitiesOnly=yes" git pull --ff-only origin main
module load Python/3.11.3-GCCcore-12.3.0
source ../.venv/bin/activate
python -m py_compile experiment_b_greedy_soup.py
mkdir -p logs
sbatch experiment_b_greedy_soup.slurm
```

Each task exclusively creates `experiment_b_outputs/<architecture>/seed_<n>/greedy_soup.pth`, `selection_trace.csv`, and `manifest.json`. Outputs are never overwritten. The GPU array allows two concurrent tasks, subject to cluster quotas. Check `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode` and `grep -H -E 'PREFLIGHT|SINGLE|TRY|ACCEPT|STOP|PASS|Traceback' logs/greedy_B_JOBID_*.out`.

**Interpretation:** The selected validation AUC is optimistically biased because the same validation split determines candidate selection. It does not establish held-out improvement. No held-out inference or post-merge finetuning is included.
