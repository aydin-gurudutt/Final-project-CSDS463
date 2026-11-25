"""Verify published three-way MIL results using Python's standard library.

Checks all 6 standalone held-out ROC-AUC records against the six-row summary,
cross-checks the 12 historical fusion AUC records, and validates the 7 new
SLURM durations. Does not re-run inference or independently verify training.
Standalone best-validation AUCs require ignored history.json files and
therefore cannot be independently recomputed from the public CSVs.
"""
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def records(relative_path):
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def equal(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11):
        raise AssertionError(f"{label}: recorded {actual}, expected {expected}")


def main():
    standalone = records("heldout_evaluation/standalone_auc_results.csv")
    fused = records("final_results/all_results.csv")
    summary = records("final_results/three_way_summary.csv")
    jobs = records("final_results/standalone_job_times.csv")
    if (len(standalone), len(fused), len(summary), len(jobs)) != (6, 12, 6, 7):
        raise AssertionError("Unexpected row counts in source, summary, or timing tables")

    unique_standalone = set()
    for row in standalone:
        arch = row["architecture"].upper()
        seed = int(row["seed"])
        key = arch, seed
        if arch not in ("CLAM", "DSMIL") or seed not in (1, 2, 3):
            raise AssertionError(f"Invalid standalone key: {key}")
        if key in unique_standalone:
            raise AssertionError(f"Duplicate standalone evaluation: {key}")
        unique_standalone.add(key)
        auc = float(row["test_auc"])
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise AssertionError(f"Invalid standalone AUC: {key}")
        if int(row["test_samples"]) != 128:
            raise AssertionError(f"Unexpected held-out test size: {key}")
        checkpoint = row["checkpoint"]
        if not checkpoint.startswith("weights/" + arch.lower() + "/"):
            raise AssertionError(f"Unexpected checkpoint architecture: {key}")
        if not checkpoint.endswith(f"/standalone_results/seed_{seed}/best_validation.pth"):
            raise AssertionError(f"Unexpected checkpoint seed: {key}")
    expected_keys = {(arch, seed) for arch in ("CLAM", "DSMIL") for seed in (1, 2, 3)}
    if unique_standalone != expected_keys:
        raise AssertionError("Incomplete standalone seed coverage")

    expected_summary = {(arch, method) for arch in ("CLAM", "DSMIL")
                        for method in ("Standalone", "Baseline", "Stability-aware")}
    seen_summary = set()
    for row in summary:
        arch, method = row["architecture"], row["method"]
        key = arch, method
        if key not in expected_summary or key in seen_summary:
            raise AssertionError(f"Unexpected or duplicated summary: {key}")
        seen_summary.add(key)
        if int(row["n_seeds"]) != 3:
            raise AssertionError(f"Incorrect summary sample count: {key}")
        if method == "Standalone":
            test = [float(r["test_auc"]) for r in standalone
                    if r["architecture"].upper() == arch]
            # The standalone validation maximum comes from training history.json,
            # which is intentionally not tracked; range-check only.
            best_val = float(row["best_validation_auc"])
            if not math.isfinite(best_val) or not 0 <= best_val <= 1:
                raise AssertionError(f"Invalid standalone best validation AUC: {key}")
        else:
            subset = [r for r in fused
                      if r["architecture"] == arch and r["method"] == method]
            if sorted(int(r["seed"]) for r in subset) != [1, 2, 3]:
                raise AssertionError(f"Incomplete fusion seeds: {key}")
            test = [float(r["test_auc"]) for r in subset]
            equal(row["best_validation_auc"],
                  max(float(r["validation_auc"]) for r in subset),
                  f"{key}: best validation AUC")
        equal(row["test_mean"], statistics.mean(test), f"{key}: mean")
        equal(row["test_sample_sd"], statistics.stdev(test), f"{key}: sample sd")
        equal(row["test_minimum"], min(test), f"{key}: min")
    if seen_summary != expected_summary:
        raise AssertionError("Missing architecture/method combinations")

    seen_jobs = set()
    for row in jobs:
        arch, stage, job_id = row["architecture"], row["stage"], row["slurm_job_id"]
        seed = int(row["seed"]) if row["seed"] else None
        if job_id in seen_jobs or row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
            raise AssertionError(f"Invalid or duplicate job: {job_id}")
        seen_jobs.add(job_id)
        hh, mm, ss = map(int, row["elapsed_hhmmss"].split(":"))
        if int(row["elapsed_seconds"]) != (hh * 3600 + mm * 60 + ss):
            raise AssertionError(f"Elapsed-time mismatch: {job_id}")
        if stage == "standalone_train":
            if (arch, seed) not in expected_keys:
                raise AssertionError(f"Unexpected training job: {job_id}")
        elif stage == "standalone_eval_all":
            if (arch, seed) != ("ALL", None):
                raise AssertionError(f"Unexpected shared evaluation job: {job_id}")
        else:
            raise AssertionError(f"Unknown job stage: {stage}")
    if seen_jobs != {str(i) for i in range(4313829, 4313836)}:
        raise AssertionError("Missing expected seven HPC job IDs")

    print("PASS: six standalone AUCs, 12 fusion AUCs, six three-way summaries,")
    print("      and seven completed SLURM elapsed durations are internally consistent.")
    print("NOTE: Does not independently validate GPU inference or standalone histories.")


if __name__ == "__main__":
    main()
