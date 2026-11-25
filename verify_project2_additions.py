"""Check extended descriptive AUC statistics and archived SLURM timing tables.

Only standard-library modules are used. This does NOT re-run GPU inference or
independently verify the original cluster's accounting records.
"""
import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "final_results"


def read(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: {actual} != {expected}")


def main():
    source = read(R / "all_results.csv")
    extended = read(R / "performance_extended.csv")
    jobs = read(R / "training_job_times.csv")
    timing = read(R / "timing_summary.csv")
    if tuple(map(len, [source, extended, jobs, timing])) != (12, 4, 18, 4):
        raise ValueError("Unexpected table lengths")
    methods = {(a, m) for a in ("CLAM", "DSMIL")
               for m in ("Baseline", "Stability-aware")}
    if {(r["architecture"], r["method"]) for r in extended} != methods:
        raise ValueError("Missing or unexpected AUC groups")
    for row in extended:
        arch, method = row["architecture"], row["method"]
        matching = [r for r in source if r["architecture"] == arch and r["method"] == method]
        if sorted(int(r["seed"]) for r in matching) != [1, 2, 3]:
            raise ValueError(f"Incomplete AUC seeds: {arch} {method}")
        val = [float(r["validation_auc"]) for r in matching]
        test = [float(r["test_auc"]) for r in matching]
        expected = {
            "validation_mean": statistics.mean(val),
            "validation_sample_sd": statistics.stdev(val),
            "best_validation_auc": max(val),
            "test_mean": statistics.mean(test),
            "test_sample_sd": statistics.stdev(test),
            "minimum_test_auc": min(test),
            "maximum_test_auc": max(test),
        }
        for field, value in expected.items():
            close(row[field], value, f"{arch} {method} {field}")

    job_by_key = {}
    seen_ids = set()
    for row in jobs:
        key = (row["architecture"], int(row["seed"]), row["stage"])
        if key in job_by_key or row["slurm_job_id"] in seen_ids:
            raise ValueError(f"Duplicate job: {key}")
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
            raise ValueError(f"Non-completed job: {key}")
        h, m, s = map(int, row["elapsed_hhmmss"].split(":"))
        elapsed = int(row["elapsed_seconds"])
        if elapsed <= 0 or elapsed != h * 3600 + m * 60 + s:
            raise ValueError(f"Invalid elapsed duration: {key}")
        job_by_key[key] = elapsed
        seen_ids.add(row["slurm_job_id"])
    for arch in ("CLAM", "DSMIL"):
        for seed in (1, 2, 3):
            for stage in ("candidate", "baseline_fusion", "stability_fusion"):
                if (arch, seed, stage) not in job_by_key:
                    raise ValueError(f"Missing job: {arch} {seed} {stage}")
    if {(r["architecture"], r["method"]) for r in timing} != methods:
        raise ValueError("Missing or unexpected timing groups")
    for row in timing:
        arch, method = row["architecture"], row["method"]
        stage = "baseline_fusion" if method == "Baseline" else "stability_fusion"
        cand = [job_by_key[arch, seed, "candidate"] for seed in (1, 2, 3)]
        fusion = [job_by_key[arch, seed, stage] for seed in (1, 2, 3)]
        expected = {
            "candidate_mean_seconds": statistics.mean(cand),
            "candidate_sd_seconds": statistics.stdev(cand),
            "fusion_mean_seconds": statistics.mean(fusion),
            "fusion_sd_seconds": statistics.stdev(fusion),
            "candidate_plus_fusion_mean_seconds":
                statistics.mean([a+b for a, b in zip(cand, fusion)]),
        }
        for field, value in expected.items():
            close(row[field], value, f"{arch} {method} {field}")
    print("PASS: extended AUC summaries and all 18 recorded job durations match their inputs.")
    print("NOTE: This does not independently verify SLURM logs or GPU inference.")


if __name__ == "__main__":
    main()
