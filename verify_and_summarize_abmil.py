#!/usr/bin/env python3
"""Verify ABMIL test predictions against published per-model AUCs and summarize.

Run on Pioneer after evaluate_abmil_heldout.py successfully produced both CSVs:
    python verify_and_summarize_abmil.py

Requires pandas? No. Requires scikit-learn for independent roc_auc_score.
Reads both ABMIL CSVs without changing them; creates the summary only
after checking all nine AUCs against 1,152 recorded slide predictions.
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "heldout_evaluation" / "abmil_test_auc_results.csv"
PREDICTIONS = ROOT / "heldout_evaluation" / "abmil_test_predictions.csv"
SUMMARY = ROOT / "final_results" / "abmil_three_way_summary.csv"
METHODS = ("Standalone", "Baseline", "Stability-aware")
EXPECTED = {(seed, method) for seed in (1, 2, 3) for method in METHODS}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{label}: reported={a}, recomputed={b}")


def main():
    if SUMMARY.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing summary: {SUMMARY}"
        )
    raw = read_csv(RESULTS)
    predictions = read_csv(PREDICTIONS)
    if len(raw) != 9 or len(predictions) != 9 * 128:
        raise ValueError(
            f"Expected 9 AUC rows and 1152 prediction rows; "
            f"found {len(raw)} and {len(predictions)}"
        )

    runs = {}
    for r in raw:
        key = int(r["seed"]), r["method"]
        if key not in EXPECTED or key in runs or r["architecture"] != "ABMIL":
            raise ValueError(f"Unexpected/duplicate result key: {key}")
        if (int(r["test_samples"]), int(r["positive_slides"]),
                int(r["negative_slides"])) != (128, 46, 82):
            raise ValueError(f"Invalid test-set count: {key}")
        lam = r["lambda"]
        if key[1] == "Standalone":
            if lam != "":
                raise ValueError(f"Standalone lambda should be blank: {lam}")
        else:
            close(lam, 0 if key[1] == "Baseline" else 2,
                  f"Penalty for {key}")
        auc = float(r["test_auc"])
        val = float(r["validation_auc"])
        if not (math.isfinite(auc) and math.isfinite(val)
                and 0 <= auc <= 1 and 0 <= val <= 1):
            raise ValueError(f"Invalid AUC: {key}")
        runs[key] = r
    if set(runs) != EXPECTED:
        raise ValueError("Incomplete ABMIL evaluation results")

    grouped = defaultdict(list)
    canonical = None
    for p in predictions:
        key = int(p["seed"]), p["method"]
        if key not in EXPECTED or p["architecture"] != "ABMIL":
            raise ValueError(f"Unexpected prediction key: {key}")
        path = p["slide_path"]
        label = int(p["label"])
        score = float(p["prediction_score"])
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid prediction: {key}, {path}")
        grouped[key].append((path, label, score))

    for key in sorted(EXPECTED):
        values = grouped[key]
        if len(values) != 128:
            raise ValueError(f"Expected 128 predictions for {key}, found {len(values)}")
        slide_labels = {path: label for path, label, _ in values}
        if len(slide_labels) != 128:
            raise ValueError(f"Duplicate slide path in predictions for {key}")
        if canonical is None:
            canonical = slide_labels
        elif slide_labels != canonical:
            raise ValueError(f"Slide/label mismatch between ABMIL models for {key}")
        labels = [label for _, label, _ in values]
        if sum(labels) != 46:
            raise ValueError(f"Unexpected positive count for {key}")
        auc = roc_auc_score(labels, [score for _, _, score in values])
        close(auc, runs[key]["test_auc"], f"Test ROC AUC for {key}")
        print(f"PASS seed={key[0]} method={key[1]} test_AUC={auc:.12f}")

    summaries = []
    for method in METHODS:
        selected = [runs[seed, method] for seed in (1, 2, 3)]
        test = [float(row["test_auc"]) for row in selected]
        validation = [float(row["validation_auc"]) for row in selected]
        summaries.append({
            "architecture": "ABMIL",
            "method": method,
            "n_seeds": 3,
            "test_mean": statistics.mean(test),
            "test_sample_sd": statistics.stdev(test),
            "test_minimum": min(test),
            "test_maximum": max(test),
            "validation_mean": statistics.mean(validation),
            "best_validation_auc": max(validation),
        })
        print(
            f"ABMIL {method}: mean={statistics.mean(test):.6f}, "
            f"sample_SD={statistics.stdev(test):.6f}, min={min(test):.6f}"
        )

    SUMMARY.parent.mkdir(exist_ok=True)
    with SUMMARY.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print("SAVED:", SUMMARY)
    print("PASS: all nine ABMIL AUCs independently reproduced from 1,152 scores")


if __name__ == "__main__":
    main()
