"""Check published MIL result tables without requiring model weights or GPUs.

Run from anywhere: python verify_published_results.py
"""
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "final_results"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def same(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def main():
    records = read_csv(RESULTS / "all_results.csv")
    summary = read_csv(RESULTS / "summary.csv")
    heldout = read_csv(ROOT / "heldout_evaluation" / "test_auc_results.csv")

    if len(records) != 12 or len(summary) != 4 or len(heldout) != 12:
        raise ValueError("Expected 12 seed results, four summaries, and 12 held-out evaluations")

    expected = {
        ("CLAM", "Baseline"): 0.0,
        ("CLAM", "Stability-aware"): 0.5,
        ("DSMIL", "Baseline"): 0.0,
        ("DSMIL", "Stability-aware"): 5.0,
    }

    grouped = defaultdict(list)
    indexed = {}
    for row in records:
        architecture, method = row["architecture"], row["method"]
        seed = int(row["seed"])
        key = (architecture, method, seed)
        if (architecture, method) not in expected or seed not in (1, 2, 3):
            raise ValueError(f"Unexpected experiment combination: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate result: {key}")
        same(row["lambda"], expected[architecture, method], str(key) + " lambda")
        for metric in ("validation_auc", "test_auc"):
            value = float(row[metric])
            if not (0.0 <= value <= 1.0 and math.isfinite(value)):
                raise ValueError(f"Invalid {metric} for {key}")
        indexed[key] = row
        grouped[architecture, method].append(row)

    for pair in expected:
        if {int(row["seed"]) for row in grouped[pair]} != {1, 2, 3}:
            raise ValueError(f"Missing experimental seeds for {pair}")

    seen_summaries = set()
    for row in summary:
        pair = (row["architecture"], row["method"])
        if pair not in expected or pair in seen_summaries:
            raise ValueError(f"Unexpected or duplicate summary: {pair}")
        seen_summaries.add(pair)
        for source, output in (
            ("validation_auc", "validation"),
            ("test_auc", "test"),
        ):
            values = [float(item[source]) for item in grouped[pair]]
            same(row[output + "_mean"], statistics.mean(values), str(pair) + " mean")
            same(row[output + "_sd"], statistics.stdev(values), str(pair) + " sample sd")
    if seen_summaries != set(expected):
        raise ValueError("Missing architecture/method summaries")

    seen_heldout = set()
    for row in heldout:
        architecture = row["architecture"].upper()
        seed = int(row["seed"])
        penalty = float(row["lambda"])
        methods = [method for (arch, method), lam in expected.items()
                   if arch == architecture and math.isclose(lam, penalty)]
        if len(methods) != 1:
            raise ValueError(f"Unexpected held-out condition: {row}")
        key = (architecture, methods[0], seed)
        if key in seen_heldout:
            raise ValueError(f"Duplicate held-out result: {key}")
        seen_heldout.add(key)
        if int(row["test_samples"]) != 128:
            raise ValueError(f"Unexpected held-out sample count: {key}")
        same(row["test_auc"], indexed[key]["test_auc"], str(key) + " held-out AUC")

    if seen_heldout != set(indexed):
        raise ValueError("Missing held-out evaluation records")

    print("PASS: 12 matched seed results, four mean/sample-SD summaries,")
    print("      12 held-out AUCs, and 128 slides per held-out evaluation.")
    print("NOTE: This checks published CSV consistency, not model training or split integrity.")


if __name__ == "__main__":
    main()
