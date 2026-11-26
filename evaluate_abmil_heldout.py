#!/usr/bin/env python3
"""Held-out CAMELYON16 evaluation for the completed ABMIL experiment.

Evaluates the nine validation-selected ABMIL checkpoints (3 seeds x 3 methods)
on the original 128-slide test split. Test AUC is calculated once per checkpoint.
No test-set threshold tuning or model selection is performed.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_repro_clam as abmil_impl

RUNS = {
    1: "20261125152831551498",
    2: "20261125152841024449",
    3: "20261125154225498903",
}
METHODS = {
    "Standalone": ("standalone", None),
    "Baseline": ("baseline", 0.0),
    "Stability-aware": ("stability", 2.0),
}
ROOT = Path("datasets/camelyon16")
MODEL_ROOT = Path("weights/abmil_fusions")
OUTPUT_ROOT = Path("heldout_evaluation")
RESULTS = OUTPUT_ROOT / "abmil_test_auc_results.csv"
PREDICTIONS = OUTPUT_ROOT / "abmil_test_predictions.csv"


def make_args():
    return type("Args", (), {
        "model": "abmil",
        "dataset": "camelyon16",
        "num_classes": 1,
        "feats_size": 512,
        "q_n": 128,
        "non_linearity": 1.0,
        "dropout_patch": 0.0,
        "dropout_node": 0.0,
        "average": True,
        "init_type": "orthogonal",
        "gpu_index": [0],
    })()


def main():
    if RESULTS.exists() or PREDICTIONS.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ABMIL results: {RESULTS}, {PREDICTIONS}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for held-out evaluation")

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device = torch.device("cuda")

    test_df = pd.read_csv(ROOT / "test.csv")
    train_df = pd.read_csv(ROOT / "train.csv")
    val_df = pd.read_csv(ROOT / "val.csv")
    if len(test_df) != 128 or not {"path", "label"}.issubset(test_df.columns):
        raise ValueError("Expected the original 128-slide test split with path and label")
    names = lambda df: {Path(str(p)).name for p in df["path"]}
    if names(train_df) & names(test_df) or names(val_df) & names(test_df):
        raise ValueError("Test slides overlap with training or validation slides")
    labels = test_df["label"].astype(int).to_numpy()
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("The held-out test split must contain both classes")

    args = make_args()
    jobs = []
    for seed, run_id in RUNS.items():
        seed_dir = MODEL_ROOT / f"seed_{seed}"
        metadata_path = seed_dir / "selection_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        if int(metadata["seed"]) != seed or metadata["candidate_run"] != run_id:
            raise ValueError(f"Selection metadata mismatch: {metadata_path}")

        for display_method, (folder, lam) in METHODS.items():
            out_dir = seed_dir / folder / "training"
            checkpoint = out_dir / "best_validation.pth"
            history_path = out_dir / "history.json"
            if not checkpoint.is_file() or not history_path.is_file():
                raise FileNotFoundError(
                    f"Missing trained checkpoint/history for seed={seed}, method={display_method}"
                )
            history = json.loads(history_path.read_text())
            if int(history["seed"]) != seed:
                raise ValueError(f"Training-history seed mismatch: {history_path}")
            val_auc = float(history["best_validation_auc"])
            if not np.isfinite(val_auc) or not 0 <= val_auc <= 1:
                raise ValueError(f"Invalid validation AUC in {history_path}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = abmil_impl.get_model(args)
            model.load_state_dict(state, strict=True)
            jobs.append((seed, display_method, lam, checkpoint, val_auc, state))

    if len(jobs) != 9:
        raise ValueError(f"Expected 9 ABMIL checkpoints, found {len(jobs)}")
    print("ABMIL HELD-OUT PREFLIGHT PASSED", flush=True)
    print("Test slides:", len(test_df), flush=True)
    print("Positive slides:", int(labels.sum()), flush=True)
    print("Negative slides:", int((labels == 0).sum()), flush=True)
    print("Checkpoints:", len(jobs), flush=True)

    results = []
    prediction_rows = []
    for seed, method, lam, checkpoint, val_auc, state in jobs:
        model = abmil_impl.get_model(args)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        scores = []

        with torch.inference_mode():
            for _, row in test_df.iterrows():
                label, features = abmil_impl.get_bag_feats(
                    row, args, random_state=1
                )
                if int(np.asarray(label).item()) != int(row["label"]):
                    raise ValueError(f"Feature/label mismatch: {row['path']}")
                features = np.asarray(features, dtype=np.float32)
                if features.ndim != 2 or features.shape[1] != 512:
                    raise ValueError(f"Unexpected feature shape for {row['path']}: {features.shape}")
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite features: {row['path']}")
                x = torch.from_numpy(features).to(device)
                logits, attention = model(x)
                score = float(torch.sigmoid(logits).reshape(-1)[0].item())
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid prediction for {row['path']}: {score}")
                scores.append(score)
                prediction_rows.append({
                    "architecture": "ABMIL",
                    "seed": seed,
                    "method": method,
                    "slide_path": str(row["path"]),
                    "label": int(row["label"]),
                    "prediction_score": score,
                })

        auc = float(roc_auc_score(labels, scores))
        if not np.isfinite(auc):
            raise ValueError(f"Non-finite test AUC: seed={seed}, method={method}")
        results.append({
            "architecture": "ABMIL",
            "seed": seed,
            "method": method,
            "lambda": "" if lam is None else lam,
            "validation_auc": val_auc,
            "test_auc": auc,
            "test_samples": len(labels),
            "positive_slides": int(labels.sum()),
            "negative_slides": int((labels == 0).sum()),
            "checkpoint": str(checkpoint),
        })
        print(
            f"ABMIL seed={seed} method={method} "
            f"validation_AUC={val_auc:.6f} test_AUC={auc:.6f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    with RESULTS.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with PREDICTIONS.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    print("SAVED:", RESULTS, flush=True)
    print("SAVED:", PREDICTIONS, flush=True)
    print("ALL NINE ABMIL HELD-OUT EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
