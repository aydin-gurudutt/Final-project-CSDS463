"""Matched non-fused MIL experiment. Requires existing GPU-side candidate checkpoints.

Usage:
  python standalone_comparison.py train --architecture clam --seed 1
  python standalone_comparison.py evaluate
  python standalone_comparison.py summarize

Do not report standalone AUC until evaluate has run successfully.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from evaluate_heldout import RUNS


def run_dir(architecture, seed):
    return Path("weights") / architecture / RUNS[architecture][seed]


def output_dir(architecture, seed):
    return run_dir(architecture, seed) / "standalone_results" / f"seed_{seed}"


def select_candidate(architecture, seed):
    root = run_dir(architecture, seed)
    path = root / "candidate_histories.npz"
    with np.load(path, allow_pickle=False) as archive:
        early = np.asarray(archive["early_aucs"], dtype=float)
        source_seed = int(archive["seed"])
    if early.shape != (5, 5) or source_seed != seed or not np.isfinite(early).all():
        raise ValueError(f"Invalid candidate history: {path}")
    weights = np.array([1, 1, 1, 1, 1] if architecture == "clam"
                       else [0, 0, 0, 1, 1], dtype=float)
    weights /= weights.sum()
    scores = np.average(early, axis=1, weights=weights)
    candidate = int(np.argsort(scores)[-1])
    checkpoint = root / "candidates" / f"candidate_{candidate}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return candidate, float(scores[candidate]), checkpoint


def train(architecture, seed):
    candidate, validation_auc, checkpoint = select_candidate(architecture, seed)
    dest = output_dir(architecture, seed)
    if dest.exists():
        raise FileExistsError(dest)
    print(f"Selected {architecture} seed={seed} candidate={candidate} "
          f"early_validation_auc={validation_auc:.6f}", flush=True)
    # Reuse the original architecture's continuation implementation,
    # but initialize from a SINGLE candidate rather than a merged checkpoint.
    command = [sys.executable, f"train_fusion_{architecture}.py",
               "--merged_checkpoint", str(checkpoint),
               "--output_dir", str(dest),
               "--seed", str(seed), "--model", architecture,
               "--dataset", "camelyon16", "--num_classes", "1",
               "--feats_size", "512", "--num_epochs", "3",
               "--lr", "0.0002", "--weight_decay", "0.005",
               "--gpu_index", "0"]
    subprocess.run(command, check=True)
    if not (dest / "best_validation.pth").is_file():
        raise FileNotFoundError(dest / "best_validation.pth")
    record = {"method": "Standalone single best candidate; no parameter merge",
              "architecture": architecture, "seed": seed,
              "candidate_index": candidate, "early_mean_validation_auc": validation_auc,
              "candidate_checkpoint": str(checkpoint), "continuation_epochs": 2}
    with (dest / "selection_record.json").open("x") as f:
        json.dump(record, f, indent=2)


def evaluate():
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    import evaluate_heldout as original

    result = Path("heldout_evaluation/standalone_auc_results.csv")
    if result.exists():
        raise FileExistsError(result)
    root = Path("datasets/camelyon16")
    train_df = pd.read_csv(root / "train.csv")
    val_df = pd.read_csv(root / "val.csv")
    test_df = pd.read_csv(root / "test.csv")
    if len(test_df) != 128 or not {"path", "label"}.issubset(test_df.columns):
        raise ValueError("Expected the original 128-slide held-out split")
    names = lambda df: {Path(str(p)).name for p in df["path"]}
    if names(test_df) & (names(train_df) | names(val_df)):
        raise ValueError("Test filename overlap with train or validation")
    labels = test_df["label"].astype(int).to_numpy()
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Test labels must contain both binary classes")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    jobs = []
    for arch, seeds in RUNS.items():
        for seed in seeds:
            folder = output_dir(arch, seed)
            checkpoint = folder / "best_validation.pth"
            history = folder / "history.json"
            selection = folder / "selection_record.json"
            for file in (checkpoint, history, selection):
                if not file.is_file():
                    raise FileNotFoundError(file)
            with selection.open() as file:
                meta = json.load(file)
            if meta["architecture"] != arch or int(meta["seed"]) != seed:
                raise ValueError(f"Bad selection metadata: {selection}")
            jobs.append((arch, seed, checkpoint))
    if len(jobs) != 6:
        raise ValueError("Exactly six standalone runs required")

    device = torch.device("cuda")
    torch.manual_seed(12345)
    np.random.seed(12345)
    results = []
    for arch, seed, checkpoint in jobs:
        args = original.make_args(arch)
        baseline = original.clam if arch == "clam" else original.dsmil
        model = baseline.get_model(args) if arch == "clam" else original.build_model(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        predictions = []
        with torch.inference_mode():
            for _, row in test_df.iterrows():
                label, features = baseline.get_bag_feats(row, args, random_state=1)
                if int(np.asarray(label).item()) != int(row["label"]):
                    raise ValueError("Feature/slide label mismatch")
                features = np.asarray(features, dtype=np.float32)
                if features.ndim != 2 or features.shape[1] != 512 or not np.isfinite(features).all():
                    raise ValueError("Unexpected feature bag shape/content")
                x = torch.from_numpy(features).to(device)
                if arch == "clam":
                    bag_prediction, _ = model(x)
                    score = torch.sigmoid(bag_prediction).reshape(-1)[0].item()
                else:
                    ins_prediction, bag_prediction, _, _ = model(x)
                    max_prediction, _ = torch.max(ins_prediction, dim=0)
                    activated = original.dsmil.activation(args, bag_prediction, max_prediction)
                    score = float(np.asarray(activated).reshape(-1)[0])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid slide score")
                predictions.append(score)
        auc = float(roc_auc_score(labels, predictions))
        results.append({"architecture": arch, "seed": seed,
                        "test_auc": auc, "test_samples": len(labels),
                        "checkpoint": str(checkpoint)})
        print(f"{arch.upper()} seed={seed} standalone held-out AUC={auc:.6f}", flush=True)
        del model
        torch.cuda.empty_cache()
    result.parent.mkdir(exist_ok=True)
    with result.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("Saved", result)


def summarize():
    import statistics
    source = Path("heldout_evaluation/standalone_auc_results.csv")
    if not source.is_file():
        raise FileNotFoundError("Complete train and evaluate stages before summarizing")
    with source.open(newline="") as f:
        standalone = list(csv.DictReader(f))
    if len(standalone) != 6:
        raise ValueError("Expected six standalone AUC measurements")
    with Path("final_results/all_results.csv").open(newline="") as f:
        fused = list(csv.DictReader(f))
    out = Path("final_results/three_way_summary.csv")
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for arch in ("CLAM", "DSMIL"):
        singles = [r for r in standalone if r["architecture"].upper() == arch]
        if {int(r["seed"]) for r in singles} != {1, 2, 3}:
            raise ValueError(f"Missing standalone seed(s) for {arch}")
        for method in ("Standalone", "Baseline", "Stability-aware"):
            if method == "Standalone":
                test = [float(r["test_auc"]) for r in singles]
                validation = []
                for r in singles:
                    with (output_dir(arch.lower(), int(r["seed"])) / "history.json").open() as f:
                        validation.append(float(json.load(f)["best_validation_auc"]))
            else:
                sub = [r for r in fused if r["architecture"] == arch and r["method"] == method]
                if len(sub) != 3:
                    raise ValueError(f"Missing fusion results: {arch} {method}")
                test = [float(r["test_auc"]) for r in sub]
                validation = [float(r["validation_auc"]) for r in sub]
            rows.append({"architecture": arch, "method": method, "n_seeds": 3,
                         "test_mean": statistics.mean(test),
                         "test_sample_sd": statistics.stdev(test),
                         "test_minimum": min(test),
                         "best_validation_auc": max(validation)})
    with out.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved measured three-way comparison:", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "summarize"))
    parser.add_argument("--architecture", choices=("clam", "dsmil"))
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.stage == "train":
        if args.architecture is None or args.seed is None:
            parser.error("train requires --architecture and --seed")
        train(args.architecture, args.seed)
    elif args.stage == "evaluate":
        evaluate()
    else:
        summarize()


if __name__ == "__main__":
    main()
