#!/usr/bin/env python3
"""Experiment A: validation-only, aligned, fixed equal-weight cross-architecture ensembles.

Run three independent GPU tasks:
  python experiment_a_ensemble.py predict --seed 1
  python experiment_a_ensemble.py predict --seed 2
  python experiment_a_ensemble.py predict --seed 3

Then CPU-only:
  python experiment_a_ensemble.py summarize

Two independent checkpoint families: strongest single checkpoint followed by matched
continuation ("standalone"), and original equal-weight merged baseline ("baseline").
Every family evaluates CLAM, ABMIL, CLAM+ABMIL, and CLAM+ABMIL+DSMIL.
Each ensemble uses unweighted arithmetic averaging of architecture probabilities.
No test split, test checkpoint, or test result is read by this program.
"""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path("datasets/camelyon16")
OUT = Path("experiment_a_outputs")
SEEDS = (1, 2, 3)
ARCHS = ("clam", "abmil", "dsmil")
FAMILIES = ("standalone", "baseline")
RUNS = {
    "clam": {
        1: "20261122212201489441",
        2: "20261123122950019167",
        3: "20261123122950435501",
    },
    "dsmil": {
        1: "20261123114547431845",
        2: "20261123123022171992",
        3: "20261123123007990510",
    },
}
METHODS = {
    "clam": ("CLAM", ("clam",)),
    "clam_abmil": ("CLAM+ABMIL", ("clam", "abmil")),
    "clam_abmil_dsmil": ("CLAM+ABMIL+DSMIL", ("clam", "abmil", "dsmil")),
}
PREDICTION_FIELDS = ["family", "seed", "slide_path", "label", *ARCHS]
RESULT_FIELDS = ["family", "seed", "method", "validation_auc",
                 "delta_vs_clam", "n_slides", "n_positive", "n_negative"]


def checkpoint_for(arch, family, seed):
    if arch == "abmil":
        folder = "standalone" if family == "standalone" else "baseline"
        return Path("weights/abmil_fusions") / f"seed_{seed}" / folder / "training" / "best_validation.pth"
    run_root = Path("weights") / arch / RUNS[arch][seed]
    if family == "standalone":
        return run_root / "standalone_results" / f"seed_{seed}" / "best_validation.pth"
    return run_root / "fusion_results" / f"lambda_0.0_seed_{seed}" / "best_validation.pth"


def load_split():
    import pandas as pd
    train = pd.read_csv(ROOT / "train.csv")
    val = pd.read_csv(ROOT / "val.csv")
    test = pd.read_csv(ROOT / "test.csv")
    for name, frame in (("train", train), ("val", val), ("test", test)):
        if not {"path", "label"}.issubset(frame.columns):
            raise ValueError(f"{name}.csv missing path or label column")
        paths = frame["path"].astype(str).tolist()
        if len(paths) != len(set(paths)) or len(paths) != len({Path(p).name for p in paths}):
            raise ValueError(f"Duplicate slide IDs in {name}.csv")
    names = lambda df: {Path(str(p)).name for p in df["path"]}
    if names(val) & (names(train) | names(test)) or names(train) & names(test):
        raise ValueError("Cross-split slide filename overlap")
    if len(train) != 215 or len(val) != 54 or len(test) != 128:
        raise ValueError(f"Unexpected split sizes: train={len(train)} val={len(val)} test={len(test)}")
    if val["path"].isna().any() or val["label"].isna().any():
        raise ValueError("Null validation values")
    labels = val["label"].astype(int)
    if not labels.eq(val["label"]).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("Validation labels must be binary and contain both classes")
    val = val.copy()
    val["label"] = labels
    return val


def check_histories(checkpoints, seed):
    records = {}
    for (family, arch), path in checkpoints.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        history = path.with_name("history.json")
        if not history.is_file():
            raise FileNotFoundError(history)
        data = json.loads(history.read_text())
        if int(data.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch in {history}")
        v = float(data["best_validation_auc"])
        if not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"Invalid validation AUC in {history}")
        if family == "standalone" and arch in ("clam", "dsmil"):
            selection = path.with_name("selection_record.json")
            if not selection.is_file():
                raise FileNotFoundError(selection)
            selection_info = json.loads(selection.read_text())
            if selection_info["architecture"] != arch or int(selection_info["seed"]) != seed:
                raise ValueError(f"Standalone provenance mismatch: {selection}")
        if family == "baseline" and arch == "abmil":
            metadata = path.parents[2] / "selection_metadata.json"
            if not metadata.is_file():
                raise FileNotFoundError(metadata)
            record = json.loads(metadata.read_text())
            if int(record["seed"]) != seed:
                raise ValueError(f"ABMIL provenance mismatch: {metadata}")
        records[f"{family}/{arch}"] = {
            "checkpoint": str(path), "history": str(history),
            "recorded_best_validation_auc": v,
        }
    return records


def write_csv_exclusive(path, fields, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predict(seed):
    import numpy as np
    import torch
    import train_repro_clam as clam
    import train_repro_dsmil as dsmil
    import evaluate_heldout as existing
    from train_fusion_dsmil import build_model

    output = OUT / f"validation_seed_{seed}.csv"
    manifest = OUT / f"validation_seed_{seed}_manifest.json"
    if output.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {manifest}")
    if not torch.cuda.is_available():
        raise RuntimeError("Prediction stage requires a CUDA GPU")

    val = load_split()
    np.random.seed(12345)
    random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    checkpoints = {(family, arch): checkpoint_for(arch, family, seed)
                   for family in FAMILIES for arch in ARCHS}
    records = check_histories(checkpoints, seed)
    print(f"EXPERIMENT A PREFLIGHT PASS seed={seed}: {len(val)} validation slides, "
          f"{len(checkpoints)} checkpoints; no test inference", flush=True)

    device = torch.device("cuda")
    rows = []
    with torch.inference_mode():
        for family in FAMILIES:
            scores = {}
            for arch in ARCHS:
                args = existing.make_args(arch)
                args.model = arch
                model = build_model(args) if arch == "dsmil" else clam.get_model(args)
                state = torch.load(checkpoints[family, arch], map_location="cpu",
                                   weights_only=True)
                model.load_state_dict(state, strict=True)
                model = model.to(device).eval()
                arch_scores = []
                for _, slide in val.iterrows():
                    # Use the original architecture's feature loader and the same
                    # fixed inference shuffle random_state used in prior heldout evals.
                    loader = dsmil if arch == "dsmil" else clam
                    label, features = loader.get_bag_feats(slide, args, random_state=1)
                    if int(np.asarray(label).reshape(-1)[0]) != int(slide["label"]):
                        raise ValueError(f"Feature/label mismatch: {slide['path']}")
                    features = np.asarray(features, dtype=np.float32)
                    if features.ndim != 2 or features.shape[1] != 512 or not np.isfinite(features).all():
                        raise ValueError(f"Invalid feature matrix for {slide['path']}: {features.shape}")
                    x = torch.from_numpy(features).to(device)
                    if arch == "dsmil":
                        instance, bag, _, _ = model(x)
                        maximum, _ = torch.max(instance, dim=0)
                        score = float(np.asarray(dsmil.activation(args, bag, maximum)).reshape(-1)[0])
                    else:
                        bag, _ = model(x)
                        score = float(torch.sigmoid(bag).reshape(-1)[0].item())
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid score {arch}: {slide['path']}")
                    arch_scores.append(score)
                scores[arch] = arch_scores
                print(f"seed={seed} family={family} arch={arch} predicted={len(arch_scores)}", flush=True)
                del model
                torch.cuda.empty_cache()

            for i, slide in val.reset_index(drop=True).iterrows():
                rows.append({
                    "family": family, "seed": seed, "slide_path": str(slide["path"]),
                    "label": int(slide["label"]),
                    **{arch: scores[arch][i] for arch in ARCHS},
                })

    if len(rows) != 2 * len(val):
        raise ValueError("Wrong prediction row count")
    OUT.mkdir(parents=True, exist_ok=True)
    # Artifact files are only created when all six checkpoint evaluations succeeded.
    write_csv_exclusive(output, PREDICTION_FIELDS, rows)
    manifest.write_text(json.dumps({
        "experiment": "A: fixed equally weighted prediction-level ensemble",
        "split": "validation_only", "seed": seed,
        "n_validation_slides": len(val), "model_family_count": len(FAMILIES),
        "checkpoint_records": records,
        "ensemble_weights": {"clam_abmil": [0.5, 0.5],
                             "clam_abmil_dsmil": [1 / 3] * 3},
        "test_predictions_generated": False,
    }, indent=2) + "\n")
    print(f"PASS seed={seed}: {len(rows)} aligned predictions; SAVED {output}", flush=True)


def summarize():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    dest = OUT / "validation_auc_results.csv"
    aggregate = OUT / "validation_summary.csv"
    differences = OUT / "paired_differences.csv"
    for path in (dest, aggregate, differences):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    val = load_split().reset_index(drop=True)
    expected = [(str(r["path"]), int(r["label"])) for _, r in val.iterrows()]
    detailed = []
    pairwise = []
    for seed in SEEDS:
        path = OUT / f"validation_seed_{seed}.csv"
        manifest = OUT / f"validation_seed_{seed}_manifest.json"
        if not path.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Missing complete seed {seed} predictions/manifest")
        info = json.loads(manifest.read_text())
        if info["seed"] != seed or info["split"] != "validation_only" or len(info["checkpoint_records"]) != 6:
            raise ValueError(f"Invalid experiment manifest {manifest}")
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 2 * len(expected):
            raise ValueError(f"Wrong prediction count: {path}")
        for family in FAMILIES:
            selected = [r for r in rows if r["family"] == family]
            if len(selected) != len(expected) or any(int(r["seed"]) != seed for r in selected):
                raise ValueError(f"Unexpected family/seed rows: {path}")
            keys = [(r["slide_path"], int(r["label"])) for r in selected]
            if keys != expected:
                raise ValueError(f"Slide ID/label alignment mismatch: {path} {family}")
            y = np.asarray([label for _, label in keys], dtype=int)
            scores = {}
            for arch in ARCHS:
                a = np.asarray([float(r[arch]) for r in selected])
                if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
                    raise ValueError(f"Nonfinite/out-of-bounds {arch}: {path}")
                scores[arch] = a
            aucs = {}
            for method_key, (method, included) in METHODS.items():
                predictions = np.mean(np.stack([scores[arch] for arch in included]), axis=0)
                aucs[method_key] = float(roc_auc_score(y, predictions))
                detailed.append({
                    "family": family, "seed": seed, "method": method,
                    "validation_auc": aucs[method_key],
                    "delta_vs_clam": aucs[method_key] - float(roc_auc_score(y, scores["clam"])),
                    "n_slides": len(y), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
                })
            pairwise.append({
                "family": family, "seed": seed,
                "clam_auc": aucs["clam"],
                "clam_abmil_auc": aucs["clam_abmil"],
                "clam_abmil_dsmil_auc": aucs["clam_abmil_dsmil"],
                "delta_two_vs_clam": aucs["clam_abmil"] - aucs["clam"],
                "delta_three_vs_clam": aucs["clam_abmil_dsmil"] - aucs["clam"],
                "delta_three_vs_two": aucs["clam_abmil_dsmil"] - aucs["clam_abmil"],
                "clam_abmil_score_correlation": float(np.corrcoef(scores["clam"], scores["abmil"])[0, 1])
                    if scores["clam"].std() > 0 and scores["abmil"].std() > 0 else "",
            })
            print(f"{family} seed={seed}: CLAM={aucs['clam']:.6f}; "
                  f"CLAM+ABMIL={aucs['clam_abmil']:.6f}; "
                  f"CLAM+ABMIL+DSMIL={aucs['clam_abmil_dsmil']:.6f}", flush=True)

    summary = []
    for family in FAMILIES:
        for method, _ in METHODS.values():
            matching = [r for r in detailed if r["family"] == family and r["method"] == method]
            vals = [float(r["validation_auc"]) for r in matching]
            deltas = [float(r["delta_vs_clam"]) for r in matching]
            summary.append({
                "family": family, "method": method, "n_seeds": len(vals),
                "mean_validation_auc": statistics.mean(vals),
                "sample_sd_validation_auc": statistics.stdev(vals),
                "mean_paired_delta_vs_clam": statistics.mean(deltas),
                "seed_min_validation_auc": min(vals),
                "seed_max_validation_auc": max(vals),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv_exclusive(dest, RESULT_FIELDS, detailed)
    write_csv_exclusive(aggregate, list(summary[0]), summary)
    write_csv_exclusive(differences, list(pairwise[0]), pairwise)
    print("PASS: all 3 seeds, both checkpoint families, 18 ensemble comparisons")
    print("SAVED:", dest, aggregate, differences, sep="\n")
    print("VALIDATION ONLY. Do not report these as held-out test AUCs.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.action == "predict":
        predict(args.seed)
    else:
        summarize()


if __name__ == "__main__":
    main()
