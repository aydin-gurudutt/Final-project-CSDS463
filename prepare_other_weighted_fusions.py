#!/usr/bin/env python3
"""Prepare matched AUC-weighted and stability-weighted CLAM/DSMIL fusion checkpoints.

Use: python prepare_other_weighted_fusions.py --architecture clam
     python prepare_other_weighted_fusions.py --architecture dsmil

Preflight reconstructs existing equal-weight TIES-style baseline for each seed,
using the original architecture-specific magnitude trimming fraction (CLAM 1,
DSMIL 0.5), and refuses to save any new checkpoints if verification fails.
The three candidates and trimming fraction are identical between new arms;
only the candidate reliability weights differ. NO TEST DATA ACCESSED.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
PENALTY = {"clam": 0.5, "dsmil": 5.0}
KEEP_FRACTION = {"clam": 1.0, "dsmil": 0.5}
CHECKPOINT_WEIGHTS = {
    "clam": np.array([1, 1, 1, 1, 1], dtype=float),
    "dsmil": np.array([0, 0, 0, 1, 1], dtype=float),
}
TEMPERATURE = 0.02  # Fixed exploratory value; not selected using held-out test.

def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)

def max_distance(a, b):
    if set(a) != set(b):
        raise ValueError("State dict keys differ")
    vals = []
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError(f"Shape mismatch for {key}")
        if a[key].is_floating_point():
            vals.append(float((a[key] - b[key]).abs().max()))
        elif not torch.equal(a[key], b[key]):
            raise ValueError(f"Non-floating state differs: {key}")
    return max(vals) if vals else 0.0

def softmax(scores):
    z = np.asarray(scores, dtype=float) / TEMPERATURE
    z -= z.max()
    w = np.exp(z)
    w /= w.sum()
    if not np.isfinite(w).all() or not (w > 0).all():
        raise ValueError("Invalid fusion weights")
    return w

def weighted_ties(states, initial, weights, keep):
    """Weighted version of existing sign-election/disjoint task-vector merge.

    Equal weights reproduce the original merge with architecture-specific trimming.
    """
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Fusion weights do not sum to one")
    result = {}
    for key, start in initial.items():
        if any(key not in s or s[key].shape != start.shape for s in states):
            raise ValueError(f"Incompatible candidate tensor {key}")
        if not start.is_floating_point():
            if any(not torch.equal(s[key], start) for s in states):
                raise ValueError(f"Nonfloating candidate mismatch: {key}")
            result[key] = start.clone()
            continue
        updates = []
        for s in states:
            d = s[key] - start
            if not torch.isfinite(d).all():
                raise ValueError(f"Nonfinite state: {key}")
            if keep < 1.0:
                # Same per-tensor, inclusive quantile rule as original TIES code.
                threshold = torch.quantile(d.reshape(-1).abs(), 1.0 - keep)
                d = torch.where(d.abs() >= threshold, d, torch.zeros_like(d))
            updates.append(d)
        delta = torch.stack(updates)
        w = torch.as_tensor(weights, dtype=delta.dtype).reshape(
            (len(states),) + (1,) * start.ndim
        )
        elected = torch.sign((delta * w).sum(dim=0))
        same = (torch.sign(delta) == elected) & (elected != 0)
        numerator = (delta * same * w).sum(dim=0)
        denominator = (same * w).sum(dim=0)
        update = torch.where(
            denominator > 0, numerator / denominator.clamp_min(1e-30),
            torch.zeros_like(numerator)
        )
        result[key] = start + update
        if not torch.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite merged state: {key}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("clam", "dsmil"), required=True)
    args = parser.parse_args()
    arch = args.architecture
    target = Path("weights") / f"{arch}_weighted_fusions"
    if target.exists():
        raise FileExistsError(f"Refusing overwrite: {target}")

    prepared = []
    for seed, run_id in RUNS[arch].items():
        source = Path("weights") / arch / run_id
        with np.load(source / "candidate_histories.npz", allow_pickle=False) as data:
            early = np.asarray(data["early_aucs"], dtype=float)
            if int(data["seed"]) != seed or early.shape != (5, 5):
                raise ValueError(f"Wrong source candidate histories: {source}")
        if not np.isfinite(early).all() or np.any((early < 0) | (early > 1)):
            raise ValueError(f"Invalid candidate AUC history: {source}")
        temporal = CHECKPOINT_WEIGHTS[arch]
        temporal = temporal / temporal.sum()
        means = np.average(early, weights=temporal, axis=1)
        stds = np.sqrt(np.average(
            (early - means[:, None]) ** 2, weights=temporal, axis=1
        ))
        selected = np.argsort(means)[-3:].tolist()
        init = load(source / "initial_weights.pth")
        states = [load(source / "candidates" / f"candidate_{i}.pth")
                  for i in selected]

        keep = KEEP_FRACTION[arch]
        equal = weighted_ties(states, init, [1 / 3] * 3, keep)
        baseline = load(source / "offline_merges" / "merged_lambda_0.0.pth")
        error = max_distance(equal, baseline)
        if error > 1e-6:
            raise ValueError(
                f"{arch} seed {seed} equal-weight reconstruction mismatch "
                f"({error:.8g}); STOP: inspect original merge before training"
            )

        auc_weights = softmax(means[selected])
        stability_weights = softmax(means[selected] - PENALTY[arch] * stds[selected])
        auc_state = weighted_ties(states, init, auc_weights, keep)
        stable_state = weighted_ties(states, init, stability_weights, keep)
        between = max_distance(auc_state, stable_state)
        if between <= 0:
            raise ValueError(f"No weight-dependent parameter change: {arch} seed {seed}")
        info = {
            "architecture": arch.upper(),
            "seed": seed,
            "source_run": run_id,
            "candidate_indices": selected,
            "mean_validation_auc": means[selected].tolist(),
            "std_validation_auc": stds[selected].tolist(),
            "temporal_checkpoint_weights": temporal.tolist(),
            "stability_penalty": PENALTY[arch],
            "temperature": TEMPERATURE,
            "trim_keep_fraction": keep,
            "auc_fusion_weights": auc_weights.tolist(),
            "stability_fusion_weights": stability_weights.tolist(),
            "max_difference_between_new_arms": between,
            "equal_weight_reconstruction_error": error,
            "test_data_accessed": False,
        }
        print(
            f"{arch.upper()} seed={seed} candidates={selected} "
            f"AUC_weights={np.round(auc_weights, 4).tolist()} "
            f"stability_weights={np.round(stability_weights, 4).tolist()} "
            f"between_arms={between:.8g} baseline_error={error:.2e}",
            flush=True,
        )
        prepared.append((seed, info, auc_state, stable_state))

    # Save only after all three baseline checks and both arms pass preflight.
    for seed, info, auc_state, stable_state in prepared:
        root = target / f"seed_{seed}"
        for method, state in (
            ("auc_weighted", auc_state),
            ("stability_weighted", stable_state),
        ):
            dest = root / method
            dest.mkdir(parents=True, exist_ok=False)
            torch.save(state, dest / "merged_initial.pth")
        (root / "merge_metadata.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8"
        )
    print(f"PASS: {arch.upper()} all 3 seeds, 6 distinct weighted initializations")
    print(f"SAVED: {target}")
    print("NO TEST DATA ACCESSED")

if __name__ == "__main__":
    main()
