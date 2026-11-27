import json
from pathlib import Path

import numpy as np
import torch
import train_candidates_clam as baseline

RUNS = {
    1: "20261125152831551498",
    2: "20261125152841024449",
    3: "20261125154225498903",
}

STABILITY_LAMBDA = 2.0
OUTPUT = Path("weights/abmil_fusions")

def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)

def merge_candidates(states, initial):
    merged = {}

    for key, initial_tensor in initial.items():
        candidate_tensors = [state[key] for state in states]

        if initial_tensor.is_floating_point():
            merged[key] = baseline.ties_merge_key(
                candidate_tensors,
                initial_tensor,
                k=1.0,
                lam=1.0
            )
        else:
            merged[key] = initial_tensor.clone()

    return merged

for seed, run_id in RUNS.items():

    root = Path("weights/abmil") / run_id
    output_dir = OUTPUT / f"seed_{seed}"

    assert root.exists()
    assert not output_dir.exists(), output_dir

    output_dir.mkdir(parents=True)

    with np.load(root / "candidate_histories.npz") as data:
        aucs = np.asarray(data["early_aucs"], dtype=float)

    assert aucs.shape == (5, 5)
    assert np.isfinite(aucs).all()

    initial = load_state(root / "initial_weights.pth")

    candidates = [
        load_state(root / "candidates" / f"candidate_{i}.pth")
        for i in range(5)
    ]

    for state in candidates:
        assert state.keys() == initial.keys()
        for key in initial:
            assert state[key].shape == initial[key].shape

    means = aucs.mean(axis=1)
    stds = aucs.std(axis=1, ddof=0)

    selections = {}

    for method, lam in [
        ("baseline", 0.0),
        ("stability", STABILITY_LAMBDA)
    ]:

        scores = means - lam * stds
        selected = np.argsort(scores)[::-1][:3].tolist()

        selected_states = [candidates[i] for i in selected]

        merged = merge_candidates(selected_states, initial)

        method_dir = output_dir / method
        method_dir.mkdir()

        checkpoint_path = method_dir / "merged_initial.pth"
        torch.save(merged, checkpoint_path)

        selections[method] = {
            "lambda": lam,
            "selected_candidates": selected,
            "scores": scores.tolist(),
            "merged_checkpoint": str(checkpoint_path)
        }

        print(
            f"Seed {seed} | {method} | "
            f"Selected: {selected}"
        )

    # Standalone uses the highest mean-AUC candidate.
    best_candidate = int(np.argmax(means))

    standalone_dir = output_dir / "standalone"
    standalone_dir.mkdir()

    standalone_path = standalone_dir / "selected_candidate.pth"

    torch.save(candidates[best_candidate], standalone_path)

    selections["standalone"] = {
        "selected_candidate": best_candidate,
        "mean_validation_auc": float(means[best_candidate]),
        "checkpoint": str(standalone_path)
    }

    metadata = {
        "seed": seed,
        "candidate_run": run_id,
        "candidate_mean_auc": means.tolist(),
        "candidate_std_auc": stds.tolist(),
        "selections": selections
    }

    with open(output_dir / "selection_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Seed {seed}: checkpoint preparation complete.")

print("\nALL ABMIL FUSION CHECKPOINTS PREPARED")
