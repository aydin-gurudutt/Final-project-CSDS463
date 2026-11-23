import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import train_repro_clam as clam
import train_repro_dsmil as dsmil
from train_fusion_dsmil import build_model

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

OUTPUT = Path("heldout_evaluation")
RESULTS = OUTPUT / "test_auc_results.csv"


def make_args(arch):
    return Namespace(
        model=arch,
        dataset="camelyon16",
        num_classes=1,
        feats_size=512,
        q_n=128,
        non_linearity=1.0,
        dropout_patch=0,
        dropout_node=0,
        average=True,
        init_type="orthogonal",
        gpu_index=[0],
    )


def main():
    if RESULTS.exists():
        raise FileExistsError(
            f"Test results already exist: {RESULTS}"
        )

    torch.manual_seed(12345)
    np.random.seed(12345)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    assert device.type == "cuda", "GPU required"

    root = Path("datasets/camelyon16")
    test_df = pd.read_csv(root / "test.csv")
    train_df = pd.read_csv(root / "train.csv")
    val_df = pd.read_csv(root / "val.csv")

    assert len(test_df) == 128
    assert {"path", "label"}.issubset(test_df.columns)

    names = lambda df: set(
        Path(str(p)).name for p in df["path"]
    )

    assert names(train_df).isdisjoint(names(test_df))
    assert names(val_df).isdisjoint(names(test_df))
    assert len(names(test_df)) == 128

    labels = test_df["label"].astype(int).to_numpy()
    assert set(labels.tolist()) == {0, 1}

    jobs = []

    for arch, seeds in RUNS.items():
        penalty = 0.5 if arch == "clam" else 5.0

        for seed, directory in seeds.items():
            for lam in (0.0, penalty):
                checkpoint = (
                    Path("weights") / arch / directory
                    / "fusion_results"
                    / f"lambda_{lam}_seed_{seed}"
                    / "best_validation.pth"
                )

                history = checkpoint.with_name("history.json")

                assert checkpoint.is_file(), checkpoint
                assert history.is_file(), history

                with history.open() as f:
                    metadata = json.load(f)

                assert int(metadata["seed"]) == seed

                jobs.append((arch, seed, lam, checkpoint))

    assert len(jobs) == 12

    print("TEST EVALUATION PREFLIGHT PASSED")
    print("Test samples:", len(test_df))
    print("Positive labels:", int(labels.sum()))
    print("Negative labels:", int((labels == 0).sum()))
    print("Checkpoints:", len(jobs), flush=True)

    OUTPUT.mkdir(exist_ok=True)
    results = []

    for arch, seed, lam, checkpoint in jobs:
        args = make_args(arch)
        baseline = clam if arch == "clam" else dsmil

        if arch == "clam":
            model = clam.get_model(args)
        else:
            model = build_model(args)

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

        predictions = []

        with torch.inference_mode():
            for _, row in test_df.iterrows():
                label, features = baseline.get_bag_feats(
                    row, args, random_state=1
                )

                assert int(np.asarray(label).item()) == int(
                    row["label"]
                )

                features = np.asarray(
                    features, dtype=np.float32
                )

                assert features.ndim == 2
                assert features.shape[1] == 512
                assert np.isfinite(features).all()

                x = torch.from_numpy(features).to(device)

                if arch == "clam":
                    bag_prediction, _ = model(x)
                    score = torch.sigmoid(
                        bag_prediction
                    ).reshape(-1)[0].item()

                else:
                    ins_prediction, bag_prediction, _, _ = model(x)
                    max_prediction, _ = torch.max(
                        ins_prediction, dim=0
                    )

                    # Preserve the original DSMIL scoring rule.
                    activated = dsmil.activation(
                        args, bag_prediction, max_prediction
                    )
                    score = float(
                        np.asarray(activated).reshape(-1)[0]
                    )

                assert np.isfinite(score)
                assert 0 <= score <= 1
                predictions.append(score)

        auc = float(roc_auc_score(labels, predictions))
        assert np.isfinite(auc)

        row = {
            "architecture": arch,
            "seed": seed,
            "lambda": lam,
            "test_auc": auc,
            "test_samples": len(labels),
            "checkpoint": str(checkpoint),
        }
        results.append(row)

        print(
            f"{arch.upper()} seed={seed} lambda={lam}: "
            f"TEST AUC={auc:.6f}",
            flush=True,
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the complete table only after all 12 evaluations pass.
    with RESULTS.open("x", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("SAVED:", RESULTS)
    print("ALL HELD-OUT EVALUATIONS COMPLETE")


if __name__ == "__main__":
    main()
