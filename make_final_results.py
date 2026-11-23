import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

out = Path("final_results")
out.mkdir(exist_ok=True)

test = pd.read_csv("heldout_evaluation/test_auc_results.csv")
assert len(test) == 12
assert not test.duplicated(
    ["architecture", "seed", "lambda"]
).any()

rows = []

for architecture, seeds in RUNS.items():
    stable_lambda = 0.5 if architecture == "clam" else 5.0

    for seed, run in seeds.items():
        for lam, method in [
            (0.0, "Baseline"),
            (stable_lambda, "Stability-aware"),
        ]:
            directory = (
                Path("weights") / architecture / run
                / "fusion_results"
                / f"lambda_{lam}_seed_{seed}"
            )

            with (directory / "history.json").open() as f:
                history = json.load(f)

            matching = test[
                (test["architecture"] == architecture)
                & (test["seed"] == seed)
                & (np.isclose(test["lambda"], lam))
            ]

            assert len(matching) == 1
            assert (directory / "best_validation.pth").is_file()

            rows.append({
                "architecture": architecture.upper(),
                "seed": seed,
                "method": method,
                "lambda": lam,
                "validation_auc": float(
                    history["best_validation_auc"]
                ),
                "test_auc": float(
                    matching.iloc[0]["test_auc"]
                ),
            })

df = pd.DataFrame(rows)
assert len(df) == 12

df.to_csv(out / "all_results.csv", index=False)

summary = (
    df.groupby(["architecture", "method"], sort=True)
    .agg(
        validation_mean=("validation_auc", "mean"),
        validation_sd=("validation_auc", "std"),
        test_mean=("test_auc", "mean"),
        test_sd=("test_auc", "std"),
    )
    .reset_index()
)

summary.to_csv(out / "summary.csv", index=False)

paired = df.pivot(
    index=["architecture", "seed"],
    columns="method",
    values=["validation_auc", "test_auc"],
)

for metric in ["validation_auc", "test_auc"]:
    paired[(metric, "difference")] = (
        paired[(metric, "Stability-aware")]
        - paired[(metric, "Baseline")]
    )

paired.to_csv(out / "paired_differences.csv")

# Figure 1: Mean performance across seeds.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, metric, ylabel in [
    (axes[0], "validation", "Validation ROC AUC"),
    (axes[1], "test", "Held-out test ROC AUC"),
]:
    architectures = ["CLAM", "DSMIL"]
    positions = np.arange(len(architectures))
    width = 0.34

    for j, method in enumerate(
        ["Baseline", "Stability-aware"]
    ):
        subset = (
            summary[summary["method"] == method]
            .set_index("architecture")
            .loc[architectures]
        )

        ax.bar(
            positions + (j - 0.5) * width,
            subset[f"{metric}_mean"],
            width,
            yerr=subset[f"{metric}_sd"],
            capsize=5,
            label=method,
        )

    ax.set_xticks(positions, architectures)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.capitalize()} performance (n=3)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(out / "mean_auc_comparison.png", dpi=200)
plt.close(fig)

# Figure 2: Each seed's paired held-out test difference.
fig, ax = plt.subplots(figsize=(8, 4.5))

for architecture, marker in [
    ("CLAM", "o"),
    ("DSMIL", "s"),
]:
    subset = paired.loc[architecture]
    differences = subset[("test_auc", "difference")]

    ax.plot(
        differences.index,
        differences.values,
        marker=marker,
        label=architecture,
    )

ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Random seed")
ax.set_ylabel("Stability-aware − baseline test ROC AUC")
ax.set_title("Paired held-out test differences")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(out / "paired_test_differences.png", dpi=200)
plt.close(fig)

print("===== FINAL SUMMARY =====")
print(summary.to_string(index=False))

print("\n===== GENERATED FILES =====")
for path in sorted(out.iterdir()):
    print(path)

print("\nFINAL RESULTS AND FIGURES COMPLETE")
