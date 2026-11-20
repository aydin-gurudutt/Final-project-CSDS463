
import numpy as np
from pathlib import Path

base = Path("downloaded_features/Camelyon16feat_256_10x")

all_splits = {}

for split in ["train", "val"]:
    print(f"\n===== {split.upper()} =====")

    indices = np.load(
        base / f"{split}_patch_corresponding_slide_index.npy"
    )

    labels = np.load(
        base / f"{split}_patch_corresponding_slide_label.npy"
    )

    names = np.load(
        base / f"{split}_patch_corresponding_slide_name.npy"
    )

    unique_indices, first_positions, counts = np.unique(
        indices,
        return_index=True,
        return_counts=True
    )

    slide_names = names[first_positions]
    slide_labels = labels[first_positions]

    # Verify slide membership and labels are consistent
    for idx, name, label in zip(
        unique_indices, slide_names, slide_labels
    ):
        mask = indices == idx
        assert np.all(names[mask] == name)
        assert np.all(labels[mask] == label)

    all_splits[split] = set(slide_names.tolist())

    print("Total slides:", len(unique_indices))
    print("Normal slides:", np.sum(slide_labels == 0))
    print("Tumor slides:", np.sum(slide_labels == 1))
    print("Smallest bag:", counts.min(), "patches")
    print("Largest bag:", counts.max(), "patches")
    print("Average bag:", round(counts.mean(), 1))

    print("\nFirst 10 slides:")
    for name, label, count in zip(
        slide_names[:10],
        slide_labels[:10],
        counts[:10]
    ):
        print(name, "| label:", label, "| patches:", count)

overlap = all_splits["train"] & all_splits["val"]

print("\n===== SPLIT CHECK =====")
print("Slides appearing in both splits:", len(overlap))
print("Overlapping slide names:", sorted(overlap)[:10])
