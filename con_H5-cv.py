
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE = Path("downloaded_features/Camelyon16feat_256_10x")
DEST = Path("datasets/camelyon16")
FEATURE_DIR = DEST / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

all_slides = []

def convert_collection(collection):
    print(f"\nProcessing {collection} collection...", flush=True)

    indices = np.load(
        SOURCE / f"{collection}_patch_corresponding_slide_index.npy"
    )
    labels = np.load(
        SOURCE / f"{collection}_patch_corresponding_slide_label.npy"
    )
    names = np.load(
        SOURCE / f"{collection}_patch_corresponding_slide_name.npy"
    )

    unique_ids, first_positions = np.unique(
        indices, return_index=True
    )

    records = []

    with h5py.File(SOURCE / f"{collection}_patch_feat.h5", "r") as h5:
        features = h5["dataset_1"]

        assert features.shape[0] == len(indices)
        assert features.shape[1] == 512

        for i, (slide_id, first) in enumerate(
            zip(unique_ids, first_positions), start=1
        ):
            positions = np.flatnonzero(indices == slide_id)

            slide_name = str(names[first])
            slide_label = int(labels[first])

            assert slide_label in (0, 1)
            assert np.all(names[positions] == slide_name)
            assert np.all(labels[positions] == slide_label)

            # Sorted positions allow HDF5 indexing.
            slide_features = features[positions, :]

            output_path = FEATURE_DIR / f"{slide_name}.csv"

            # Original loader uses pandas.read_csv with a header.
            pd.DataFrame(slide_features).to_csv(
                output_path,
                index=False,
                header=True,
                float_format="%.9g"
            )

            records.append({
                "path": output_path.as_posix(),
                "label": slide_label,
                "slide": slide_name
            })

            if i % 20 == 0 or i == len(unique_ids):
                print(
                    f"{collection}: {i}/{len(unique_ids)} slides",
                    flush=True
                )

    return pd.DataFrame(records)


# Original training collection
training_collection = convert_collection("train")

# Downloaded "val" collection is reserved for testing
testing_collection = convert_collection("val")

assert not (
    set(training_collection["slide"])
    & set(testing_collection["slide"])
)

# Create a reproducible validation split from training slides
train_df, val_df = train_test_split(
    training_collection,
    test_size=0.20,
    random_state=42,
    stratify=training_collection["label"]
)

# All slides and labels
master = pd.concat(
    [training_collection, testing_collection],
    ignore_index=True
)

assert master["slide"].is_unique

master[["path", "label"]].to_csv(
    DEST / "camelyon16.csv", index=False
)

train_df[["path"]].to_csv(
    DEST / "train.csv", index=False
)

val_df[["path"]].to_csv(
    DEST / "val.csv", index=False
)

testing_collection[["path", "label"]].to_csv(
    DEST / "test.csv", index=False
)

print("\n===== FINAL DATASET =====")

for title, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", testing_collection)
]:
    print(
        title,
        "slides:", len(df),
        "class counts:", df["label"].value_counts().to_dict()
    )

print("\nConversion completed successfully.")
print("Files saved to:", DEST)
