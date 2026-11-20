from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Raymvp12/focusmil-camelyon16",
    repo_type="dataset",
    allow_patterns=["Camelyon16feat_256_10x/*"],
    local_dir="downloaded_features"
)

print("Feature download complete!")

#Check features

import h5py
import numpy as np
from pathlib import Path

base = Path("downloaded_features/Camelyon16feat_256_10x")

for split in ["train", "val"]:
    print(f"\n===== {split.upper()} =====")

    with h5py.File(base / f"{split}_patch_feat.h5", "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print("HDF5:", name, "shape:", obj.shape,
                      "dtype:", obj.dtype)

        f.visititems(show)

    for name in [
        "patch_corresponding_slide_index",
        "patch_corresponding_slide_label",
        "patch_corresponding_slide_name",
        "patch_label"
    ]:
        arr = np.load(
            base / f"{split}_{name}.npy",
            mmap_mode="r",
            allow_pickle=True
        )

        print(name)
        print("  shape:", arr.shape)
        print("  dtype:", arr.dtype)
        print("  first 5:", arr[:5])
