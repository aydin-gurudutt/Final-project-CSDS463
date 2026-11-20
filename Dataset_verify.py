from pathlib import Path
import pandas as pd

dataset = Path("datasets/camelyon16")

expected = {
    "camelyon16.csv": 397,
    "train.csv": 215,
    "val.csv": 54,
    "test.csv": 128,
}

for filename, count in expected.items():
    df = pd.read_csv(dataset / filename)

    assert len(df) == count, f"{filename}: incorrect count"

    missing = [
        p for p in df["path"]
        if not Path(p).is_file()
    ]
    assert not missing, f"{filename}: {len(missing)} missing paths"

    print(f"{filename}: {count} slides, all paths valid")

feature_files = list((dataset / "features").glob("*.csv"))
print("Feature files:", len(feature_files))
assert len(feature_files) == 397

sample = pd.read_csv(feature_files[0])
print("Sample shape:", sample.shape)
assert sample.shape[1] == 512

train = pd.read_csv(dataset / "train.csv")
val = pd.read_csv(dataset / "val.csv")
test = pd.read_csv(dataset / "test.csv")

assert set(train.path).isdisjoint(val.path)
assert set(train.path).isdisjoint(test.path)
assert set(val.path).isdisjoint(test.path)

print("Dataset verification passed!")