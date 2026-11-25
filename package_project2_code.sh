#!/usr/bin/env bash
# Run from the repo on Pioneer after synchronizing with origin/main.
set -euo pipefail
test -f README.md && test -f standalone_comparison.py && test -f verify_published_results.py
OUT="${1:-Project2_Source_Code.zip}"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing archive: $OUT" >&2
  exit 1
fi
python3 - "$OUT" <<'PY'
import sys, subprocess, zipfile
from pathlib import Path
destination = Path(sys.argv[1])
files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
allow_ext = {".py", ".slurm", ".sbatch", ".sh", ".csv", ".md", ".txt", ".yml", ".yaml"}
kept = [Path(x) for x in files if x and Path(x).suffix.lower() in allow_ext and Path(x).is_file()]
if not kept:
    raise SystemExit("No tracked source files found")
with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in kept:
        archive.write(path, arcname=str(path))
print(f"Created {destination} with {len(kept)} tracked files")
PY
