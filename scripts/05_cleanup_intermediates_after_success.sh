#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_DELETE_INTERMEDIATE_CACHES:-}" != "YES" ]]; then
  echo "Disabled by default. Inspect cache paths and set CONFIRM_DELETE_INTERMEDIATE_CACHES=YES."
  exit 1
fi

# Run only after representation_invariants.csv contains no failed checks
# and all dataset files are uploaded to storage.

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("ehrshot_state_or_space_sequence_datasets/representation_invariants.csv")
if not path.exists():
    raise SystemExit(f"Missing {path}")
df = pd.read_csv(path)
if "passed" not in df.columns:
    raise SystemExit("representation_invariants.csv has no passed column")
passed = df["passed"].astype(str).str.lower().isin(["true", "1"])
failed = df[~passed]
if len(failed):
    raise SystemExit(f"Cannot clean: {len(failed)} invariant checks failed")
print(f"All {len(df)} invariant checks passed")
PY

find ehrshot_state_or_space_sequence_datasets \
  -type d -name '_long_parts' -prune -exec rm -rf {} +

rm -rf ehrshot_state_or_space_cache

echo "Intermediate long parts and local cache removed."
