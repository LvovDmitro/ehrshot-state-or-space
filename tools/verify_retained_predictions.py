"""Exercise current analysis contracts on authorized retained outputs, without training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_pipeline_contracts import extract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
namespace = extract("03_analyze_state_or_space.py",
                    {"validate_wide_predictions", "make_ensemble", "sigmoid_np", "expected_task_versions"},
                    {"WIDE_REQUIRED_COLUMNS", "REQUIRED_SEEDS"})
config = json.loads((ROOT / "configs/state_or_space_analysis_5seeds.json").read_text())
wide = pd.read_csv(args.package_root / (
    "artifacts/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/"
    "sequence_multiseed_heldout_predictions_wide.csv"))
validated = namespace["validate_wide_predictions"](wide, config)
ensemble = namespace["make_ensemble"](validated)
stored = pd.read_csv(args.package_root /
    "artifacts/ehrshot_state_or_space_final_analysis_5seeds_wide/ensemble_predictions.csv")
keys = ["task", "compression_version", "row_id"]
paired = ensemble.merge(stored[keys + ["risk_calibrated"]], on=keys, how="outer",
                        suffixes=("", "_stored"), indicator=True, validate="one_to_one")
if not paired["_merge"].eq("both").all():
    raise ValueError("Ensemble identity mismatch")
difference = float(np.max(abs(paired.risk_calibrated - paired.risk_calibrated_stored)))
if difference > 1e-12:
    raise ValueError("Current contracts changed fitted ensemble probabilities")
summary = {"validated_prediction_rows": len(validated), "ensemble_rows": len(ensemble),
           "exact_seeds": [42, 43, 44, 45, 46], "maximum_probability_difference": difference,
           "training_performed": False, "protected_rows_exported": False}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
