"""Recreate published tables and figures from public aggregate data."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "analysis"))
from generate_decision_gap_figure import render as render_decision
from generate_evaluation_profile_figure import render as render_profile
from tae_submission_analysis import plot_calibration


def verify_manifest(root: Path) -> int:
    manifest = root / "RELEASE_MANIFEST.csv"
    if not manifest.exists():
        raise ValueError("Missing release manifest")
    count = 0
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = (root / row["relative_path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Missing or unsafe manifest entry")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual.lower() != row["sha256"].lower():
                raise ValueError(f"Checksum mismatch: {row['relative_path']}")
            count += 1
    return count


def verify_aggregates(directory: Path) -> dict:
    for path in directory.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if columns & {"row_id", "subject_id", "patient_id", "candidate_code", "prediction_time"}:
            raise ValueError(f"Restricted columns in {path.name}")
    primary = pd.read_csv(directory / "multiplicity_sensitivity.csv")
    primary = primary[primary.comparison == "Full4096"]
    if len(primary) != 10 or not ((primary.ci_low <= 0) & (primary.ci_high >= 0)).all():
        raise ValueError("Unexpected primary interval coverage")
    sets = pd.read_csv(directory / "ensemble_top10_cross_representation.csv")
    if not np.allclose(sets.jaccard, sets.intersection / (2 * sets.top_k - sets.intersection)):
        raise ValueError("Selected-set arithmetic does not reconcile")
    if not (sets.raw_only == sets.top_k - sets.intersection).all():
        raise ValueError("Replacement counts do not reconcile")
    metrics = pd.read_csv(directory / "recomputed_ensemble_metrics.csv")
    for metric in ("auroc", "auprc", "brier", "top_10pct_precision"):
        if not metrics[metric].between(0, 1).all():
            raise ValueError("Invalid published metric: " + metric)
    if not np.isfinite(metrics.logloss).all() or (metrics.logloss < 0).any():
        raise ValueError("Invalid published log loss")
    matched = pd.read_csv(directory / "matched_size_stability.csv")
    if len(matched) != 12 or not matched.mean_jaccard.between(0, 1).all():
        raise ValueError("Incomplete matched-size stability profile")
    probability = pd.read_csv(directory / "copy_forward_strict_pre_prediction_probability.csv")
    counts = probability.groupby("task").n_episodes.unique()
    if any(list(counts[task]) != [n] for task, n in
           [("guo_readmission", 1286), ("guo_icu", 1028)]):
        raise ValueError("Wrong strict stress cohort")
    return {"primary_comparisons": 10, "all_primary_intervals_include_zero": True,
            "matched_size_summaries": 12, "strict_readmission_episodes": 1286,
            "strict_icu_episodes": 1028,
            "scope": "Aggregate checks and figure reproduction; no retraining or row-level bootstrap."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "reproduced")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    inputs = ROOT / "paper" / "aggregate_outputs"
    if output == ROOT or output == inputs.resolve() or inputs.resolve().is_relative_to(output):
        raise ValueError("Output must not overwrite source inputs")
    verify_manifest(ROOT)
    verify_aggregates(inputs)
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    for path in inputs.glob("*.csv"):
        shutil.copy2(path, tables / path.name)
    render_decision(output / "figure_decision_gap.png", inputs)
    render_profile(inputs, output / "figure_evaluation_profile.png")
    bins = pd.read_csv(inputs / "calibration_bins.csv")
    plot_calibration(bins, output)
    print(f"Generated three figures and {len(list(inputs.glob('*.csv')))} tables in {output}.")


if __name__ == "__main__":
    main()
