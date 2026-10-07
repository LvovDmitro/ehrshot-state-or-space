"""Privacy-safe, descriptive camera-ready analyses of retained predictions."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from characterize_selected_episodes import ERA, RAW, TASKS, load_inputs, prepare_plan, quantiles, top_ids


def matched_size_stability(wide: pd.DataFrame) -> pd.DataFrame:
    # Pair counts are descriptive: comparisons reuse fitted models and are dependent.
    rows = []
    for task in TASKS:
        part = wide[(wide.task == task) & wide.compression_version.isin([RAW, ERA])]
        if part.duplicated(["compression_version", "seed", "row_id"]).any():
            raise ValueError("Duplicate prediction key")
        identity_columns = [name for name in ("subject_id", "y_true", "prediction_time", "split")
                            if name in part]
        if (part.groupby("row_id")[identity_columns].nunique() > 1).any().any():
            raise ValueError("Prediction identities differ across runs")
        matrices = {
            version: part[part.compression_version == version].pivot(
                index="row_id", columns="seed", values="risk_calibrated"
            ).sort_index() for version in (RAW, ERA)
        }
        raw, era = matrices[RAW], matrices[ERA]
        if not raw.index.equals(era.index) or not raw.columns.equals(era.columns):
            raise ValueError("Run cohorts or seeds differ")
        if raw.isna().any().any() or era.isna().any().any() or set(raw.columns) != set(range(42, 47)):
            raise ValueError("Expected five complete seeds")
        for matrix in matrices.values():
            values = matrix.to_numpy()
            if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError("Invalid probability")
        for size in (1, 2):
            subsets = list(itertools.combinations(raw.columns.tolist(), size))
            sets = {}
            for version, matrix in matrices.items():
                for subset in subsets:
                    frame = pd.DataFrame({"row_id": matrix.index,
                        "risk_calibrated": matrix.loc[:, list(subset)].mean(axis=1).to_numpy()})
                    sets[(version, subset)] = top_ids(frame)
            comparisons = {
                "raw_within_disjoint_seeds": [(sets[(RAW, a)], sets[(RAW, b)])
                    for a, b in itertools.combinations(subsets, 2) if not set(a) & set(b)],
                "era_within_disjoint_seeds": [(sets[(ERA, a)], sets[(ERA, b)])
                    for a, b in itertools.combinations(subsets, 2) if not set(a) & set(b)],
                "cross_representation_matched_seeds": [(sets[(RAW, a)], sets[(ERA, a)])
                    for a in subsets],
            }
            for comparison, pairs in comparisons.items():
                overlaps = [len(a & b) / len(a | b) for a, b in pairs]
                rows.append({"task": task, "ensemble_size": size, "comparison": comparison,
                    "n_pairs": len(pairs), "mean_jaccard": np.mean(overlaps),
                    "min_jaccard": min(overlaps), "max_jaccard": max(overlaps),
                    "n_episodes": len(raw), "top_k": math.ceil(0.10 * len(raw))})
    return pd.DataFrame(rows)


def characterize_shift_tails(stress: pd.DataFrame, eligible: pd.DataFrame,
                             plan: pd.DataFrame) -> pd.DataFrame:
    affected = eligible[pd.to_datetime(eligible.visit_time) >= pd.to_datetime(
        eligible.prediction_time)][["task", "row_id"]].drop_duplicates()
    strict = stress.merge(affected.assign(_affected=True), on=["task", "row_id"],
                          how="left", validate="many_to_one")
    strict = strict[strict._affected.isna()]
    rows = []
    for task in TASKS:
        for version in (RAW, ERA):
            part = strict[(strict.task == task) & (strict.compression_version == version)]
            baseline = part[np.isclose(part.requested_copy_fraction, 0.0)]
            full = part[np.isclose(part.requested_copy_fraction, 1.0)]
            if set(full.row_id) != set(baseline.row_id):
                raise ValueError("Stress cohorts differ")
            paired = full.merge(baseline[["row_id", "subject_id", "y_true", "risk_calibrated"]], on="row_id",
                                suffixes=("", "_baseline"), validate="one_to_one")
            if len(paired) != len(baseline):
                raise ValueError("Stress cohorts differ")
            if not (paired.subject_id == paired.subject_id_baseline).all() or not (
                    paired.y_true == paired.y_true_baseline).all():
                raise ValueError("Stress identities differ")
            if not np.isfinite(paired[["risk_calibrated", "risk_calibrated_baseline"]]).all().all():
                raise ValueError("Invalid probability")
            paired["abs_shift"] = abs(paired.risk_calibrated - paired.risk_calibrated_baseline)
            paired = paired.sort_values(["abs_shift", "row_id"], ascending=[False, True])
            k = math.ceil(0.05 * len(paired))
            for group, frame in (("largest_5pct_shifts", paired.iloc[:k]),
                                 ("remaining_95pct", paired.iloc[k:])):
                joined = frame.merge(plan, on=["task", "row_id"], how="left",
                                     suffixes=("", "_plan"), validate="one_to_one")
                if joined.candidate_age_days.isna().any():
                    raise ValueError("Stress episodes missing candidate profile")
                row = {"task": task, "compression_version": version, "shift_group": group,
                    "n_episodes": len(frame), "n_positive": int(frame.y_true.sum()),
                    "n_patients": int(frame.subject_id.nunique()),
                    "mean_abs_shift": frame.abs_shift.mean(), "max_abs_shift": frame.abs_shift.max(),
                    "raw_context_at_4096_fraction": float((joined.raw_sequence_length == 4096).mean())}
                for column in ("n_existing_mentions", "candidate_age_days", "n_eligible_visits"):
                    row.update(quantiles(joined, column))
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, plan, predictions, stress, eligible = load_inputs(args.package_root)
    plan = prepare_plan(plan, predictions)
    wide = pd.read_csv(args.package_root / (
        "artifacts/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/"
        "sequence_multiseed_heldout_predictions_wide.csv"))
    outputs = {"matched_size_stability.csv": matched_size_stability(wide),
               "stress_shift_tail_characterization.csv": characterize_shift_tails(stress, eligible, plan)}
    for filename, frame in outputs.items():
        if {"row_id", "subject_id", "candidate_code"}.intersection(frame.columns):
            raise ValueError("Restricted columns in aggregate output")
        frame.to_csv(args.output_dir / filename, index=False)
    print("Created two aggregate tables; no identifiers or episode predictions exported.")


if __name__ == "__main__":
    main()
