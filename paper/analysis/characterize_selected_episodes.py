#!/usr/bin/env python3
"""Create privacy-safe camera-ready aggregates from retained row-level outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


RAW = "raw_4096"
ERA = "condition_era_90_backfill_4096"
TASKS = ("guo_readmission", "guo_icu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def top_ids(frame: pd.DataFrame, risk_column: str = "risk_calibrated") -> set[int]:
    risk = frame[risk_column].to_numpy(dtype=float)
    row_id = frame["row_id"].to_numpy(dtype=int)
    if not len(frame) or frame["row_id"].duplicated().any():
        raise ValueError("Expected a nonempty cohort with unique episode keys")
    if not np.isfinite(risk).all() or not ((risk >= 0) & (risk <= 1)).all():
        raise ValueError("Invalid probability")
    order = np.lexsort((row_id, -risk))
    k = max(1, int(math.ceil(0.10 * len(frame))))
    return set(row_id[order[:k]].tolist())


def quantiles(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame[column].dropna().to_numpy(dtype=float)
    if not len(values):
        return {
            f"{column}_median": np.nan,
            f"{column}_q1": np.nan,
            f"{column}_q3": np.nan,
        }
    return {
        f"{column}_median": float(np.median(values)),
        f"{column}_q1": float(np.quantile(values, 0.25)),
        f"{column}_q3": float(np.quantile(values, 0.75)),
    }


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = root / "artifacts/ehrshot_state_or_space_final_analysis_5seeds_wide"
    copy_forward = root / "artifacts/state_or_space_copy_forward/inference"
    ensemble = pd.read_csv(analysis / "ensemble_predictions.csv")
    plan = pd.read_csv(copy_forward / "copy_forward_episode_plan/copy_forward_episode_plan.csv")
    predictions = pd.read_csv(copy_forward / "copy_forward_predictions/copy_forward_predictions.csv")
    stress_ensemble = pd.read_csv(
        copy_forward / "copy_forward_ensemble_predictions/copy_forward_ensemble_predictions.csv"
    )
    eligible = pd.read_csv(copy_forward / "copy_forward_eligible_visits/copy_forward_eligible_visits.csv")
    return ensemble, plan, predictions, stress_ensemble, eligible


def prepare_plan(plan: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    plan = plan.copy()
    plan["prediction_time"] = pd.to_datetime(plan["prediction_time"], errors="raise")
    plan["first_diagnosis_time"] = pd.to_datetime(
        plan["first_diagnosis_time"], errors="raise"
    )
    plan["candidate_age_days"] = (
        plan["prediction_time"] - plan["first_diagnosis_time"]
    ).dt.total_seconds() / 86400.0

    baseline = predictions[np.isclose(predictions["requested_copy_fraction"], 0.0)]
    lengths = baseline.groupby(["task", "row_id", "compression_version"])["seq_len"].nunique()
    if not lengths.eq(1).all():
        raise ValueError("Sequence length differs across seeds")
    baseline = (
        baseline.groupby(["task", "row_id", "compression_version"], as_index=False)
        .agg(seq_len=("seq_len", "first"))
    )
    raw_length = baseline[baseline["compression_version"] == RAW][
        ["task", "row_id", "seq_len"]
    ].rename(columns={"seq_len": "raw_sequence_length"})
    return plan.merge(
        raw_length,
        on=["task", "row_id"],
        how="left",
        validate="one_to_one",
    )


def characterize_discordant_sets(
    ensemble: pd.DataFrame, plan: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for task in TASKS:
        task_frame = ensemble[
            (ensemble["task"] == task)
            & ensemble["compression_version"].isin([RAW, ERA])
        ]
        raw = task_frame[task_frame["compression_version"] == RAW]
        era = task_frame[task_frame["compression_version"] == ERA]
        raw_ids = top_ids(raw)
        era_ids = top_ids(era)
        overlap = raw_ids & era_ids
        stability_rows.append(
            {
                "task": task,
                "top_k": len(raw_ids),
                "intersection": len(overlap),
                "ensemble_cross_representation_jaccard": len(overlap)
                / len(raw_ids | era_ids),
                "raw_only": len(raw_ids - era_ids),
                "era_only": len(era_ids - raw_ids),
            }
        )

        labels = raw.set_index("row_id")["y_true"]
        subjects = raw.set_index("row_id")["subject_id"]
        groups = (("raw_only", raw_ids - era_ids), ("era_only", era_ids - raw_ids))
        for group_name, identifiers in groups:
            identifier_list = sorted(identifiers)
            characterized = plan[
                (plan["task"] == task) & plan["row_id"].isin(identifier_list)
            ].copy()
            row: dict[str, object] = {
                "task": task,
                "selection_group": group_name,
                "n_episodes": len(identifier_list),
                "n_patients": int(subjects.loc[identifier_list].nunique()),
                "n_positive": int(labels.loc[identifier_list].sum()),
                "event_rate": float(labels.loc[identifier_list].mean()),
                "n_characterized": int(len(characterized)),
                "characterization_coverage": float(
                    len(characterized) / len(identifier_list)
                ),
                "raw_context_at_4096_fraction": float(
                    np.mean(characterized["raw_sequence_length"] == 4096)
                ),
            }
            for column in (
                "n_existing_mentions",
                "candidate_age_days",
                "raw_sequence_length",
            ):
                row.update(quantiles(characterized, column))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(stability_rows)


def characterize_stress_churn(
    stress_ensemble: pd.DataFrame, eligible: pd.DataFrame
) -> pd.DataFrame:
    eligible = eligible.copy()
    eligible["visit_time"] = pd.to_datetime(eligible["visit_time"], errors="raise")
    eligible["prediction_time"] = pd.to_datetime(
        eligible["prediction_time"], errors="raise"
    )
    affected = eligible.loc[
        eligible["visit_time"] >= eligible["prediction_time"], ["task", "row_id"]
    ].drop_duplicates()
    strict = stress_ensemble.merge(
        affected.assign(_affected=True),
        on=["task", "row_id"],
        how="left",
        validate="many_to_one",
    )
    strict = strict[strict["_affected"].isna()].drop(columns="_affected")

    rows: list[dict[str, object]] = []
    selected = strict[strict["compression_version"].isin([RAW, ERA])]
    for (task, version), frame in selected.groupby(
        ["task", "compression_version"], sort=True
    ):
        baseline = frame[np.isclose(frame["requested_copy_fraction"], 0.0)]
        perturbed = frame[np.isclose(frame["requested_copy_fraction"], 1.0)]
        baseline_ids = top_ids(baseline)
        perturbed_ids = top_ids(perturbed)
        entered = perturbed_ids - baseline_ids
        exited = baseline_ids - perturbed_ids
        labels = baseline.set_index("row_id")["y_true"]
        subjects = baseline.set_index("row_id")["subject_id"]
        changed = entered | exited
        rows.append(
            {
                "task": task,
                "compression_version": version,
                "requested_copy_fraction": 1.0,
                "top_k": len(baseline_ids),
                "n_entered": len(entered),
                "n_entered_positive": int(labels.loc[sorted(entered)].sum())
                if entered
                else 0,
                "n_exited": len(exited),
                "n_exited_positive": int(labels.loc[sorted(exited)].sum())
                if exited
                else 0,
                "n_changed_patients": int(subjects.loc[sorted(changed)].nunique())
                if changed
                else 0,
                "churn_fraction": len(exited) / len(baseline_ids),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensemble, plan, predictions, stress_ensemble, eligible = load_inputs(
        args.package_root
    )
    plan = prepare_plan(plan, predictions)
    discordant, stability = characterize_discordant_sets(ensemble, plan)
    churn = characterize_stress_churn(stress_ensemble, eligible)

    expected = {
        ("guo_readmission", "raw_only"): (34, 10),
        ("guo_readmission", "era_only"): (34, 13),
        ("guo_icu", "raw_only"): (51, 5),
        ("guo_icu", "era_only"): (51, 6),
    }
    observed = {
        (row.task, row.selection_group): (int(row.n_episodes), int(row.n_positive))
        for row in discordant.itertuples()
    }
    if observed != expected:
        raise ValueError(f"Unexpected discordant-set counts: {observed}")

    for frame in (discordant, stability, churn):
        forbidden = {"row_id", "subject_id"}.intersection(frame.columns)
        if forbidden:
            raise ValueError(f"Private identifier columns in aggregate output: {forbidden}")

    discordant.to_csv(
        args.output_dir / "discordant_top_decile_characterization.csv", index=False
    )
    stability.to_csv(
        args.output_dir / "ensemble_cross_representation_stability.csv", index=False
    )
    churn.to_csv(args.output_dir / "full_repetition_top_decile_churn.csv", index=False)


if __name__ == "__main__":
    main()
