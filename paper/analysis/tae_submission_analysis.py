#!/usr/bin/env python3
from __future__ import annotations

"""Independent audit and paper-facing analyses for the TAE 2026 submission."""

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


TASK_LABELS = {
    "guo_readmission": "30-day readmission",
    "guo_icu": "ICU transfer",
}

METRICS = ["auroc", "auprc", "brier", "logloss", "top_10pct_precision"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AP",
    "brier": "Brier score",
    "logloss": "Log loss",
    "top_10pct_precision": "Top-10% precision",
}

PRIMARY_VERSIONS = ["raw_4096", "condition_era_90_backfill_4096"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--clearml-audit", type=Path)
    parser.add_argument("--calibration-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package_hashes(package_root: Path) -> pd.DataFrame:
    rows = []
    manifest = package_root / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = package_root / relative
        if not path.is_file():
            rows.append(
                {"relative_path": relative, "status": "missing", "expected": expected, "actual": ""}
            )
            continue
        actual = sha256_file(path)
        rows.append(
            {
                "relative_path": relative,
                "status": "ok" if actual.lower() == expected.lower() else "mismatch",
                "expected": expected.lower(),
                "actual": actual.lower(),
            }
        )
    return pd.DataFrame(rows)


def metric_bundle(y: np.ndarray, p: np.ndarray, tie: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    tie = np.asarray(tie)
    order = np.lexsort((tie, -p))
    k = max(1, int(math.ceil(0.10 * len(y))))
    top = order[:k]
    n_positive = int(y.sum())
    precision = float(y[top].mean())
    base_rate = float(y.mean())
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "top_10pct_precision": precision,
        "top_10pct_lift": float(precision / base_rate),
        "top_10pct_event_capture": float(y[top].sum() / n_positive),
        "n": int(len(y)),
        "n_positive": n_positive,
        "event_rate": base_rate,
    }


def load_prediction_files(package_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(
        package_root
        / "artifacts/ehrshot_state_or_space_final_sequence_results/combined_5seeds_wide/"
        "sequence_multiseed_heldout_predictions_wide.csv"
    )
    analysis = package_root / "artifacts/ehrshot_state_or_space_final_analysis_5seeds_wide"
    ensemble = pd.read_csv(analysis / "ensemble_predictions.csv")
    metrics = pd.read_csv(analysis / "ensemble_metrics.csv")
    return wide, ensemble, metrics


def audit_predictions(
    wide: pd.DataFrame,
    ensemble: pd.DataFrame,
    stored_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    wide_key = ["task", "compression_version", "seed", "row_id"]
    duplicate_wide = int(wide.duplicated(wide_key).sum())
    identity_variation = (
        wide.groupby(["task", "row_id"], dropna=False)[
            ["subject_id", "prediction_time", "y_true"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )

    ensemble_keys = [
        "task",
        "model",
        "model_family",
        "representation",
        "compression_version",
        "max_len",
        "era_gap",
        "numeric_on",
        "split",
        "row_id",
        "subject_id",
        "prediction_time",
        "y_true",
    ]
    recomputed_ensemble = (
        wide.groupby(ensemble_keys, dropna=False, as_index=False)
        .agg(
            risk_calibrated=("risk_calibrated", "mean"),
            risk_calibrated_std=("risk_calibrated", "std"),
            risk_raw=("risk_raw", "mean"),
            logit_mean=("logit", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["task", "compression_version", "row_id"])
        .reset_index(drop=True)
    )
    stored_ensemble = ensemble[ensemble_keys + [
        "risk_calibrated", "risk_calibrated_std", "risk_raw", "logit_mean", "n_seeds"
    ]].sort_values(["task", "compression_version", "row_id"]).reset_index(drop=True)
    ensemble_merge = recomputed_ensemble.merge(
        stored_ensemble,
        on=ensemble_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_recomputed", "_stored"),
        indicator=True,
    )
    ensemble_diffs = {}
    for column in ["risk_calibrated", "risk_calibrated_std", "risk_raw", "logit_mean"]:
        ensemble_diffs[column] = float(
            np.nanmax(
                np.abs(
                    ensemble_merge[f"{column}_recomputed"].to_numpy(dtype=float)
                    - ensemble_merge[f"{column}_stored"].to_numpy(dtype=float)
                )
            )
        )

    metric_rows = []
    for (task, version), part in ensemble.groupby(["task", "compression_version"], sort=True):
        row = {"task": task, "compression_version": version}
        row.update(
            metric_bundle(
                part["y_true"].to_numpy(),
                part["risk_calibrated"].to_numpy(),
                part["row_id"].to_numpy(),
            )
        )
        row["n_patients"] = int(part["subject_id"].nunique())
        metric_rows.append(row)
    recomputed_metrics = pd.DataFrame(metric_rows)

    metric_columns = METRICS + [
        "top_10pct_lift",
        "top_10pct_event_capture",
        "n",
        "n_positive",
        "event_rate",
        "n_patients",
    ]
    metric_check = recomputed_metrics.merge(
        stored_metrics[["task", "compression_version"] + metric_columns],
        on=["task", "compression_version"],
        validate="one_to_one",
        suffixes=("_recomputed", "_stored"),
    )
    for column in metric_columns:
        metric_check[f"{column}_abs_diff"] = np.abs(
            metric_check[f"{column}_recomputed"] - metric_check[f"{column}_stored"]
        )

    sigmoid_difference = float(
        np.max(np.abs(expit(wide["logit"].to_numpy()) - wide["risk_raw"].to_numpy()))
    )
    summary = {
        "wide_rows": int(len(wide)),
        "ensemble_rows": int(len(ensemble)),
        "duplicate_wide_keys": duplicate_wide,
        "identity_inconsistency_task_row": int(identity_variation.sum()),
        "wide_splits": sorted(wide["split"].astype(str).unique().tolist()),
        "wide_task_version_pairs": int(wide[["task", "compression_version"]].drop_duplicates().shape[0]),
        "seeds": sorted(wide["seed"].astype(int).unique().tolist()),
        "seed_count_per_task_version": sorted(
            wide.groupby(["task", "compression_version"])["seed"].nunique().unique().tolist()
        ),
        "sigmoid_logit_max_abs_diff": sigmoid_difference,
        "ensemble_merge_counts": ensemble_merge["_merge"].value_counts().to_dict(),
        "ensemble_max_abs_diffs": ensemble_diffs,
        "metric_max_abs_diff": float(
            metric_check[[f"{column}_abs_diff" for column in metric_columns]].to_numpy().max()
        ),
    }
    return recomputed_metrics, metric_check, summary


def calibration_fit(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    design = np.column_stack([np.ones(len(x)), x])
    sample_weight = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        mu = expit(np.clip(design @ beta, -30, 30))
        variance = np.clip(mu * (1 - mu), 1e-9, None) * sample_weight
        information = design.T @ (variance[:, None] * design)
        score = design.T @ (sample_weight * (y - mu))
        step = np.linalg.pinv(information) @ score
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def calibration_bins(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    rows = []
    ece = 0.0
    for bin_index, indices in enumerate(np.array_split(order, n_bins), start=1):
        if len(indices) == 0:
            continue
        mean_predicted = float(p[indices].mean())
        observed_rate = float(y[indices].mean())
        weight = len(indices) / len(y)
        ece += weight * abs(observed_rate - mean_predicted)
        rows.append(
            {
                "bin": bin_index,
                "n": int(len(indices)),
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "min_predicted": float(p[indices].min()),
                "max_predicted": float(p[indices].max()),
            }
        )
    return pd.DataFrame(rows), float(ece)


def cluster_bootstrap_calibration(
    frame: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame["risk_calibrated"].to_numpy(dtype=float)
    subjects = frame["subject_id"].to_numpy()
    unique_subjects = pd.unique(subjects)
    groups = [np.flatnonzero(subjects == subject) for subject in unique_subjects]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled_groups])
        y_boot = y[indices]
        if np.unique(y_boot).size < 2:
            continue
        p_boot = p[indices]
        intercept, slope = calibration_fit(y_boot, p_boot)
        _, ece = calibration_bins(y_boot, p_boot)
        rows.append(
            {
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_10_equal_frequency": ece,
                "mean_prediction_minus_event_rate": float(p_boot.mean() - y_boot.mean()),
            }
        )
    boot = pd.DataFrame(rows)
    result: dict[str, Any] = {"n_bootstrap_valid": int(len(boot))}
    for column in boot.columns:
        result[f"{column}_ci_low"] = float(boot[column].quantile(0.025))
        result[f"{column}_ci_high"] = float(boot[column].quantile(0.975))
    return result


def analyze_calibration(
    ensemble: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bin_frames = []
    primary_counter = 0
    for (task, version), part in ensemble.groupby(["task", "compression_version"], sort=True):
        y = part["y_true"].to_numpy(dtype=int)
        p = part["risk_calibrated"].to_numpy(dtype=float)
        intercept, slope = calibration_fit(y, p)
        bins, ece = calibration_bins(y, p)
        bins.insert(0, "compression_version", version)
        bins.insert(0, "task", task)
        bin_frames.append(bins)
        row: dict[str, Any] = {
            "task": task,
            "compression_version": version,
            "n": int(len(part)),
            "n_patients": int(part["subject_id"].nunique()),
            "n_positive": int(y.sum()),
            "event_rate": float(y.mean()),
            "mean_prediction": float(p.mean()),
            "mean_prediction_minus_event_rate": float(p.mean() - y.mean()),
            "ece_10_equal_frequency": ece,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        }
        if version in PRIMARY_VERSIONS:
            primary_counter += 1
            row.update(
                cluster_bootstrap_calibration(
                    part,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed + primary_counter,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(bin_frames, ignore_index=True)


def top_fraction_set(frame: pd.DataFrame, risk_column: str, fraction: float = 0.10) -> set[int]:
    risk = frame[risk_column].to_numpy(dtype=float)
    row_id = frame["row_id"].to_numpy(dtype=int)
    order = np.lexsort((row_id, -risk))
    k = max(1, int(math.ceil(fraction * len(frame))))
    return set(row_id[order[:k]].tolist())


def analyze_seed_top10(wide: pd.DataFrame, ensemble: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    for (task, version), part in wide.groupby(["task", "compression_version"], sort=True):
        seed_sets = {
            int(seed): top_fraction_set(seed_part, "risk_calibrated")
            for seed, seed_part in part.groupby("seed", sort=True)
        }
        for seed_a, seed_b in itertools.combinations(sorted(seed_sets), 2):
            set_a = seed_sets[seed_a]
            set_b = seed_sets[seed_b]
            intersection = len(set_a & set_b)
            pair_rows.append(
                {
                    "task": task,
                    "compression_version": version,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "top_k": len(set_a),
                    "intersection": intersection,
                    "retention": intersection / len(set_a),
                    "jaccard": intersection / len(set_a | set_b),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    summary = (
        pairs.groupby(["task", "compression_version"], as_index=False)
        .agg(
            n_seed_pairs=("jaccard", "size"),
            mean_jaccard=("jaccard", "mean"),
            min_jaccard=("jaccard", "min"),
            max_jaccard=("jaccard", "max"),
            mean_retention=("retention", "mean"),
            min_retention=("retention", "min"),
        )
    )

    cross_rows = []
    for task, part in ensemble.groupby("task", sort=True):
        raw = part[part["compression_version"] == "raw_4096"]
        compressed = part[part["compression_version"] == "condition_era_90_backfill_4096"]
        raw_set = top_fraction_set(raw, "risk_calibrated")
        compressed_set = top_fraction_set(compressed, "risk_calibrated")
        intersection = len(raw_set & compressed_set)
        cross_rows.append(
            {
                "task": task,
                "top_k": len(raw_set),
                "intersection": intersection,
                "retention": intersection / len(raw_set),
                "jaccard": intersection / len(raw_set | compressed_set),
                "raw_only": len(raw_set - compressed_set),
                "compressed_only": len(compressed_set - raw_set),
            }
        )
    return pairs, summary, pd.DataFrame(cross_rows)


def adjust_pvalues(pvalues: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(pvalues), dtype=float)
    m = len(p)
    order = np.argsort(p)
    ordered = p[order]

    holm_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    holm_ordered = np.clip(holm_ordered, 0, 1)
    holm = np.empty(m, dtype=float)
    holm[order] = holm_ordered

    ranks = np.arange(1, m + 1)
    bh_ordered = ordered * m / ranks
    bh_ordered = np.minimum.accumulate(bh_ordered[::-1])[::-1]
    bh_ordered = np.clip(bh_ordered, 0, 1)
    bh = np.empty(m, dtype=float)
    bh[order] = bh_ordered
    return holm, bh


def multiplicity_table(episode_bootstrap: pd.DataFrame) -> pd.DataFrame:
    table = episode_bootstrap.copy()
    n = table["n_bootstrap_valid"].to_numpy(dtype=int)
    smaller_tail = np.minimum(
        table["fraction_bootstrap_model_a_better"].to_numpy(dtype=float),
        table["fraction_bootstrap_model_b_better"].to_numpy(dtype=float),
    )
    tail_counts = np.rint(smaller_tail * n).astype(int)
    table["bootstrap_p_two_sided_approx"] = np.minimum(1.0, 2 * (tail_counts + 1) / (n + 1))
    holm, bh = adjust_pvalues(table["bootstrap_p_two_sided_approx"])
    table["p_holm_all_60"] = holm
    table["p_bh_all_60"] = bh
    table["nominal_ci_excludes_zero"] = (table["ci_low"] > 0) | (table["ci_high"] < 0)
    table["survives_holm_005"] = table["p_holm_all_60"] < 0.05
    table["survives_bh_005"] = table["p_bh_all_60"] < 0.05
    benefit_ci_low = np.where(table["higher_is_better"], table["ci_low"], -table["ci_high"])
    benefit_ci_high = np.where(table["higher_is_better"], table["ci_high"], -table["ci_low"])
    table["benefit_ci_low"] = benefit_ci_low
    table["benefit_ci_high"] = benefit_ci_high
    return table


def build_conclusion_matrix(
    episode: pd.DataFrame,
    equal_patient: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for task in TASK_LABELS:
        for metric in METRICS:
            for protocol, source in [
                ("Episode-weighted ensemble", episode),
                ("Equal-patient ensemble", equal_patient),
            ]:
                row = source[
                    (source["task"] == task)
                    & (source["comparison"] == "Full4096")
                    & (source["metric"] == metric)
                ].iloc[0]
                benefit = float(row["benefit_delta"])
                supported = bool((row["ci_low"] > 0) or (row["ci_high"] < 0))
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "protocol": protocol,
                        "preference_value": 1 if benefit > 1e-12 else (-1 if benefit < -1e-12 else 0),
                        "preferred": "compressed" if benefit > 1e-12 else ("raw" if benefit < -1e-12 else "tie"),
                        "nominal_ci_excludes_zero": supported,
                        "details": f"benefit_delta={benefit:.8g}",
                    }
                )

            seed = seed_summary[
                (seed_summary["task"] == task)
                & (seed_summary["comparison"] == "Full4096")
                & (seed_summary["metric"] == metric)
            ].iloc[0]
            n_a = int(seed["n_seeds_model_a_better"])
            n_b = int(seed["n_seeds_model_b_better"])
            value = 1 if n_a > n_b else (-1 if n_b > n_a else 0)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "protocol": "Seed majority",
                    "preference_value": value,
                    "preferred": "compressed" if value > 0 else ("raw" if value < 0 else "tie"),
                    "nominal_ci_excludes_zero": bool(seed["all_seeds_same_direction"]),
                    "details": f"compressed={n_a}; raw={n_b}; equal={int(seed['n_seeds_equal'])}",
                }
            )
    return pd.DataFrame(rows)


def strict_copy_forward_tables(
    ensemble: pd.DataFrame,
    eligible: pd.DataFrame,
    bootstrap_seed: int,
    n_bootstrap: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    affected = eligible.loc[
        eligible["visit_time"] >= eligible["prediction_time"], ["task", "row_id"]
    ].drop_duplicates()
    strict = ensemble.merge(
        affected.assign(_affected=True),
        on=["task", "row_id"],
        how="left",
        validate="many_to_one",
    )
    strict = strict[strict["_affected"].isna()].drop(columns="_affected")

    probability_rows = []
    top10_rows = []
    for (task, version, fraction), part in strict.groupby(
        ["task", "compression_version", "requested_copy_fraction"], sort=True
    ):
        baseline = part["risk_calibrated_at_0"].to_numpy(dtype=float)
        perturbed = part["risk_calibrated"].to_numpy(dtype=float)
        absolute_delta = np.abs(perturbed - baseline)
        probability_rows.append(
            {
                "task": task,
                "compression_version": version,
                "copy_fraction": float(fraction),
                "n_episodes": int(len(part)),
                "n_patients": int(part["subject_id"].nunique()),
                "n_positive": int(part["y_true"].sum()),
                "mean_abs_delta_probability": float(absolute_delta.mean()),
                "median_abs_delta_probability": float(np.median(absolute_delta)),
                "p95_abs_delta_probability": float(np.quantile(absolute_delta, 0.95)),
                "max_abs_delta_probability": float(absolute_delta.max()),
                "spearman_risk_vs_0": float(pd.Series(baseline).corr(pd.Series(perturbed), method="spearman")),
            }
        )
        baseline_frame = part[["row_id"]].copy()
        baseline_frame["risk"] = baseline
        perturbed_frame = part[["row_id"]].copy()
        perturbed_frame["risk"] = perturbed
        baseline_set = top_fraction_set(baseline_frame, "risk")
        perturbed_set = top_fraction_set(perturbed_frame, "risk")
        intersection = len(baseline_set & perturbed_set)
        top10_rows.append(
            {
                "task": task,
                "compression_version": version,
                "copy_fraction": float(fraction),
                "top_k": len(baseline_set),
                "n_overlap": intersection,
                "retention_fraction": intersection / len(baseline_set),
                "jaccard": intersection / len(baseline_set | perturbed_set),
                "churn_fraction": len(baseline_set - perturbed_set) / len(baseline_set),
            }
        )

    bootstrap_rows = []
    rng = np.random.default_rng(bootstrap_seed)
    for task in sorted(strict["task"].unique()):
        for fraction in [0.25, 0.50, 1.00]:
            raw = strict[
                (strict["task"] == task)
                & (strict["compression_version"] == "raw_4096")
                & np.isclose(strict["requested_copy_fraction"], fraction)
            ][["row_id", "subject_id", "abs_delta_risk_vs_0"]].rename(
                columns={"abs_delta_risk_vs_0": "raw_abs_delta"}
            )
            compressed = strict[
                (strict["task"] == task)
                & (strict["compression_version"] == "condition_era_90_backfill_4096")
                & np.isclose(strict["requested_copy_fraction"], fraction)
            ][["row_id", "subject_id", "abs_delta_risk_vs_0"]].rename(
                columns={"abs_delta_risk_vs_0": "compressed_abs_delta"}
            )
            paired = raw.merge(
                compressed,
                on=["row_id", "subject_id"],
                validate="one_to_one",
            )
            paired["delta"] = paired["raw_abs_delta"] - paired["compressed_abs_delta"]
            patient = paired.groupby("subject_id", as_index=False).agg(
                delta_sum=("delta", "sum"),
                n_episodes=("delta", "size"),
            )
            delta_sum = patient["delta_sum"].to_numpy(dtype=float)
            counts = patient["n_episodes"].to_numpy(dtype=float)
            values = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                sample = rng.integers(0, len(patient), size=len(patient))
                values[index] = delta_sum[sample].sum() / counts[sample].sum()
            bootstrap_rows.append(
                {
                    "task": task,
                    "copy_fraction": fraction,
                    "comparison": "raw_abs_change_minus_compressed_abs_change",
                    "positive_means": "compressed representation is less sensitive",
                    "mean_abs_delta_raw": float(paired["raw_abs_delta"].mean()),
                    "mean_abs_delta_compressed": float(paired["compressed_abs_delta"].mean()),
                    "point_mean": float(paired["delta"].mean()),
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_std": float(values.std(ddof=1)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "fraction_bootstrap_positive": float(np.mean(values > 0)),
                    "n_patients": int(paired["subject_id"].nunique()),
                    "n_episodes": int(len(paired)),
                    "n_bootstrap": n_bootstrap,
                }
            )

    strict_cohort = (
        strict[strict["requested_copy_fraction"] == 0]
        .drop_duplicates(["task", "row_id"])
        .groupby("task", as_index=False)
        .agg(
            n_episodes=("row_id", "size"),
            n_patients=("subject_id", "nunique"),
            n_positive=("y_true", "sum"),
        )
    )
    strict_cohort["event_rate"] = strict_cohort["n_positive"] / strict_cohort["n_episodes"]
    return (
        pd.DataFrame(probability_rows),
        pd.DataFrame(top10_rows),
        pd.DataFrame(bootstrap_rows),
        strict_cohort,
    )


def audit_copy_forward(
    package_root: Path,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inference = package_root / "artifacts/state_or_space_copy_forward/inference"
    per_seed = pd.read_csv(inference / "copy_forward_predictions/copy_forward_predictions.csv")
    ensemble = pd.read_csv(
        inference / "copy_forward_ensemble_predictions/copy_forward_ensemble_predictions.csv"
    )
    eligible = pd.read_csv(inference / "copy_forward_eligible_visits/copy_forward_eligible_visits.csv")

    group_keys = [
        "task",
        "compression_version",
        "representation",
        "model",
        "requested_copy_fraction",
        "row_id",
        "subject_id",
        "y_true",
        "candidate_code",
        "n_existing_visits",
        "n_eligible_visits",
        "n_copied_visits",
        "realized_copy_fraction",
    ]
    recomputed = (
        per_seed.groupby(group_keys, dropna=False, as_index=False)
        .agg(
            risk_calibrated_recomputed=("risk_calibrated", "mean"),
            risk_std_recomputed=("risk_calibrated", "std"),
            n_seeds_recomputed=("seed", "nunique"),
        )
    )
    merged = recomputed.merge(
        ensemble,
        on=group_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    risk_diff = float(
        np.nanmax(
            np.abs(
                merged["risk_calibrated_recomputed"].to_numpy(dtype=float)
                - merged["risk_calibrated"].to_numpy(dtype=float)
            )
        )
    )
    std_diff = float(
        np.nanmax(
            np.abs(
                merged["risk_std_recomputed"].to_numpy(dtype=float)
                - merged["risk_std_across_seeds"].to_numpy(dtype=float)
            )
        )
    )

    count_expected = np.ceil(
        ensemble["requested_copy_fraction"].to_numpy(dtype=float)
        * ensemble["n_eligible_visits"].to_numpy(dtype=int)
    ).astype(int)
    copied_count_issues = int(
        np.sum(count_expected != ensemble["n_copied_visits"].to_numpy(dtype=int))
    )
    monotonic_issues = 0
    for _, part in ensemble.groupby(["task", "compression_version", "row_id"], sort=False):
        ordered = part.sort_values("requested_copy_fraction")
        if np.any(np.diff(ordered["n_copied_visits"].to_numpy(dtype=int)) < 0):
            monotonic_issues += 1

    eligible["prediction_time"] = pd.to_datetime(eligible["prediction_time"], errors="raise")
    eligible["visit_time"] = pd.to_datetime(eligible["visit_time"], errors="raise")
    same_time_visits = int((eligible["visit_time"] == eligible["prediction_time"]).sum())
    future_visits = int((eligible["visit_time"] > eligible["prediction_time"]).sum())
    rank_issues = 0
    for _, part in eligible.groupby(["task", "row_id"], sort=False):
        ranks = np.sort(part["eligible_rank_zero_based"].to_numpy(dtype=int))
        if not np.array_equal(ranks, np.arange(len(part))) or len(part) != int(part["n_eligible_visits"].iloc[0]):
            rank_issues += 1

    cohort = pd.read_csv(inference / "copy_forward_cohort_summary/copy_forward_cohort_summary.csv")
    heldout_counts = {
        task: {
            "n": int(part["row_id"].nunique()),
            "n_positive": int(part.drop_duplicates("row_id")["y_true"].sum()),
            "event_rate": float(part.drop_duplicates("row_id")["y_true"].mean()),
        }
        for task, part in per_seed[per_seed["requested_copy_fraction"] == 0].groupby("task")
    }
    cohort = cohort.copy()
    cohort["event_rate"] = cohort["n_positive"] / cohort["n_episodes"]
    strict_probability, strict_top10, strict_bootstrap, strict_cohort = strict_copy_forward_tables(
        ensemble,
        eligible,
        bootstrap_seed=bootstrap_seed,
    )
    summary = {
        "per_seed_rows": int(len(per_seed)),
        "ensemble_rows": int(len(ensemble)),
        "ensemble_merge_counts": merged["_merge"].value_counts().to_dict(),
        "ensemble_risk_max_abs_diff": risk_diff,
        "ensemble_std_max_abs_diff": std_diff,
        "copied_count_issues": copied_count_issues,
        "nonmonotonic_copy_count_groups": monotonic_issues,
        "eligible_rank_groups_with_issues": rank_issues,
        "eligible_visits_equal_prediction_time": same_time_visits,
        "eligible_visits_after_prediction_time": future_visits,
        "episodes_with_equal_time_eligible_visit": int(
            eligible.loc[eligible["visit_time"] == eligible["prediction_time"], ["task", "row_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "stress_cohort_from_zero_fraction": heldout_counts,
    }
    return cohort, summary, strict_probability, strict_top10, strict_bootstrap, strict_cohort


def compare_repository_snapshot(package_root: Path, repo_root: Path) -> pd.DataFrame:
    snapshot = package_root / "repository_snapshot"
    rows = []
    for folder in ["final_exps", "configs", "scripts"]:
        for source in sorted((snapshot / folder).rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(snapshot)
            current = repo_root / relative
            if not current.is_file():
                status = "missing_current"
            else:
                source_text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
                current_text = current.read_text(encoding="utf-8").replace("\r\n", "\n")
                status = "same_normalized" if source_text == current_text else "changed_content"
            rows.append({"relative_path": relative.as_posix(), "status": status})
    return pd.DataFrame(rows)


def plot_conclusion_matrix(matrix: pd.DataFrame, output_dir: Path) -> None:
    protocols = ["Episode-weighted ensemble", "Equal-patient ensemble", "Seed majority"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
    for axis, task in zip(axes, TASK_LABELS):
        values = np.zeros((len(METRICS), len(protocols)))
        annotations: list[list[str]] = []
        for row_index, metric in enumerate(METRICS):
            annotation_row = []
            for column_index, protocol in enumerate(protocols):
                item = matrix[
                    (matrix["task"] == task)
                    & (matrix["metric"] == metric)
                    & (matrix["protocol"] == protocol)
                ].iloc[0]
                values[row_index, column_index] = int(item["preference_value"])
                label = "Compressed" if item["preference_value"] > 0 else (
                    "Raw" if item["preference_value"] < 0 else "Tie"
                )
                if bool(item["nominal_ci_excludes_zero"]):
                    label += "*"
                annotation_row.append(label)
            annotations.append(annotation_row)
        color_by_preference = {-1: "#b5543c", 0: "#f3f3f3", 1: "#2f6f9f"}
        for row_index in range(len(METRICS)):
            for column_index in range(len(protocols)):
                axis.add_patch(
                    matplotlib.patches.Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor=color_by_preference[int(values[row_index, column_index])],
                        edgecolor="white",
                        linewidth=1.5,
                    )
                )
        axis.set_xlim(-0.5, len(protocols) - 0.5)
        axis.set_ylim(len(METRICS) - 0.5, -0.5)
        axis.set_aspect("auto")
        axis.set_title(TASK_LABELS[task], fontsize=11)
        axis.set_xticks(range(len(protocols)), ["Episode\nensemble", "Equal-patient\nensemble", "Seed\nmajority"])
        axis.set_yticks(range(len(METRICS)), [METRIC_LABELS[item] for item in METRICS])
        axis.tick_params(axis="both", labelsize=8)
        for row_index in range(len(METRICS)):
            for column_index in range(len(protocols)):
                axis.text(
                    column_index,
                    row_index,
                    annotations[row_index][column_index],
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#111111",
                )
    fig.suptitle("Preferred 4,096-event representation depends on evaluation protocol", fontsize=12)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_evaluation_conclusion_matrix.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_copy_forward(probability: pd.DataFrame, top10: pd.DataFrame, output_dir: Path) -> None:
    versions = ["raw_4096", "condition_era_90_backfill_4096"]
    labels = {"raw_4096": "Raw", "condition_era_90_backfill_4096": "Era+backfill"}
    colors = {"raw_4096": "#333333", "condition_era_90_backfill_4096": "#2f6f9f"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    for column, task in enumerate(TASK_LABELS):
        for version in versions:
            part = probability[
                (probability["task"] == task) & (probability["compression_version"] == version)
            ].sort_values("copy_fraction")
            axes[0, column].plot(
                100 * part["copy_fraction"],
                part["mean_abs_delta_probability"],
                marker="o",
                linewidth=2,
                label=labels[version],
                color=colors[version],
            )
            top_part = top10[
                (top10["task"] == task) & (top10["compression_version"] == version)
            ].sort_values("copy_fraction")
            axes[1, column].plot(
                100 * top_part["copy_fraction"],
                top_part["churn_fraction"],
                marker="o",
                linewidth=2,
                label=labels[version],
                color=colors[version],
            )
        axes[0, column].set_title(TASK_LABELS[task], fontsize=11)
        axes[0, column].set_ylabel("Mean absolute risk change")
        axes[1, column].set_ylabel("Top-10% episode churn")
        axes[1, column].set_xlabel("Eligible later visits receiving copy-forward (%)")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].set_xticks([0, 25, 50, 100])
            axes[row, column].tick_params(labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen-model response to strictly pre-prediction diagnosis copy-forward", fontsize=12)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_copy_forward_stress_test.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(calibration_bins_frame: pd.DataFrame, output_dir: Path) -> None:
    labels = {"raw_4096": "Raw", "condition_era_90_backfill_4096": "Persistence-aware"}
    colors = {"raw_4096": "#333333", "condition_era_90_backfill_4096": "#2f6f9f"}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    for axis, task in zip(axes, TASK_LABELS):
        task_frame = calibration_bins_frame[calibration_bins_frame["task"] == task]
        maximum = 0.0
        for version in PRIMARY_VERSIONS:
            part = task_frame[task_frame["compression_version"] == version]
            maximum = max(maximum, float(part[["mean_predicted", "observed_rate"]].to_numpy().max()))
            axis.plot(
                part["mean_predicted"],
                part["observed_rate"],
                marker="o",
                linewidth=1.8,
                label=labels[version],
                color=colors[version],
            )
        limit = min(1.0, maximum * 1.12)
        axis.plot([0, limit], [0, limit], linestyle="--", color="#888888", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(TASK_LABELS[task], fontsize=11)
        axis.set_xlabel("Mean predicted risk")
        axis.set_ylabel("Observed event rate")
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out calibration by equal-frequency risk decile", fontsize=12)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_calibration_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    hashes = verify_package_hashes(package_root)
    hashes.to_csv(output_dir / "package_hash_audit.csv", index=False)

    wide, ensemble, stored_metrics = load_prediction_files(package_root)
    recomputed_metrics, metric_check, prediction_summary = audit_predictions(
        wide, ensemble, stored_metrics
    )
    recomputed_metrics.to_csv(output_dir / "recomputed_ensemble_metrics.csv", index=False)
    metric_check.to_csv(output_dir / "metric_reproduction_check.csv", index=False)

    calibration, bins = analyze_calibration(
        ensemble,
        n_bootstrap=args.calibration_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    calibration.to_csv(output_dir / "calibration_summary.csv", index=False)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)

    seed_pairs, seed_top10_summary, cross_representation = analyze_seed_top10(wide, ensemble)
    seed_pairs.to_csv(output_dir / "seed_top10_pairwise.csv", index=False)
    seed_top10_summary.to_csv(output_dir / "seed_top10_summary.csv", index=False)
    cross_representation.to_csv(output_dir / "ensemble_top10_cross_representation.csv", index=False)

    analysis_root = package_root / "artifacts/ehrshot_state_or_space_final_analysis_5seeds_wide"
    episode = pd.read_csv(analysis_root / "paired_patient_bootstrap_deltas.csv")
    equal_patient = pd.read_csv(analysis_root / "equal_patient_weight_paired_bootstrap_deltas.csv")
    last_episode = pd.read_csv(analysis_root / "last_episode_ensemble_metrics.csv")
    seed_summary = pd.read_csv(analysis_root / "seed_direction_summary.csv")

    multiplicity = multiplicity_table(episode)
    multiplicity.to_csv(output_dir / "multiplicity_sensitivity.csv", index=False)
    conclusion_matrix = build_conclusion_matrix(episode, equal_patient, seed_summary)
    conclusion_matrix.to_csv(output_dir / "evaluation_conclusion_matrix.csv", index=False)

    last_episode_validity = (
        last_episode.groupby("task", as_index=False)
        .agg(
            n_examples=("n", "min"),
            n_patients=("n_patients", "min"),
            n_positive=("n_positive", "min"),
            event_rate=("event_rate", "min"),
        )
    )
    last_episode_validity["warning"] = np.where(
        last_episode_validity["n_positive"] < 20,
        "too_few_positive_events_for_stable_comparison",
        "none",
    )
    last_episode_validity.to_csv(output_dir / "last_episode_validity.csv", index=False)

    (
        copy_forward_cohort,
        copy_forward_summary,
        strict_probability,
        strict_top10,
        strict_bootstrap,
        strict_cohort,
    ) = audit_copy_forward(package_root, bootstrap_seed=args.bootstrap_seed)
    copy_forward_cohort.to_csv(output_dir / "copy_forward_cohort_recomputed.csv", index=False)
    strict_probability.to_csv(
        output_dir / "copy_forward_strict_pre_prediction_probability.csv", index=False
    )
    strict_top10.to_csv(output_dir / "copy_forward_strict_pre_prediction_top10.csv", index=False)
    strict_bootstrap.to_csv(
        output_dir / "copy_forward_strict_pre_prediction_bootstrap.csv", index=False
    )
    strict_cohort.to_csv(output_dir / "copy_forward_strict_pre_prediction_cohort.csv", index=False)

    repository_summary: dict[str, Any] | None = None
    if args.repo_root:
        repo_comparison = compare_repository_snapshot(package_root, args.repo_root.resolve())
        repo_comparison.to_csv(output_dir / "repository_snapshot_comparison.csv", index=False)
        repository_summary = repo_comparison["status"].value_counts().to_dict()

    clearml_summary: dict[str, Any] | None = None
    if args.clearml_audit and args.clearml_audit.is_file():
        clearml = json.loads(args.clearml_audit.read_text(encoding="utf-8"))
        clearml_summary = {
            "n_requested": clearml.get("n_requested"),
            "n_fetched": clearml.get("n_fetched"),
            "n_failures": clearml.get("n_failures"),
            "statuses": sorted({str(item.get("status")) for item in clearml.get("tasks", [])}),
        }

    plot_conclusion_matrix(conclusion_matrix, figures_dir)
    plot_copy_forward(strict_probability, strict_top10, figures_dir)
    plot_calibration(bins, figures_dir)

    summary = {
        "package_hashes": hashes["status"].value_counts().to_dict(),
        "prediction_audit": prediction_summary,
        "copy_forward_audit": copy_forward_summary,
        "copy_forward_strict_pre_prediction_cohort": strict_cohort.to_dict(orient="records"),
        "multiplicity": {
            "n_comparisons": int(len(multiplicity)),
            "nominal_ci_excludes_zero": int(multiplicity["nominal_ci_excludes_zero"].sum()),
            "survives_holm_005": int(multiplicity["survives_holm_005"].sum()),
            "survives_bh_005": int(multiplicity["survives_bh_005"].sum()),
            "p_values_are_approximate": True,
        },
        "last_episode_validity": last_episode_validity.to_dict(orient="records"),
        "repository_snapshot": repository_summary,
        "clearml": clearml_summary,
        "calibration_bootstrap_requested": args.calibration_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
