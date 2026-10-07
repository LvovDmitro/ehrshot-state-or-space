from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


TASKS = ["guo_readmission", "guo_icu"]
TASK_LABELS = {"guo_readmission": "Readm.", "guo_icu": "ICU"}
METRICS = ["auroc", "auprc", "brier", "logloss", "top_10pct_precision"]
METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AP",
    "brier": "Brier",
    "logloss": "Log loss",
    "top_10pct_precision": "P@10%",
}
PROTOCOLS = [
    "Episode-weighted ensemble",
    "Equal-patient ensemble",
    "Seed majority",
]
PROTOCOL_LABELS = ["Episode-\nweighted", "Equal-\npatient", "Seed\nmajority"]

BLUE = "#0072B2"
ORANGE = "#D55E00"
NAVY = "#17324D"
GREEN = "#009E73"
GRID = "#D9D9D9"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def build_preference_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    lookup = {
        (row["task"], row["metric"], row["protocol"]): int(row["preference_value"])
        for row in rows
    }
    matrix = np.array(
        [
            [lookup[(task, metric, protocol)] for protocol in PROTOCOLS]
            for task in TASKS
            for metric in METRICS
        ],
        dtype=int,
    )
    if matrix.shape != (10, 3) or not np.isin(matrix, [-1, 0, 1]).all():
        raise ValueError("Unexpected protocol-preference matrix")
    return matrix


def build_stress_series(
    rows: list[dict[str, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result = {}
    for task in TASKS:
        selected = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: float(row["copy_fraction"]),
        )
        if [float(row["copy_fraction"]) for row in selected] != [0.25, 0.5, 1.0]:
            raise ValueError(f"Unexpected stress fractions for {task}")
        x = np.array([100 * float(row["copy_fraction"]) for row in selected])
        point = 1000 * np.array([float(row["point_mean"]) for row in selected])
        low = 1000 * np.array([float(row["ci_low"]) for row in selected])
        high = 1000 * np.array([float(row["ci_high"]) for row in selected])
        result[task] = (x, point, point - low, high - point)
    return result


def render(analysis_dir: Path, output: Path) -> None:
    preference_rows = read_csv(analysis_dir / "evaluation_conclusion_matrix.csv")
    stress_rows = read_csv(
        analysis_dir / "copy_forward_strict_pre_prediction_bootstrap.csv"
    )
    preference = build_preference_matrix(preference_rows)
    stress = build_stress_series(stress_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, (ax_matrix, ax_stress) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.8),
        gridspec_kw={"width_ratios": [1.03, 1.0]},
    )

    cmap = ListedColormap([ORANGE, "#F3F3F3", BLUE])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax_matrix.imshow(preference, cmap=cmap, norm=norm, aspect="auto")
    ax_matrix.set_title("(a) Protocol comparison", loc="left", weight="bold")
    ax_matrix.set_xticks(range(3), PROTOCOL_LABELS)
    row_labels = [
        f"{TASK_LABELS[task]}  {METRIC_LABELS[metric]}"
        for task in TASKS
        for metric in METRICS
    ]
    ax_matrix.set_yticks(range(10), row_labels)
    ax_matrix.tick_params(axis="both", length=0, pad=6)
    ax_matrix.axhline(4.5, color="white", linewidth=4)
    for row in range(preference.shape[0]):
        for column in range(preference.shape[1]):
            value = preference[row, column]
            label = "Era" if value == 1 else "Raw" if value == -1 else "Tie"
            color = "white" if value else "#444444"
            ax_matrix.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                weight="bold",
            )
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    ax_matrix.legend(
        handles=[
            Patch(facecolor=BLUE, label="Era+backfill"),
            Patch(facecolor=ORANGE, label="Raw"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )

    styles = {
        "guo_readmission": (NAVY, "o", "Readmission"),
        "guo_icu": (GREEN, "s", "ICU transfer"),
    }
    for task in TASKS:
        x, point, low_error, high_error = stress[task]
        color, marker, label = styles[task]
        ax_stress.errorbar(
            x,
            point,
            yerr=np.vstack([low_error, high_error]),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=4,
            label=label,
        )
    ax_stress.axhline(0, color="#555555", linewidth=1.2, linestyle="--")
    ax_stress.set_title("(b) Repetition sensitivity", loc="left", weight="bold")
    ax_stress.set_xlabel("Requested eligible visits (%)")
    ax_stress.set_ylabel(r"Raw - Era mean $|\Delta p|$ ($\times 10^{-3}$)")
    ax_stress.set_xticks([25, 50, 100])
    ax_stress.set_xlim(19, 106)
    ax_stress.set_ylim(-0.9, 1.65)
    ax_stress.grid(axis="y", color=GRID, linewidth=0.8)
    ax_stress.set_axisbelow(True)
    ax_stress.legend(loc="upper left", frameon=False)
    ax_stress.spines["top"].set_visible(False)
    ax_stress.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.17, right=0.99, top=0.89, bottom=0.24, wspace=0.62)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
