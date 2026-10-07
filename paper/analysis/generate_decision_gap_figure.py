from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_rows(analysis_dir: Path) -> tuple[list, list]:
    with (analysis_dir / "multiplicity_sensitivity.csv").open(encoding="utf-8") as source:
        comparisons = list(csv.DictReader(source))
    metrics = [("auroc", "AUROC"), ("auprc", "AP"), ("brier", "Brier"),
               ("logloss", "Log loss"), ("top_10pct_precision", "P@10%")]
    rows = []
    for task, label, color in [("guo_readmission", "Readm.", "#17324D"),
                                ("guo_icu", "ICU", "#0072B2")]:
        for metric, metric_label in metrics:
            matches = [row for row in comparisons
                       if row["task"] == task and row["comparison"] == "Full4096"
                       and row["metric"] == metric]
            if len(matches) != 1:
                raise ValueError("Missing or duplicate primary comparison")
            row = matches[0]
            rows.append((f"{label}  {metric_label}", float(row["model_b_value"]),
                         float(row["benefit_delta"]), float(row["benefit_ci_low"]),
                         float(row["benefit_ci_high"]), color))
    with (analysis_dir / "ensemble_top10_cross_representation.csv").open(encoding="utf-8") as source:
        sets = {row["task"]: row for row in csv.DictReader(source)}
    set_rows = [(label, int(sets[task]["top_k"]), int(sets[task]["raw_only"]),
                 float(sets[task]["jaccard"]))
                for task, label in [("guo_readmission", "Readmission"),
                                    ("guo_icu", "ICU transfer")]]
    return rows, set_rows


def render(output: Path, analysis_dir: Path) -> None:
    rows, set_rows = load_rows(analysis_dir)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, (ax_effect, ax_set) = plt.subplots(
        1,
        2,
        figsize=(13.2, 4.55),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )

    labels = [row[0] for row in rows]
    base = np.array([row[1] for row in rows])
    point = 100 * np.array([row[2] for row in rows]) / base
    low = 100 * np.array([row[3] for row in rows]) / base
    high = 100 * np.array([row[4] for row in rows]) / base
    colors = [row[5] for row in rows]
    y = np.arange(len(rows))

    for idx in range(len(rows)):
        ax_effect.errorbar(
            point[idx],
            y[idx],
            xerr=np.array([[point[idx] - low[idx]], [high[idx] - point[idx]]]),
            fmt="o",
            color=colors[idx],
            markersize=5.8,
            linewidth=1.8,
            capsize=3.2,
        )
    ax_effect.axvline(0, color="#666666", linewidth=1.1, linestyle="--")
    ax_effect.axhline(4.5, color="#D9D9D9", linewidth=1.0)
    ax_effect.set_yticks(y, labels)
    ax_effect.invert_yaxis()
    ax_effect.set_xlim(-14.5, 25.8)
    ax_effect.set_xlabel("Oriented effect relative to Raw (%)")
    ax_effect.set_title(
        "(a) Point estimates agree; paired uncertainty remains",
        loc="left",
        weight="bold",
        y=1.15,
        pad=0,
    )
    ax_effect.text(
        0,
        1.055,
        "All 10 paired 95% CIs include zero",
        transform=ax_effect.transAxes,
        color="#666666",
    )
    ax_effect.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax_effect.tick_params(axis="y", length=0)
    ax_effect.spines["top"].set_visible(False)
    ax_effect.spines["right"].set_visible(False)
    ax_effect.spines["left"].set_visible(False)

    shared_color = "#D5DDE5"
    specific_color = "#D55E00"
    y_set = np.arange(len(set_rows))
    for idx, (task, top_k, unique, jaccard) in enumerate(set_rows):
        specific = 100 * unique / top_k
        shared = 100 - specific
        ax_set.barh(idx, shared, color=shared_color, height=0.80)
        ax_set.barh(idx, specific, left=shared, color=specific_color, height=0.80)
        ax_set.text(
            shared / 2,
            idx - 0.05,
            f"{shared:.1f}% shared",
            ha="center",
            va="center",
            color="#17324D",
            weight="bold",
        )
        ax_set.text(
            shared / 2,
            idx + 0.18,
            f"Jaccard {jaccard:.3f}",
            ha="center",
            va="center",
            color="#666666",
        )
        ax_set.text(
            shared + specific / 2,
            idx,
            f"{specific:.1f}%\nspecific",
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=9.0,
            linespacing=0.85,
        )
    ax_set.set_yticks(y_set, [row[0] for row in set_rows])
    ax_set.invert_yaxis()
    ax_set.set_xlim(0, 104)
    ax_set.set_xticks([0, 20, 40, 60, 80, 100])
    ax_set.set_xlabel("Share of each top-decile episode set (%)")
    ax_set.set_title(
        "(b) Top-decile prioritization sets differ",
        loc="left",
        weight="bold",
        y=1.15,
        pad=0,
    )
    ax_set.text(
        0,
        1.055,
        "Gray: shared  |  Orange: representation-specific",
        transform=ax_set.transAxes,
        color="#666666",
    )
    ax_set.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax_set.set_axisbelow(True)
    ax_set.tick_params(axis="y", length=0)
    ax_set.spines["top"].set_visible(False)
    ax_set.spines["right"].set_visible(False)
    ax_set.spines["left"].set_visible(False)

    fig.subplots_adjust(left=0.125, right=0.995, top=0.78, bottom=0.20, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "aggregate_outputs")
    args = parser.parse_args()
    render(args.output.resolve(), args.analysis_dir.resolve())


if __name__ == "__main__":
    main()
