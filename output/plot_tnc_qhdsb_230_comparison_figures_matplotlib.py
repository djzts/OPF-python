#!/usr/bin/env python
# coding: utf-8

"""
Matplotlib version of the focused TNC vs QHD-SB comparison figures.

The Pillow-only script is kept as:
- output/plot_tnc_qhdsb_230_comparison_figures.py

This script restores the earlier Matplotlib visual style and writes the same
primary PNG filenames:
- output/3bus_tnc_qhdsb_230_start_metrics.png
- output/3bus_tnc_qhdsb_230_objective_selection_comparison.png
- output/3bus_tnc_qhdsb_230_iteration_metrics.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
START_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_start_metrics.csv"
OBJECTIVE_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_objective_comparison.csv"
ITERATION_CSV = OUT_DIR / "3bus_tnc_qhdsb_230_iteration_metrics.csv"
REFERENCE_OBJECTIVE_CSV = OUT_DIR / "3bus_objective_comparison.csv"

START_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_start_metrics.png"
OBJECTIVE_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_objective_selection_comparison.png"
ITERATION_FIG = OUT_DIR / "3bus_tnc_qhdsb_230_iteration_metrics.png"

TNC_COLOR = "#F58518"
QHDSB_COLOR = "#4C78A8"
REFERENCE_COLOR = "#666666"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def reference_objective() -> float:
    for row in read_csv(REFERENCE_OBJECTIVE_CSV):
        if row.get("method") == "SQP (SLSQP)":
            return float(row["objective"])
    raise RuntimeError(f"SQP reference objective not found in {REFERENCE_OBJECTIVE_CSV}")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def format_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", which="major", length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_start_metrics() -> None:
    rows = read_csv(START_CSV)
    metrics = [
        ("objective", "Normalized\nmonetary units (NMU)"),
        ("l2_h", "L2 equality residual"),
        ("max_abs_h", "Max abs residual"),
        ("load_supplied_pct", "Load supplied (%)"),
    ]
    labels = ["TNC", "QHD-SB"]
    colors = [TNC_COLOR, QHDSB_COLOR]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.4), constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, metrics):
        values = [as_float(row, key) for row in rows]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        format_axes(ax)
        if key in {"l2_h", "max_abs_h"}:
            ax.set_yscale("log")
        for bar, value in zip(bars, values):
            if key in {"l2_h", "max_abs_h"}:
                text = f"{value:.2e}"
            elif key == "load_supplied_pct":
                text = f"{value:.1f}"
            else:
                text = f"{value:.3f}"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
            )
    fig.savefig(START_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_objective_selection_comparison(reference_obj: float) -> None:
    rows = [
        row
        for row in read_csv(OBJECTIVE_CSV)
        if row["selection"]
        in {
            "first_l2_le_1e-3",
            "best_l2_residual",
            "final_plot_iteration_230",
            "min_objective_with_l2_le_1e-3",
        }
    ]
    selection_labels = [
        "First L2<=1e-3",
        "Best L2",
        "Final @230",
        "Min obj\nwith L2<=1e-3",
    ]
    x = np.arange(len(rows))
    width = 0.36

    tnc_obj = [as_float(row, "tnc_objective") for row in rows]
    qhd_obj = [as_float(row, "qhdsb_objective") for row in rows]
    tnc_l2 = [as_float(row, "tnc_l2_h") for row in rows]
    qhd_l2 = [as_float(row, "qhdsb_l2_h") for row in rows]

    fig, (ax_obj, ax_res) = plt.subplots(
        2,
        1,
        figsize=(14.5, 10.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )

    ax_obj.plot(
        x - width / 2,
        tnc_obj,
        marker="o",
        markersize=10,
        linewidth=0,
        color=TNC_COLOR,
        label="Truncated Newton",
    )
    ax_obj.plot(
        x + width / 2,
        qhd_obj,
        marker="o",
        markersize=10,
        linewidth=0,
        color=QHDSB_COLOR,
        label="QHD-SB coarse-only",
    )
    for xpos, low, high in zip(x, np.minimum(tnc_obj, qhd_obj), np.maximum(tnc_obj, qhd_obj)):
        ax_obj.vlines(xpos, low, high, color="#999999", linewidth=1.2, alpha=0.7)
    ax_obj.axhline(reference_obj, color=REFERENCE_COLOR, linewidth=1.6, linestyle="--", label="Optimum reference")
    ax_obj.set_ylabel("Normalized\nmonetary units (NMU)", labelpad=12)
    ax_obj.set_xticks(x)
    ax_obj.set_xticklabels([])
    ax_obj.set_ylim(0.52, 0.55)
    ax_obj.legend(loc="upper right", frameon=True)
    format_axes(ax_obj)

    ax_res.plot(
        x - width / 2,
        tnc_l2,
        marker="o",
        markersize=10,
        linewidth=0,
        color=TNC_COLOR,
        label="Truncated Newton",
    )
    ax_res.plot(
        x + width / 2,
        qhd_l2,
        marker="o",
        markersize=10,
        linewidth=0,
        color=QHDSB_COLOR,
        label="QHD-SB coarse-only",
    )
    for xpos, low, high in zip(x, np.minimum(tnc_l2, qhd_l2), np.maximum(tnc_l2, qhd_l2)):
        ax_res.vlines(xpos, low, high, color="#999999", linewidth=1.2, alpha=0.7)
    ax_res.axhline(1e-3, color=REFERENCE_COLOR, linewidth=1.6, linestyle="--", label="1e-3 residual")
    ax_res.set_yscale("log")
    ax_res.set_ylabel("L2 equality residual")
    ax_res.set_xticks(x)
    ax_res.set_xticklabels(selection_labels)
    format_axes(ax_res)

    fig.savefig(OBJECTIVE_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_metrics(reference_obj: float) -> None:
    rows = read_csv(ITERATION_CSV)
    x = np.array([int(row["plot_iteration"]) for row in rows])
    tnc_l2 = np.array([as_float(row, "tnc_l2_h") for row in rows])
    qhd_l2 = np.array([as_float(row, "qhdsb_l2_h") for row in rows])
    tnc_max = np.array([as_float(row, "tnc_max_abs_h") for row in rows])
    qhd_max = np.array([as_float(row, "qhdsb_max_abs_h") for row in rows])
    tnc_obj = np.array([as_float(row, "tnc_objective") for row in rows])
    qhd_obj = np.array([as_float(row, "qhdsb_objective") for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 13.5), sharex=True, constrained_layout=True)
    ax_l2, ax_max, ax_obj = axes

    ax_l2.plot(x, tnc_l2, color=TNC_COLOR, linewidth=3.0, label="Truncated Newton")
    ax_l2.plot(x, qhd_l2, color=QHDSB_COLOR, linewidth=3.0, label="QHD-SB coarse-only")
    ax_l2.axhline(1e-3, color=REFERENCE_COLOR, linewidth=1.5, linestyle="--", label="1e-3 residual")
    ax_l2.set_yscale("log")
    ax_l2.set_ylabel("L2 equality residual")
    ax_l2.legend(loc="upper right", frameon=True)
    format_axes(ax_l2)

    ax_max.plot(x, tnc_max, color=TNC_COLOR, linewidth=3.0, label="Truncated Newton")
    ax_max.plot(x, qhd_max, color=QHDSB_COLOR, linewidth=3.0, label="QHD-SB coarse-only")
    ax_max.set_yscale("log")
    ax_max.set_ylabel("Max abs residual")
    ax_max.legend(loc="upper right", frameon=True)
    format_axes(ax_max)

    ax_obj.plot(x, tnc_obj, color=TNC_COLOR, linewidth=3.0, label="Truncated Newton")
    ax_obj.plot(x, qhd_obj, color=QHDSB_COLOR, linewidth=3.0, label="QHD-SB coarse-only")
    ax_obj.axhline(reference_obj, color=REFERENCE_COLOR, linewidth=1.5, linestyle="--", label="Optimum reference")
    ax_obj.set_ylabel("Normalized\nmonetary units (NMU)", labelpad=12)
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylim(-0.02, 0.60)
    ax_obj.legend(loc="lower right", frameon=True)
    format_axes(ax_obj)

    fig.savefig(ITERATION_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    ref_obj = reference_objective()
    plot_start_metrics()
    plot_objective_selection_comparison(ref_obj)
    plot_iteration_metrics(ref_obj)
    print(f"Wrote {START_FIG}")
    print(f"Wrote {OBJECTIVE_FIG}")
    print(f"Wrote {ITERATION_FIG}")


if __name__ == "__main__":
    main()
