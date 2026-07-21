#!/usr/bin/env python
# coding: utf-8

"""
Plot a focused 3-bus convergence comparison between the truncated Newton
baseline and the QHD-SB/QCE coarse-only trajectory.

Both curves are explicitly initialized from the same QHD-SB center:
case.build_initial_x0() = [P_G, Q_G, V_R, V_I, V_sq, P_ij, Q_ij, S_tot_sq].
The plotted x-axis uses plot_iteration=0 for this common initial point.

Inputs:
- output/3bus_solver_convergence_history.csv

Outputs:
- output/3bus_tnc_vs_qhdsb_convergence.csv
- output/3bus_tnc_vs_qhdsb_convergence.png
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solve_3bus_classical_methods import (  # noqa: E402
    clip_to_bounds,
    load_case,
    solve_truncated_newton,
)

HISTORY_CSV = OUT_DIR / "3bus_solver_convergence_history.csv"
PLOT_SOURCE_CSV = OUT_DIR / "3bus_tnc_vs_qhdsb_convergence.csv"
PLOT_PATH = OUT_DIR / "3bus_tnc_vs_qhdsb_convergence.png"
MAX_PLOT_ITERATION = 230
RIGHT_MARGIN_ITERATIONS = 80


def read_history() -> list[dict[str, str]]:
    with HISTORY_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_row(
    *,
    case,
    label: str,
    method: str,
    source: str,
    start_name: str,
    plot_iteration: int,
    solver_iteration: int | str,
    stage: str,
    x,
) -> dict[str, object]:
    x = np.asarray(x, dtype=float)
    h = case.h(x)
    return {
        "label": label,
        "method": method,
        "source": source,
        "start_name": start_name,
        "plot_iteration": plot_iteration,
        "solver_iteration": solver_iteration,
        "stage": stage,
        "objective": case.objective(x),
        "l2_h": float(np.linalg.norm(h)),
        "max_abs_h": float(np.max(np.abs(h))),
        "load_supplied_pct": case.active_load_supplied_percent(x),
    }


def run_tnc_from_qhdsb_initial(case, x0) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    solve_truncated_newton(case, x0, "qhdsb_initial_center", history)
    selected: list[dict[str, object]] = [
        metric_row(
            case=case,
            label="Truncated Newton",
            method="Truncated Newton (TNC penalty)",
            source="TNC rerun from the exact same initial center used by QHD-SB",
            start_name="qhdsb_initial_center",
            plot_iteration=0,
            solver_iteration=0,
            stage="initial",
            x=x0,
        )
    ]
    for row in history:
        if row["method"] != "Truncated Newton (TNC penalty)" or row["start_name"] != "qhdsb_initial_center":
            continue
        selected.append(
            {
                "label": "Truncated Newton",
                "method": row["method"],
                "source": "TNC rerun from the exact same initial center used by QHD-SB",
                "start_name": row["start_name"],
                "plot_iteration": int(row["iteration"]),
                "solver_iteration": int(row["iteration"]),
                "stage": row.get("stage", ""),
                "objective": float(row["objective"]),
                "l2_h": float(row["l2_h"]),
                "max_abs_h": float(row["max_abs_h"]),
                "load_supplied_pct": float(row["load_supplied_pct"]),
            }
        )
    return selected


def selected_qhdsb_rows(rows: list[dict[str, str]], case, x0) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    selected.append(
        metric_row(
            case=case,
            label="QHD-SB coarse-only",
            method="QCE coarse-only",
            source="Common QHD-SB initial center before the first beam-search update",
            start_name="QCE beam",
            plot_iteration=0,
            solver_iteration=0,
            stage="initial",
            x=x0,
        )
    )
    for row in rows:
        method = row["method"]
        start = row["start_name"]
        if method == "QCE coarse-only" and start == "QCE beam":
            label = "QHD-SB coarse-only"
            source = "QHD-SB/QCE beam search trajectory"
        else:
            continue
        solver_iteration = int(float(row["iteration"]))
        selected.append(
            {
                "method": method,
                "label": label,
                "source": source,
                "start_name": start,
                "plot_iteration": solver_iteration + 1,
                "solver_iteration": solver_iteration,
                "stage": row.get("stage", ""),
                "objective": float(row["objective"]),
                "l2_h": float(row["l2_h"]),
                "max_abs_h": float(row["max_abs_h"]),
                "load_supplied_pct": float(row["load_supplied_pct"]),
            }
        )
    return selected


def write_source(rows: list[dict[str, object]]) -> None:
    fields = [
        "label",
        "method",
        "source",
        "start_name",
        "plot_iteration",
        "solver_iteration",
        "stage",
        "objective",
        "l2_h",
        "max_abs_h",
        "load_supplied_pct",
    ]
    with PLOT_SOURCE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot(rows: list[dict[str, object]]) -> None:
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
    series_config = {
        "Truncated Newton": {
            "color": "#F58518",
            "linewidth": 3.0,
            "linestyle": "-",
            "best_text_xy": (238, 5.9e-4),
        },
        "QHD-SB coarse-only": {
            "color": "#4C78A8",
            "linewidth": 3.1,
            "linestyle": "-",
            "best_text_xy": (238, 1.8e-3),
        },
    }

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    for label, config in series_config.items():
        series = [row for row in rows if row["label"] == label]
        series = sorted(series, key=lambda row: int(row["plot_iteration"]))
        ax.plot(
            [int(row["plot_iteration"]) for row in series],
            [max(float(row["l2_h"]), 1e-14) for row in series],
            label=label,
            color=config["color"],
            linewidth=config["linewidth"],
            linestyle=config["linestyle"],
        )

        best = min(series, key=lambda row: float(row["l2_h"]))
        ax.scatter(
            [int(best["plot_iteration"])],
            [max(float(best["l2_h"]), 1e-14)],
            color=config["color"],
            edgecolor="white",
            s=90,
            linewidth=1.1,
            zorder=4,
        )
        iteration_label = "initial" if int(best["plot_iteration"]) == 0 else f"iter {int(best['solver_iteration'])}"
        label_prefix = "QHD-SB best" if label.startswith("QHD") else "TNC best"
        ax.annotate(
            f"{label_prefix}\n{float(best['l2_h']):.2e}, {iteration_label}",
            xy=(int(best["plot_iteration"]), max(float(best["l2_h"]), 1e-14)),
            xytext=config["best_text_xy"],
            textcoords="data",
            fontsize=18,
            color=config["color"],
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "->",
                "color": config["color"],
                "linewidth": 1.5,
                "shrinkA": 4,
                "shrinkB": 5,
            },
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": config["color"], "alpha": 0.9},
        )

    ax.axhline(1e-3, color="#666666", linewidth=1.4, linestyle="--", alpha=0.75, label="1e-3 residual")
    ax.set_yscale("log")
    ax.set_xlabel("Plot iteration (0 = shared QHD-SB initial center)", fontsize=20, labelpad=10)
    ax.set_ylabel("L2 equality residual", fontsize=20, labelpad=12)
    ax.set_xlim(0, MAX_PLOT_ITERATION + RIGHT_MARGIN_ITERATIONS)
    ax.set_ylim(3.0e-4, 3.0)
    ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=3, width=0.9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=18, frameon=True)
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    case = load_case(3, ROOT)
    bounds = case.bounds()
    x0 = clip_to_bounds(case.build_initial_x0(), bounds)
    rows = run_tnc_from_qhdsb_initial(case, x0)
    rows.extend(selected_qhdsb_rows(read_history(), case, x0))
    rows = [row for row in rows if int(row["plot_iteration"]) <= MAX_PLOT_ITERATION]
    if not rows:
        raise RuntimeError(f"No selected TNC/QHD-SB rows found in {HISTORY_CSV}")
    write_source(rows)
    plot(rows)
    print(f"Wrote {PLOT_PATH}")
    print(f"Wrote {PLOT_SOURCE_CSV}")


if __name__ == "__main__":
    main()
