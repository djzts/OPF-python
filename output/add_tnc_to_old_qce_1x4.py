#!/usr/bin/env python
# coding: utf-8

"""Overlay first-230-step TNC diagnostics on the old 3-bus QCE 1x4 figure.

The QCE curves are read from the existing old-diagnostics CSV.  The TNC curve
is rerun from the QCE coarse solution at iteration 0 so the overlay starts from
the same point as the plotted QCE trajectory.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent
for path in (ROOT, OUT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import solve_3bus_classical_methods as classical  # noqa: E402
from plot_qhd_convergence_diagnostics import (  # noqa: E402
    load_case,
    record_iteration_metrics,
    solve_standard_acopf,
)
from plot_qhd_convergence_diagnostics_QCE import (  # noqa: E402
    _positive,
    _series,
    parse_qce_coarse_log,
)


MAX_TNC_ITERATION = 230
CASE_NAME = "3-bus"
OLD_ANALYSIS_DIR = (
    ROOT
    / "logs"
    / "old_diagnostics"
    / "Buses-3_06-22-2026_05-08-43_QCE_analysis"
)
QCE_LOG = ROOT / "logs" / "QCE_result" / "Buses-3_06-22-2026_05-08-43.txt"
QCE_METRICS_CSV = (
    OLD_ANALYSIS_DIR / "Buses-3_06-22-2026_05-08-43_QCE_coarse_only_metrics.csv"
)
TNC_METRICS_CSV = (
    OLD_ANALYSIS_DIR
    / "Buses-3_06-22-2026_05-08-43_TNC_from_QCE_iter0_first230_metrics.csv"
)
TARGET_PNG = OLD_ANALYSIS_DIR / "3-bus_QCE_coarse_only_diagnostics_1x4.png"
TARGET_PDF = OLD_ANALYSIS_DIR / "3-bus_QCE_coarse_only_diagnostics_1x4.pdf"

TNC_COLOR = "#F58518"
TNC_ALT_COLOR = "#E45756"
TITLE_FONTSIZE = 20
AXIS_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def read_qce_history(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    history: list[dict[str, object]] = []
    for row in rows:
        parsed: dict[str, object] = {}
        for key, value in row.items():
            if key == "answer_type":
                parsed[key] = value
            elif key == "iteration":
                parsed[key] = int(value)
            else:
                parsed[key] = float(value)
        history.append(parsed)
    return history


def write_tnc_history(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "iteration",
        "solver_iteration",
        "stage",
        "tnc_objective",
        "standard_objective",
        "objective_difference",
        "bus_state_l2_distance",
        "branch_flow_l2_distance",
        "combined_l2_distance",
        "max_constraint_residual",
        "l2_constraint_residual",
        "active_load_supplied_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    *,
    case,
    standard_solution: dict[str, object],
    iteration: int,
    solver_iteration: int,
    stage: str,
    x: np.ndarray,
) -> dict[str, object]:
    metrics = record_iteration_metrics(
        {
            "iteration": iteration,
            "answer_type": "tnc",
            "objective": float(case.objective(x)),
            "x": np.asarray(x, dtype=float),
        },
        standard_solution,
        case,
    )
    return {
        "iteration": iteration,
        "solver_iteration": solver_iteration,
        "stage": stage,
        "tnc_objective": metrics["qhd_objective"],
        "standard_objective": metrics["standard_objective"],
        "objective_difference": metrics["objective_difference"],
        "bus_state_l2_distance": metrics["bus_state_l2_distance"],
        "branch_flow_l2_distance": metrics["branch_flow_l2_distance"],
        "combined_l2_distance": metrics["combined_l2_distance"],
        "max_constraint_residual": metrics["max_constraint_residual"],
        "l2_constraint_residual": metrics["l2_constraint_residual"],
        "active_load_supplied_percent": metrics["active_load_supplied_percent"],
    }


def build_standard_solution(case, records: list[dict[str, object]]) -> dict[str, object]:
    warm_records = sorted(records, key=lambda item: item.get("log_l2_norm_h", float("inf")))
    warm_starts = [item["x"] for item in warm_records[:8]]
    warm_starts.append(warm_records[-1]["x"])
    return solve_standard_acopf(case, warm_starts=warm_starts)


def run_tnc_from_qce_iter0(case, standard_solution: dict[str, object], x0: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        metric_row(
            case=case,
            standard_solution=standard_solution,
            iteration=0,
            solver_iteration=0,
            stage="initial QCE iter 0",
            x=x0,
        )
    ]

    original_record_history = classical.record_history

    def capture_record_history(history, case_arg, method, start_name, iteration, x, stage=""):
        if (
            method == "Truncated Newton (TNC penalty)"
            and start_name == "qce_iter0"
            and 1 <= int(iteration) <= MAX_TNC_ITERATION
        ):
            rows.append(
                metric_row(
                    case=case_arg,
                    standard_solution=standard_solution,
                    iteration=int(iteration),
                    solver_iteration=int(iteration),
                    stage=str(stage),
                    x=np.asarray(x, dtype=float),
                )
            )
        original_record_history(history, case_arg, method, start_name, iteration, x, stage)

    classical.record_history = capture_record_history
    try:
        scratch_history: list[dict[str, object]] = []
        classical.solve_truncated_newton(case, x0, "qce_iter0", scratch_history)
    finally:
        classical.record_history = original_record_history

    rows.sort(key=lambda item: int(item["iteration"]))
    return rows[: MAX_TNC_ITERATION + 1]


def tnc_series(history: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(history, key=lambda item: int(item["iteration"]))
    return (
        np.asarray([int(item["iteration"]) for item in ordered], dtype=int),
        np.asarray([float(item[key]) for item in ordered], dtype=float),
    )


def align_active_load_axis_with_objective_zero(ax, ax2) -> None:
    """Set right-axis lower bound to 0 while aligning right 100 with left 0."""
    left_min, left_max = ax.get_ylim()
    if left_max <= left_min or not (left_min < 0.0 < left_max):
        ax2.set_ylim(0.0, 100.0)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        return

    zero_position = (0.0 - left_min) / (left_max - left_min)
    right_upper = max(100.0 / zero_position, 100.0)
    ax2.set_ylim(0.0, right_upper)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])


def backup_original_outputs() -> None:
    for target in (TARGET_PNG, TARGET_PDF):
        if not target.exists():
            continue
        backup = target.with_name(f"{target.stem}_without_tnc{target.suffix}")
        if not backup.exists():
            shutil.copy2(target, backup)


def plot_overlay(
    qce_history: list[dict[str, object]],
    tnc_history: list[dict[str, object]],
    bound_min_factor: float = 0.01,
) -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(32, 7.2), constrained_layout=True)

    ax = axes[0]
    x, y = _series(qce_history, "objective_difference")
    ax.plot(x, y, color="tab:blue", lw=1.6, label="Coarse objective difference")
    xt, yt = tnc_series(tnc_history, "objective_difference")
    ax.plot(xt, yt, color=TNC_COLOR, lw=1.8, ls="-", label="TNC objective difference")
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="Reference objective")

    ax2 = ax.twinx()
    x, supplied = _series(qce_history, "active_load_supplied_percent")
    ax2.plot(
        x,
        supplied,
        color="tab:cyan",
        lw=1.4,
        label="Coarse active load supplied",
    )
    xt, supplied_tnc = tnc_series(tnc_history, "active_load_supplied_percent")
    ax2.plot(
        xt,
        supplied_tnc,
        color=TNC_ALT_COLOR,
        lw=1.5,
        ls=":",
        label="TNC active load supplied",
    )
    ax2.axhline(
        100.0,
        color="tab:gray",
        lw=1.0,
        ls=":",
        label="Reference load = 100%",
    )
    align_active_load_axis_with_objective_zero(ax, ax2)
    ax.set_title("Objective difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coarse/TNC objective - reference objective")
    ax2.set_ylabel("Active load supplied (%)")
    ax.grid(True, ls=":", alpha=0.55)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=LEGEND_FONTSIZE)

    ax = axes[1]
    for key, label, color, linestyle in [
        ("bus_state_l2_distance", "Bus state", "tab:orange", "-"),
        ("branch_flow_l2_distance", "Branch flow", "tab:green", "--"),
        ("combined_l2_distance", "Combined", "tab:red", "-."),
    ]:
        x, y = _series(qce_history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    for key, label, linestyle in [
        ("bus_state_l2_distance", "TNC bus state", ":"),
        ("branch_flow_l2_distance", "TNC branch flow", "--"),
        ("combined_l2_distance", "TNC combined", "-"),
    ]:
        xt, yt = tnc_series(tnc_history, key)
        ax.semilogy(xt, _positive(yt), color=TNC_COLOR, ls=linestyle, lw=1.8, label=label)
    ax.set_title("Distance to reference solution")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 distance")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    ax = axes[2]
    for key, label, color, linestyle in [
        ("max_constraint_residual", "Coarse max |h|", "tab:purple", "-"),
        ("l2_constraint_residual", "Coarse ||h||2", "tab:brown", "--"),
    ]:
        x, y = _series(qce_history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    for key, label, linestyle in [
        ("max_constraint_residual", "TNC max |h|", "-"),
        ("l2_constraint_residual", "TNC ||h||2", "--"),
    ]:
        xt, yt = tnc_series(tnc_history, key)
        ax.semilogy(xt, _positive(yt), color=TNC_COLOR, ls=linestyle, lw=1.8, label=label)
    ax.set_title("Constraint residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    ax = axes[3]
    x, y = _series(qce_history, "cumulative_bound_factor")
    ax.semilogy(x, _positive(y), color="tab:olive", lw=1.6, label="Bound width factor")
    ax.axhline(
        bound_min_factor,
        color="black",
        lw=1.0,
        ls="--",
        label=f"Minimum = {bound_min_factor:g}",
    )
    ax.set_title("Cumulative bound shrink factor")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of original bound width")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    for panel in axes:
        panel.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        panel.xaxis.label.set_size(AXIS_FONTSIZE)
        panel.yaxis.label.set_size(AXIS_FONTSIZE)
        panel.title.set_size(TITLE_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax2.yaxis.label.set_size(AXIS_FONTSIZE)

    fig.savefig(TARGET_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(TARGET_PDF, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    case = load_case(3, ROOT)
    qce_records = parse_qce_coarse_log(QCE_LOG, case)
    if not qce_records:
        raise RuntimeError(f"No QCE records parsed from {QCE_LOG}")
    qce_records.sort(key=lambda item: int(item["iteration"]))

    standard_solution = build_standard_solution(case, qce_records)
    qce_history = read_qce_history(QCE_METRICS_CSV)
    x0 = np.asarray(qce_records[0]["x"], dtype=float)
    tnc_history = run_tnc_from_qce_iter0(case, standard_solution, x0)
    write_tnc_history(tnc_history, TNC_METRICS_CSV)

    backup_original_outputs()
    plot_overlay(qce_history, tnc_history)
    print(f"Wrote TNC metrics: {TNC_METRICS_CSV}")
    print(f"Wrote overlay PNG: {TARGET_PNG}")
    print(f"Wrote overlay PDF: {TARGET_PDF}")


if __name__ == "__main__":
    main()
