#!/usr/bin/env python
# coding: utf-8

"""Create coarse-only QCE convergence diagnostics for QHD/Simbi ACOPF logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_qhd_convergence_diagnostics import (
    align_yaxis_values,
    load_case,
    parse_case_name_from_log,
    parse_float_after_colon,
    record_iteration_metrics,
    record_to_decision_vector,
    solve_standard_acopf,
)


DEFAULT_LEGEND_FONTSIZE = 12
DEFAULT_AXIS_FONTSIZE = 12


def parse_qce_coarse_log(log_path: Path, case) -> list[dict]:
    """Parse only main-loop coarse QCE records; skip final best summaries."""
    records: list[dict] = []
    current_iteration: int | None = None
    record: dict | None = None
    mode: str | None = None

    def finish_record() -> None:
        nonlocal record
        note = None if record is None else record.get("note")
        if isinstance(note, str) and (
            note.startswith("coarse_objective_best_from_top_")
            or note == "coarse_solution_before_refine"
        ):
            records.append(record)
        record = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Iteration:"):
            current_iteration = int(line.split(":", 1)[1].strip())
            mode = None
            continue

        if line.startswith("objective_value:"):
            finish_record()
            record = {
                "iteration": current_iteration,
                "objective": parse_float_after_colon(line),
                "buses": {},
                "branches": {},
            }
            mode = None
            continue

        if record is None:
            continue

        if line.startswith("lalm_energy:"):
            record["lalm_energy"] = parse_float_after_colon(line)
        elif line.startswith("feasible:"):
            record["feasible"] = line.split(":", 1)[1].strip()
        elif line.startswith("note:"):
            record["note"] = line.split(":", 1)[1].strip()
        elif line.startswith("max_abs_h:"):
            record["log_max_abs_h"] = parse_float_after_colon(line)
        elif line.startswith("l2_norm_h:"):
            record["log_l2_norm_h"] = parse_float_after_colon(line)
        elif line == "Bus Results":
            mode = "bus_header"
        elif line == "Branch Results":
            mode = "branch_header"
        elif line == "Summary":
            mode = "summary"
        elif mode == "bus_header":
            mode = "bus"
        elif mode == "branch_header":
            mode = "branch"
        elif mode == "bus":
            parts = re.split(r"\s+", line)
            if len(parts) >= 8 and parts[0].isdigit():
                bus_id = int(parts[0])
                record["buses"][bus_id] = {
                    "V_R": float(parts[1]),
                    "V_I": float(parts[2]),
                    "Vmag": float(parts[3]),
                    "Pg": float(parts[4]),
                    "Qg": float(parts[5]),
                    "Pl": float(parts[6]),
                    "Ql": float(parts[7]),
                }
        elif mode == "branch":
            parts = re.split(r"\s+", line)
            if len(parts) >= 11 and parts[0].isdigit():
                line_id = int(parts[0])
                record["branches"][line_id] = {
                    "From": int(parts[1]),
                    "To": int(parts[2]),
                    "Pik": float(parts[3]),
                    "Pki": float(parts[4]),
                    "Qik": float(parts[5]),
                    "Qki": float(parts[6]),
                    "Sik_sq": float(parts[7]),
                    "Ski_sq": float(parts[8]),
                    "LossP": float(parts[9]),
                    "LossQ": float(parts[10]),
                }
        elif mode == "summary" and line.startswith("Total Load Supplied:"):
            record["log_active_load_supplied_percent"] = parse_float_after_colon(line)

    finish_record()
    for item in records:
        item["x"] = record_to_decision_vector(item, case)
        item["answer_type"] = "coarse"
    return records


def _series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(history, key=lambda item: item["iteration"])
    return (
        np.asarray([item["iteration"] for item in ordered], dtype=int),
        np.asarray([item[key] for item in ordered], dtype=float),
    )


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 1e-14)


def add_bound_shrink_history(
    history: list[dict],
    records: list[dict],
    log_path: Path,
    shrink_factor: float,
    min_factor: float,
    start_iter: int,
    improve_tol: float,
) -> None:
    """Attach cumulative bound factors, preferring explicit step-log values."""
    explicit_factors: dict[int, float] = {}
    pattern = re.compile(
        r"bounds_shrink iter=(\d+).*?cumulative_factor=([-+0-9.eE]+)"
    )
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        explicit_factors[int(match.group(1))] = float(match.group(2))

    record_by_iteration = {int(item["iteration"]): item for item in records}
    shrink_count = 0
    previous_norm: float | None = None
    inferred_factors: dict[int, float] = {}
    for iteration in sorted(record_by_iteration):
        norm_h = float(record_by_iteration[iteration].get("log_l2_norm_h", np.nan))
        improved = (
            previous_norm is not None
            and np.isfinite(norm_h)
            and norm_h <= previous_norm * (1.0 - improve_tol)
        )
        if iteration >= start_iter and improved:
            shrink_count += 1
        inferred_factors[iteration] = max(min_factor, shrink_factor**shrink_count)
        previous_norm = norm_h

    for item in history:
        iteration = int(item["iteration"])
        item["cumulative_bound_factor"] = explicit_factors.get(
            iteration,
            inferred_factors[iteration],
        )


def write_qce_history_csv(history: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def detect_bound_config(
    log_path: Path,
    shrink_factor: float | None,
    min_factor: float | None,
    start_iter: int | None,
) -> tuple[float, float, int]:
    """Read bound settings from a step-log run header when available."""
    text = log_path.read_text(encoding="utf-8", errors="replace")

    def logged_float(name: str) -> float | None:
        match = re.search(rf"\b{name}=([-+0-9.eE]+)", text)
        return None if match is None else float(match.group(1))

    def logged_int(name: str) -> int | None:
        value = logged_float(name)
        return None if value is None else int(value)

    return (
        shrink_factor
        if shrink_factor is not None
        else (logged_float("bound_shrink_factor") or 0.9),
        min_factor
        if min_factor is not None
        else (logged_float("bound_min_factor") or 0.01),
        start_iter
        if start_iter is not None
        else (
            logged_int("bound_start_iter")
            if logged_int("bound_start_iter") is not None
            else 3
        ),
    )


def plot_qce_coarse_only(
    history: list[dict],
    case_name: str,
    output_dir: Path,
    legend_fontsize: float,
    axis_fontsize: float,
    bound_min_factor: float,
) -> tuple[Path, Path]:
    """Draw the QCE-specific 2x2 layout with no refine-only content."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    x, y = _series(history, "objective_difference")
    ax.plot(x, y, color="tab:blue", lw=1.6, label="Coarse objective difference")
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="Reference objective")
    ax2 = ax.twinx()
    x, supplied = _series(history, "active_load_supplied_percent")
    ax2.plot(
        x,
        supplied,
        color="tab:cyan",
        lw=1.4,
        label="Coarse active load supplied",
    )
    ax2.axhline(
        100.0,
        color="tab:gray",
        lw=1.0,
        ls=":",
        label="Reference load = 100%",
    )
    align_yaxis_values(ax, 0.0, ax2, 100.0)
    ax.set_title("Objective difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coarse objective - reference objective")
    ax2.set_ylabel("Active load supplied (%)")
    ax.grid(True, ls=":", alpha=0.55)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=legend_fontsize)

    ax = axes[0, 1]
    distance_specs = [
        ("bus_state_l2_distance", "Bus state", "tab:orange", "-"),
        ("branch_flow_l2_distance", "Branch flow", "tab:green", "--"),
        ("combined_l2_distance", "Combined", "tab:red", "-."),
    ]
    for key, label, color, linestyle in distance_specs:
        x, y = _series(history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    ax.set_title("Distance to reference solution")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 distance")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1, 0]
    for key, label, color, linestyle in [
        ("max_constraint_residual", "Coarse max |h|", "tab:purple", "-"),
        ("l2_constraint_residual", "Coarse ||h||2", "tab:brown", "--"),
    ]:
        x, y = _series(history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    ax.set_title("Constraint residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[1, 1]
    x, y = _series(history, "cumulative_bound_factor")
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
    ax.legend(fontsize=legend_fontsize)

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=axis_fontsize)
        ax.xaxis.label.set_size(axis_fontsize)
        ax.yaxis.label.set_size(axis_fontsize)
    ax2.tick_params(axis="both", labelsize=axis_fontsize)
    ax2.yaxis.label.set_size(axis_fontsize)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = case_name.replace(" ", "_").replace("/", "_")
    png_path = output_dir / f"{stem}_QCE_coarse_only_diagnostics.png"
    pdf_path = output_dir / f"{stem}_QCE_coarse_only_diagnostics.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_qce_coarse_only_1x4(
    history: list[dict],
    case_name: str,
    output_dir: Path,
    legend_fontsize: float,
    axis_fontsize: float,
    bound_min_factor: float,
) -> tuple[Path, Path]:
    """Draw the same coarse-only diagnostics as a single row of four panels."""
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5), constrained_layout=True)

    ax = axes[0]
    x, y = _series(history, "objective_difference")
    ax.plot(x, y, color="tab:blue", lw=1.6, label="Coarse objective difference")
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="Reference objective")
    ax2 = ax.twinx()
    x, supplied = _series(history, "active_load_supplied_percent")
    ax2.plot(
        x,
        supplied,
        color="tab:cyan",
        lw=1.4,
        label="Coarse active load supplied",
    )
    ax2.axhline(
        100.0,
        color="tab:gray",
        lw=1.0,
        ls=":",
        label="Reference load = 100%",
    )
    align_yaxis_values(ax, 0.0, ax2, 100.0)
    ax.set_title("Objective difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coarse objective - reference objective")
    ax2.set_ylabel("Active load supplied (%)")
    ax.grid(True, ls=":", alpha=0.55)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=legend_fontsize)

    ax = axes[1]
    for key, label, color, linestyle in [
        ("bus_state_l2_distance", "Bus state", "tab:orange", "-"),
        ("branch_flow_l2_distance", "Branch flow", "tab:green", "--"),
        ("combined_l2_distance", "Combined", "tab:red", "-."),
    ]:
        x, y = _series(history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    ax.set_title("Distance to reference solution")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 distance")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[2]
    for key, label, color, linestyle in [
        ("max_constraint_residual", "Coarse max |h|", "tab:purple", "-"),
        ("l2_constraint_residual", "Coarse ||h||2", "tab:brown", "--"),
    ]:
        x, y = _series(history, key)
        ax.semilogy(x, _positive(y), color=color, ls=linestyle, lw=1.5, label=label)
    ax.set_title("Constraint residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(fontsize=legend_fontsize)

    ax = axes[3]
    x, y = _series(history, "cumulative_bound_factor")
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
    ax.legend(fontsize=legend_fontsize)

    for panel in axes:
        panel.tick_params(axis="both", labelsize=axis_fontsize)
        panel.xaxis.label.set_size(axis_fontsize)
        panel.yaxis.label.set_size(axis_fontsize)
    ax2.tick_params(axis="both", labelsize=axis_fontsize)
    ax2.yaxis.label.set_size(axis_fontsize)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = case_name.replace(" ", "_").replace("/", "_")
    png_path = output_dir / f"{stem}_QCE_coarse_only_diagnostics_1x4.png"
    pdf_path = output_dir / f"{stem}_QCE_coarse_only_diagnostics_1x4.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create coarse-only QCE convergence diagnostics.")
    parser.add_argument("--log", type=Path, required=True, help="QCE coarse-only log file")
    parser.add_argument("--output-dir", type=Path, default=None, help="CSV/PNG/PDF output directory")
    parser.add_argument("--legend-fontsize", type=float, default=DEFAULT_LEGEND_FONTSIZE)
    parser.add_argument("--axis-fontsize", type=float, default=DEFAULT_AXIS_FONTSIZE)
    parser.add_argument("--bound-shrink-factor", type=float, default=None)
    parser.add_argument("--bound-min-factor", type=float, default=None)
    parser.add_argument("--bound-start-iter", type=int, default=None)
    parser.add_argument("--improve-tol", type=float, default=0.005)
    args = parser.parse_args()

    log_path = args.log.resolve()
    base_dir = Path(__file__).resolve().parent
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else log_path.parent / f"{log_path.stem}_QCE_analysis"
    )

    n_bus, case_name = parse_case_name_from_log(log_path)
    case = load_case(n_bus, base_dir)
    records = parse_qce_coarse_log(log_path, case)
    if not records:
        raise RuntimeError(f"No coarse-only QCE iteration records parsed from {log_path}")

    warm_records = sorted(records, key=lambda item: item.get("log_l2_norm_h", float("inf")))
    warm_starts = [item["x"] for item in warm_records[:8]]
    warm_starts.append(warm_records[-1]["x"])
    standard_solution = solve_standard_acopf(case, warm_starts=warm_starts)
    bound_shrink_factor, bound_min_factor, bound_start_iter = detect_bound_config(
        log_path,
        args.bound_shrink_factor,
        args.bound_min_factor,
        args.bound_start_iter,
    )

    history = [record_iteration_metrics(item, standard_solution, case) for item in records]
    history.sort(key=lambda item: item["iteration"])
    add_bound_shrink_history(
        history,
        records,
        log_path,
        shrink_factor=bound_shrink_factor,
        min_factor=bound_min_factor,
        start_iter=bound_start_iter,
        improve_tol=args.improve_tol,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{log_path.stem}_QCE_coarse_only_metrics.csv"
    write_qce_history_csv(history, csv_path)
    png_path, pdf_path = plot_qce_coarse_only(
        history,
        case_name,
        output_dir,
        legend_fontsize=args.legend_fontsize,
        axis_fontsize=args.axis_fontsize,
        bound_min_factor=bound_min_factor,
    )
    row_png_path, row_pdf_path = plot_qce_coarse_only_1x4(
        history,
        case_name,
        output_dir,
        legend_fontsize=args.legend_fontsize,
        axis_fontsize=args.axis_fontsize,
        bound_min_factor=bound_min_factor,
    )

    best_distance = min(history, key=lambda item: item["combined_l2_distance"])
    best_residual = min(history, key=lambda item: item["l2_constraint_residual"])
    print(f"Parsed coarse-only records: {len(records)}")
    print(
        "Reference solve: "
        f"success={standard_solution['success']}, "
        f"objective={standard_solution['objective']:.12g}, "
        f"max|h|={standard_solution['max_abs_h']:.3e}"
    )
    print(
        f"Best combined distance: iter={best_distance['iteration']}, "
        f"distance={best_distance['combined_l2_distance']:.6g}"
    )
    print(
        f"Best residual: iter={best_residual['iteration']}, "
        f"||h||2={best_residual['l2_constraint_residual']:.6g}"
    )
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"1x4 PNG: {row_png_path}")
    print(f"1x4 PDF: {row_pdf_path}")


if __name__ == "__main__":
    main()
