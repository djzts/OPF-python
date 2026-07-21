#!/usr/bin/env python
# coding: utf-8

"""
Build focused comparison tables for the rerun TNC trajectory and QHD-SB
coarse-only trajectory over plot_iteration = 0..230.

Input:
- output/3bus_tnc_vs_qhdsb_convergence.csv

Outputs:
- output/3bus_tnc_qhdsb_230_start_metrics.csv
- output/3bus_tnc_qhdsb_230_objective_comparison.csv
- output/3bus_tnc_qhdsb_230_iteration_metrics.csv
- output/3bus_tnc_qhdsb_230_key_iteration_metrics.csv
- output/3bus_tnc_qhdsb_230_summary.md
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SOURCE_CSV = OUT_DIR / "3bus_tnc_vs_qhdsb_convergence.csv"
REFERENCE_OBJECTIVE_CSV = OUT_DIR / "3bus_objective_comparison.csv"

MAX_PLOT_ITERATION = 230
REFERENCE_METHOD = "SQP (SLSQP)"
TNC_LABEL = "Truncated Newton"
QHDSB_LABEL = "QHD-SB coarse-only"
COARSE_RESIDUAL_TOL = 1e-3
STRICT_MAX_TOL = 1e-5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def fmt_float(value: object, digits: int = 6) -> str:
    if value == "" or value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(fmt_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reference_objective() -> float:
    for row in read_csv(REFERENCE_OBJECTIVE_CSV):
        if row.get("method") == REFERENCE_METHOD:
            return float(row["objective"])
    raise RuntimeError(f"Could not find {REFERENCE_METHOD} objective in {REFERENCE_OBJECTIVE_CSV}")


def objective_gap_pct(objective: float, reference_obj: float) -> float:
    return 100.0 * (objective / reference_obj - 1.0)


def normalize_rows(rows: list[dict[str, str]], reference_obj: float) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        plot_iteration = int(row["plot_iteration"])
        if plot_iteration > MAX_PLOT_ITERATION:
            continue
        objective = float(row["objective"])
        l2_h = float(row["l2_h"])
        max_abs_h = float(row["max_abs_h"])
        normalized.append(
            {
                "label": row["label"],
                "method": row["method"],
                "source": row["source"],
                "start_name": row["start_name"],
                "plot_iteration": plot_iteration,
                "solver_iteration": row["solver_iteration"],
                "stage": row["stage"],
                "objective": objective,
                "objective_gap_pct_vs_sqp": objective_gap_pct(objective, reference_obj),
                "l2_h": l2_h,
                "max_abs_h": max_abs_h,
                "load_supplied_pct": float(row["load_supplied_pct"]),
                "meets_l2_1e_3": l2_h <= COARSE_RESIDUAL_TOL,
                "meets_max_abs_1e_3": max_abs_h <= COARSE_RESIDUAL_TOL,
                "meets_max_abs_1e_5": max_abs_h <= STRICT_MAX_TOL,
            }
        )
    return normalized


def by_label(rows: list[dict[str, object]], label: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row["label"] == label]
    selected.sort(key=lambda row: int(row["plot_iteration"]))
    if not selected:
        raise RuntimeError(f"No rows found for {label}")
    return selected


def select_row(rows: list[dict[str, object]], selector: str) -> dict[str, object] | None:
    if selector == "initial_common_center":
        return next(row for row in rows if int(row["plot_iteration"]) == 0)
    if selector == "final_plot_iteration_230":
        return next(row for row in rows if int(row["plot_iteration"]) == MAX_PLOT_ITERATION)
    if selector == "best_l2_residual":
        return min(rows, key=lambda row: as_float(row, "l2_h"))
    if selector == "first_l2_le_1e-3":
        eligible = [row for row in rows if as_float(row, "l2_h") <= COARSE_RESIDUAL_TOL]
        return eligible[0] if eligible else None
    if selector == "min_objective_with_l2_le_1e-3":
        eligible = [row for row in rows if as_float(row, "l2_h") <= COARSE_RESIDUAL_TOL]
        return min(eligible, key=lambda row: as_float(row, "objective")) if eligible else None
    if selector == "min_objective_overall_unfiltered":
        return min(rows, key=lambda row: as_float(row, "objective"))
    raise ValueError(f"Unknown selector: {selector}")


def objective_row(selector: str, tnc: dict[str, object] | None, qhdsb: dict[str, object] | None) -> dict[str, object]:
    row: dict[str, object] = {"selection": selector}
    for prefix, selected in [("tnc", tnc), ("qhdsb", qhdsb)]:
        if selected is None:
            continue
        row.update(
            {
                f"{prefix}_plot_iteration": selected["plot_iteration"],
                f"{prefix}_solver_iteration": selected["solver_iteration"],
                f"{prefix}_objective": selected["objective"],
                f"{prefix}_objective_gap_pct_vs_sqp": selected["objective_gap_pct_vs_sqp"],
                f"{prefix}_l2_h": selected["l2_h"],
                f"{prefix}_max_abs_h": selected["max_abs_h"],
                f"{prefix}_load_supplied_pct": selected["load_supplied_pct"],
                f"{prefix}_meets_l2_1e_3": selected["meets_l2_1e_3"],
                f"{prefix}_meets_max_abs_1e_5": selected["meets_max_abs_1e_5"],
            }
        )
    if tnc is not None and qhdsb is not None:
        tnc_objective = as_float(tnc, "objective")
        qhdsb_objective = as_float(qhdsb, "objective")
        tnc_l2 = as_float(tnc, "l2_h")
        qhdsb_l2 = as_float(qhdsb, "l2_h")
        row["objective_delta_tnc_minus_qhdsb"] = tnc_objective - qhdsb_objective
        row["objective_delta_pct_of_qhdsb"] = (
            100.0 * (tnc_objective / qhdsb_objective - 1.0) if qhdsb_objective else ""
        )
        row["l2_delta_tnc_minus_qhdsb"] = tnc_l2 - qhdsb_l2
        row["l2_ratio_tnc_over_qhdsb"] = tnc_l2 / qhdsb_l2 if qhdsb_l2 else ""
        row["lower_l2_method"] = TNC_LABEL if tnc_l2 < qhdsb_l2 else QHDSB_LABEL if qhdsb_l2 < tnc_l2 else "tie"
        row["lower_objective_method"] = (
            TNC_LABEL if tnc_objective < qhdsb_objective else QHDSB_LABEL if qhdsb_objective < tnc_objective else "tie"
        )
    return row


def pair_iteration_rows(tnc_rows: list[dict[str, object]], qhdsb_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tnc_by_iter = {int(row["plot_iteration"]): row for row in tnc_rows}
    qhd_by_iter = {int(row["plot_iteration"]): row for row in qhdsb_rows}
    paired: list[dict[str, object]] = []
    for plot_iteration in range(MAX_PLOT_ITERATION + 1):
        tnc = tnc_by_iter[plot_iteration]
        qhd = qhd_by_iter[plot_iteration]
        tnc_l2 = as_float(tnc, "l2_h")
        qhd_l2 = as_float(qhd, "l2_h")
        tnc_obj = as_float(tnc, "objective")
        qhd_obj = as_float(qhd, "objective")
        tnc_max = as_float(tnc, "max_abs_h")
        qhd_max = as_float(qhd, "max_abs_h")
        paired.append(
            {
                "plot_iteration": plot_iteration,
                "tnc_solver_iteration": tnc["solver_iteration"],
                "tnc_stage": tnc["stage"],
                "tnc_objective": tnc_obj,
                "tnc_objective_gap_pct_vs_sqp": tnc["objective_gap_pct_vs_sqp"],
                "tnc_l2_h": tnc_l2,
                "tnc_max_abs_h": tnc_max,
                "tnc_load_supplied_pct": tnc["load_supplied_pct"],
                "qhdsb_solver_iteration": qhd["solver_iteration"],
                "qhdsb_stage": qhd["stage"],
                "qhdsb_objective": qhd_obj,
                "qhdsb_objective_gap_pct_vs_sqp": qhd["objective_gap_pct_vs_sqp"],
                "qhdsb_l2_h": qhd_l2,
                "qhdsb_max_abs_h": qhd_max,
                "qhdsb_load_supplied_pct": qhd["load_supplied_pct"],
                "objective_delta_tnc_minus_qhdsb": tnc_obj - qhd_obj,
                "objective_gap_delta_pct_tnc_minus_qhdsb": tnc["objective_gap_pct_vs_sqp"]
                - qhd["objective_gap_pct_vs_sqp"],
                "l2_delta_tnc_minus_qhdsb": tnc_l2 - qhd_l2,
                "l2_ratio_tnc_over_qhdsb": tnc_l2 / qhd_l2 if qhd_l2 else "",
                "max_abs_delta_tnc_minus_qhdsb": tnc_max - qhd_max,
                "max_abs_ratio_tnc_over_qhdsb": tnc_max / qhd_max if qhd_max else "",
                "lower_l2_method": TNC_LABEL if tnc_l2 < qhd_l2 else QHDSB_LABEL if qhd_l2 < tnc_l2 else "tie",
                "lower_objective_method": TNC_LABEL if tnc_obj < qhd_obj else QHDSB_LABEL if qhd_obj < tnc_obj else "tie",
                "both_l2_le_1e_3": tnc_l2 <= COARSE_RESIDUAL_TOL and qhd_l2 <= COARSE_RESIDUAL_TOL,
                "tnc_l2_le_1e_3": tnc_l2 <= COARSE_RESIDUAL_TOL,
                "qhdsb_l2_le_1e_3": qhd_l2 <= COARSE_RESIDUAL_TOL,
                "both_max_abs_le_1e_5": tnc_max <= STRICT_MAX_TOL and qhd_max <= STRICT_MAX_TOL,
            }
        )
    return paired


def first_interval(rows: list[dict[str, object]], method_name: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    for row in rows:
        is_method = row["lower_l2_method"] == method_name
        plot_iteration = int(row["plot_iteration"])
        if is_method and start is None:
            start = plot_iteration
            last = plot_iteration
        elif is_method:
            last = plot_iteration
        elif start is not None and last is not None:
            intervals.append((start, last))
            start = None
            last = None
    if start is not None and last is not None:
        intervals.append((start, last))
    return intervals


def build_start_metrics(tnc_rows: list[dict[str, object]], qhdsb_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tnc_start = tnc_rows[0]
    qhd_start = qhdsb_rows[0]
    return [
        {
            "method": TNC_LABEL,
            "plot_iteration": tnc_start["plot_iteration"],
            "solver_iteration": tnc_start["solver_iteration"],
            "objective": tnc_start["objective"],
            "objective_gap_pct_vs_sqp": tnc_start["objective_gap_pct_vs_sqp"],
            "l2_h": tnc_start["l2_h"],
            "max_abs_h": tnc_start["max_abs_h"],
            "load_supplied_pct": tnc_start["load_supplied_pct"],
            "same_as_other_start": all(
                abs(as_float(tnc_start, key) - as_float(qhd_start, key)) <= 1e-14
                for key in ["objective", "l2_h", "max_abs_h", "load_supplied_pct"]
            ),
            "source": tnc_start["source"],
        },
        {
            "method": QHDSB_LABEL,
            "plot_iteration": qhd_start["plot_iteration"],
            "solver_iteration": qhd_start["solver_iteration"],
            "objective": qhd_start["objective"],
            "objective_gap_pct_vs_sqp": qhd_start["objective_gap_pct_vs_sqp"],
            "l2_h": qhd_start["l2_h"],
            "max_abs_h": qhd_start["max_abs_h"],
            "load_supplied_pct": qhd_start["load_supplied_pct"],
            "same_as_other_start": all(
                abs(as_float(tnc_start, key) - as_float(qhd_start, key)) <= 1e-14
                for key in ["objective", "l2_h", "max_abs_h", "load_supplied_pct"]
            ),
            "source": qhd_start["source"],
        },
    ]


def build_summary_markdown(
    reference_obj: float,
    start_rows: list[dict[str, object]],
    objective_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    key_rows: list[dict[str, object]],
) -> str:
    tnc_better = sum(1 for row in paired_rows if row["lower_l2_method"] == TNC_LABEL)
    qhd_better = sum(1 for row in paired_rows if row["lower_l2_method"] == QHDSB_LABEL)
    ties = len(paired_rows) - tnc_better - qhd_better
    tnc_intervals = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in first_interval(paired_rows, TNC_LABEL))
    qhd_intervals = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in first_interval(paired_rows, QHDSB_LABEL))

    final_row = next(row for row in objective_rows if row["selection"] == "final_plot_iteration_230")

    start_table = markdown_table(
        start_rows,
        [
            ("method", "Method"),
            ("objective", "Objective"),
            ("objective_gap_pct_vs_sqp", "Obj gap vs SQP %"),
            ("l2_h", "L2 h"),
            ("max_abs_h", "Max abs h"),
            ("load_supplied_pct", "Load %"),
            ("same_as_other_start", "Same start"),
        ],
    )
    objective_table = markdown_table(
        objective_rows,
        [
            ("selection", "Selection"),
            ("tnc_plot_iteration", "TNC plot iter"),
            ("tnc_objective", "TNC obj"),
            ("tnc_objective_gap_pct_vs_sqp", "TNC obj gap %"),
            ("tnc_l2_h", "TNC L2 h"),
            ("tnc_max_abs_h", "TNC max abs h"),
            ("qhdsb_plot_iteration", "QHD-SB plot iter"),
            ("qhdsb_objective", "QHD-SB obj"),
            ("qhdsb_objective_gap_pct_vs_sqp", "QHD-SB obj gap %"),
            ("qhdsb_l2_h", "QHD-SB L2 h"),
            ("qhdsb_max_abs_h", "QHD-SB max abs h"),
            ("lower_objective_method", "Lower obj"),
            ("lower_l2_method", "Lower L2"),
        ],
    )
    key_table = markdown_table(
        key_rows,
        [
            ("plot_iteration", "Plot iter"),
            ("tnc_objective", "TNC obj"),
            ("tnc_l2_h", "TNC L2 h"),
            ("qhdsb_objective", "QHD-SB obj"),
            ("qhdsb_l2_h", "QHD-SB L2 h"),
            ("l2_ratio_tnc_over_qhdsb", "TNC/QHD L2"),
            ("lower_l2_method", "Lower L2"),
        ],
    )

    return "\n".join(
        [
            "# TNC vs QHD-SB 3-bus comparison over plot_iteration 0..230",
            "",
            "## Scope",
            "",
            f"- Source data: `{SOURCE_CSV.name}`.",
            f"- Window: `plot_iteration = 0..{MAX_PLOT_ITERATION}`. The first row is the shared QHD-SB initial center for both methods.",
            "- TNC is the rerun from that same initial center. QHD-SB is the coarse-only QCE beam trajectory.",
            f"- Objective gaps use the SQP reference objective `{fmt_float(reference_obj, digits=12)}` from `{REFERENCE_OBJECTIVE_CSV.name}`.",
            "- Objective values from infeasible points are reported, but should not be interpreted as final OPF objective quality without the residual columns.",
            "",
            "## Start metrics",
            "",
            start_table,
            "",
            "## Objective comparison",
            "",
            objective_table,
            "",
            "## Iteration-level comparison highlights",
            "",
            f"- Same-iteration residual winner counts: TNC lower L2 in {tnc_better} rows, QHD-SB lower L2 in {qhd_better} rows, tie in {ties} row.",
            f"- TNC lower-L2 intervals: {tnc_intervals}.",
            f"- QHD-SB lower-L2 interval: {qhd_intervals}.",
            f"- TNC first reaches L2 <= 1e-3 at plot iteration {next(row['tnc_plot_iteration'] for row in objective_rows if row['selection'] == 'first_l2_le_1e-3')}; QHD-SB first reaches it at plot iteration {next(row['qhdsb_plot_iteration'] for row in objective_rows if row['selection'] == 'first_l2_le_1e-3')}.",
            f"- At plot iteration 230, TNC has L2 h = {fmt_float(final_row['tnc_l2_h'])} and objective = {fmt_float(final_row['tnc_objective'])}; QHD-SB has L2 h = {fmt_float(final_row['qhdsb_l2_h'])} and objective = {fmt_float(final_row['qhdsb_objective'])}.",
            "- Neither method reaches the strict max-abs residual tolerance 1e-5 within this 0..230 window, so objective comparisons should be read together with residual violation levels.",
            "",
            "## Key iteration metrics",
            "",
            key_table,
            "",
            "## Output tables",
            "",
            "- `3bus_tnc_qhdsb_230_start_metrics.csv`: two-row start comparison confirming the common initial point.",
            "- `3bus_tnc_qhdsb_230_objective_comparison.csv`: objective-focused selections, with residuals attached for feasibility context.",
            "- `3bus_tnc_qhdsb_230_iteration_metrics.csv`: complete side-by-side data for plot iterations 0..230.",
            "- `3bus_tnc_qhdsb_230_key_iteration_metrics.csv`: compact checkpoint table for manuscript or slides.",
        ]
    ) + "\n"


def main() -> None:
    reference_obj = reference_objective()
    rows = normalize_rows(read_csv(SOURCE_CSV), reference_obj)
    tnc_rows = by_label(rows, TNC_LABEL)
    qhdsb_rows = by_label(rows, QHDSB_LABEL)
    if len(tnc_rows) != MAX_PLOT_ITERATION + 1 or len(qhdsb_rows) != MAX_PLOT_ITERATION + 1:
        raise RuntimeError(
            f"Expected {MAX_PLOT_ITERATION + 1} rows per method, got "
            f"TNC={len(tnc_rows)}, QHD-SB={len(qhdsb_rows)}"
        )

    start_rows = build_start_metrics(tnc_rows, qhdsb_rows)
    selections = [
        "initial_common_center",
        "first_l2_le_1e-3",
        "best_l2_residual",
        "final_plot_iteration_230",
        "min_objective_with_l2_le_1e-3",
        "min_objective_overall_unfiltered",
    ]
    objective_rows = [
        objective_row(selection, select_row(tnc_rows, selection), select_row(qhdsb_rows, selection))
        for selection in selections
    ]
    paired_rows = pair_iteration_rows(tnc_rows, qhdsb_rows)
    key_iterations = [0, 1, 5, 10, 25, 50, 100, 150, 195, 197, 200, 220, 228, 229, 230]
    key_rows = [row for row in paired_rows if int(row["plot_iteration"]) in key_iterations]

    start_fields = [
        "method",
        "plot_iteration",
        "solver_iteration",
        "objective",
        "objective_gap_pct_vs_sqp",
        "l2_h",
        "max_abs_h",
        "load_supplied_pct",
        "same_as_other_start",
        "source",
    ]
    objective_fields = [
        "selection",
        "tnc_plot_iteration",
        "tnc_solver_iteration",
        "tnc_objective",
        "tnc_objective_gap_pct_vs_sqp",
        "tnc_l2_h",
        "tnc_max_abs_h",
        "tnc_load_supplied_pct",
        "tnc_meets_l2_1e_3",
        "tnc_meets_max_abs_1e_5",
        "qhdsb_plot_iteration",
        "qhdsb_solver_iteration",
        "qhdsb_objective",
        "qhdsb_objective_gap_pct_vs_sqp",
        "qhdsb_l2_h",
        "qhdsb_max_abs_h",
        "qhdsb_load_supplied_pct",
        "qhdsb_meets_l2_1e_3",
        "qhdsb_meets_max_abs_1e_5",
        "objective_delta_tnc_minus_qhdsb",
        "objective_delta_pct_of_qhdsb",
        "l2_delta_tnc_minus_qhdsb",
        "l2_ratio_tnc_over_qhdsb",
        "lower_l2_method",
        "lower_objective_method",
    ]
    iteration_fields = [
        "plot_iteration",
        "tnc_solver_iteration",
        "tnc_stage",
        "tnc_objective",
        "tnc_objective_gap_pct_vs_sqp",
        "tnc_l2_h",
        "tnc_max_abs_h",
        "tnc_load_supplied_pct",
        "qhdsb_solver_iteration",
        "qhdsb_stage",
        "qhdsb_objective",
        "qhdsb_objective_gap_pct_vs_sqp",
        "qhdsb_l2_h",
        "qhdsb_max_abs_h",
        "qhdsb_load_supplied_pct",
        "objective_delta_tnc_minus_qhdsb",
        "objective_gap_delta_pct_tnc_minus_qhdsb",
        "l2_delta_tnc_minus_qhdsb",
        "l2_ratio_tnc_over_qhdsb",
        "max_abs_delta_tnc_minus_qhdsb",
        "max_abs_ratio_tnc_over_qhdsb",
        "lower_l2_method",
        "lower_objective_method",
        "both_l2_le_1e_3",
        "tnc_l2_le_1e_3",
        "qhdsb_l2_le_1e_3",
        "both_max_abs_le_1e_5",
    ]

    write_csv(OUT_DIR / "3bus_tnc_qhdsb_230_start_metrics.csv", start_rows, start_fields)
    write_csv(OUT_DIR / "3bus_tnc_qhdsb_230_objective_comparison.csv", objective_rows, objective_fields)
    write_csv(OUT_DIR / "3bus_tnc_qhdsb_230_iteration_metrics.csv", paired_rows, iteration_fields)
    write_csv(OUT_DIR / "3bus_tnc_qhdsb_230_key_iteration_metrics.csv", key_rows, iteration_fields)

    summary = build_summary_markdown(reference_obj, start_rows, objective_rows, paired_rows, key_rows)
    (OUT_DIR / "3bus_tnc_qhdsb_230_summary.md").write_text(summary, encoding="utf-8")

    print("Wrote focused TNC vs QHD-SB comparison tables to", OUT_DIR)
    print(f"Rows: TNC={len(tnc_rows)}, QHD-SB={len(qhdsb_rows)}, paired={len(paired_rows)}")


if __name__ == "__main__":
    main()
