#!/usr/bin/env python
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_qhd_convergence_diagnostics import load_case  # noqa: E402


BUS_LIST = [2, 3, 5, 9]
EXPERIMENTS = [
    ("before-multi-beam", ROOT / "logs" / "before-multi-beam"),
    ("coarse only", ROOT / "logs" / "coarse only"),
    ("multi_beam", ROOT / "logs" / "multi_beam"),
    ("single_beam", ROOT / "logs" / "single_beam"),
]

REFERENCE_OBJECTIVES = {
    2: (0.609639734673, "SLSQP reference from plot_qhd_convergence_diagnostics.py case model"),
    3: (0.531710655973, "logs/QCE_result/Buses-3_06-22-2026_05-08-43_vs_44_QCE_analysis.md"),
    5: (9.19504654127, "5bus-answer.txt / SLSQP cross-check"),
    9: (4.09813222, "9bus-answer.txt"),
}

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def safe_float(text: str | None) -> float:
    if text is None:
        return math.nan
    cleaned = text.strip().rstrip("%").replace(",", "")
    if not cleaned:
        return math.nan
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


def safe_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_case_from_name(path: Path) -> int | None:
    match = re.search(r"Buses-(\d+)_", path.name)
    return int(match.group(1)) if match else None


def parse_datetime_from_name(path: Path) -> datetime | None:
    match = re.search(
        r"Buses-\d+_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})",
        path.name,
    )
    if not match:
        return None
    month, day, year, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)


def parse_created_at(line: str) -> str:
    return line.split(":", 1)[1].strip()


def parse_run_start(line: str) -> dict[str, str]:
    if "] run_start " in line:
        line = line.split("] run_start ", 1)[1]
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"(\w+)=([^,]+)", line)}


def classify_note(note: str) -> str:
    lower = (note or "").strip().lower()
    if lower.startswith("best_iteration") or lower.startswith("best_"):
        return "best_snapshot"
    if "coarse_solution" in lower or (lower.startswith("coarse") and "refined" not in lower):
        return "coarse"
    if "refined" in lower or "tnc" in lower or "gurobi" in lower or "active_beam" in lower:
        return "refined"
    if lower:
        return "evaluation"
    return "unknown"


def record_to_decision_vector(record: dict[str, Any], case: Any) -> np.ndarray | None:
    buses = record.get("buses", {})
    branches = record.get("branches", {})
    if len(buses) < case.n_buses or len(branches) < case.n_lines:
        return None

    try:
        p_g = []
        q_g = []
        for gid in case.gen_ids:
            bus_id = int(case.gens[gid][0])
            p_g.append(buses[bus_id]["Pg"])
            q_g.append(buses[bus_id]["Qg"])

        v_r = np.array([buses[bid]["V_R"] for bid in case.bus_ids], dtype=float)
        v_i = np.array([buses[bid]["V_I"] for bid in case.bus_ids], dtype=float)
        v_sq = np.array([buses[bid]["Vmag"] ** 2 for bid in case.bus_ids], dtype=float)

        p_ij = []
        q_ij = []
        s_tot_sq = []
        for lid in case.line_ids:
            branch = branches[lid]
            p_ij.extend([branch["Pik"], branch["Pki"]])
            q_ij.extend([branch["Qik"], branch["Qki"]])
            s_tot_sq.extend([branch["Sik_sq"], branch["Ski_sq"]])

        return np.concatenate(
            [
                np.asarray(p_g, dtype=float),
                np.asarray(q_g, dtype=float),
                v_r,
                v_i,
                v_sq,
                np.asarray(p_ij, dtype=float),
                np.asarray(q_ij, dtype=float),
                np.asarray(s_tot_sq, dtype=float),
            ]
        )
    except KeyError:
        return None


def add_computed_metrics(record: dict[str, Any], case: Any, bus: int) -> dict[str, Any]:
    note = record.get("note", "")
    answer_type = classify_note(note)
    is_main = answer_type != "best_snapshot"

    x = record_to_decision_vector(record, case)
    max_h_recomputed = math.nan
    l2_h_recomputed = math.nan
    load_recomputed = math.nan
    objective_recomputed = math.nan
    if x is not None:
        try:
            h = case.h(x)
            max_h_recomputed = float(np.max(np.abs(h)))
            l2_h_recomputed = float(np.linalg.norm(h))
            load_recomputed = float(case.active_load_supplied_percent(x))
            objective_recomputed = float(case.objective(x))
        except Exception:
            pass

    objective = record.get("objective", math.nan)
    objective_used = objective_recomputed if math.isfinite(objective_recomputed) else objective
    max_h_used = max_h_recomputed if math.isfinite(max_h_recomputed) else record.get("max_abs_h_log", math.nan)
    l2_h_used = l2_h_recomputed if math.isfinite(l2_h_recomputed) else record.get("l2_norm_h_log", math.nan)
    load_used = load_recomputed if math.isfinite(load_recomputed) else record.get("load_supplied_pct_log", math.nan)

    ref_obj, _ref_source = REFERENCE_OBJECTIVES.get(bus, (math.nan, ""))
    objective_gap = objective_used - ref_obj if math.isfinite(ref_obj) and math.isfinite(objective_used) else math.nan
    objective_gap_pct = 100.0 * objective_gap / ref_obj if math.isfinite(ref_obj) and ref_obj != 0 else math.nan

    return {
        "answer_type": answer_type,
        "is_main_record": is_main,
        "objective_used": objective_used,
        "objective_gap": objective_gap,
        "objective_gap_pct": objective_gap_pct,
        "max_abs_h_recomputed": max_h_recomputed,
        "l2_norm_h_recomputed": l2_h_recomputed,
        "max_abs_h_used": max_h_used,
        "l2_norm_h_used": l2_h_used,
        "load_supplied_pct_used": load_used,
        "objective_recomputed": objective_recomputed,
        "has_complete_tables": x is not None,
    }


def parse_step_evaluations(
    path: Path,
    experiment: str,
    bus: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    pattern = re.compile(
        rf"\[step [^\]]+\] evaluation iter=(\d+),.*?objective=({NUM}).*?"
        rf"l2_norm_h=({NUM}).*?max_abs_h=({NUM})"
    )
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for record_index, line in enumerate(handle):
            match = pattern.search(line)
            if not match:
                continue
            objective = safe_float(match.group(2))
            l2_h = safe_float(match.group(3))
            max_h = safe_float(match.group(4))
            ref_obj, _ref_source = REFERENCE_OBJECTIVES.get(bus, (math.nan, ""))
            objective_gap = objective - ref_obj if math.isfinite(ref_obj) else math.nan
            objective_gap_pct = 100.0 * objective_gap / ref_obj if math.isfinite(ref_obj) and ref_obj else math.nan
            records.append(
                {
                    **metadata,
                    "experiment": experiment,
                    "bus": bus,
                    "log_name": path.name,
                    "source_log": str(path.relative_to(ROOT)),
                    "record_index": record_index,
                    "iteration": int(match.group(1)),
                    "answer_type": "evaluation",
                    "is_main_record": True,
                    "objective_log": objective,
                    "objective_used": objective,
                    "objective_recomputed": math.nan,
                    "objective_gap": objective_gap,
                    "objective_gap_pct": objective_gap_pct,
                    "lalm_energy": math.nan,
                    "feasible": "",
                    "note": "step_evaluation_fallback",
                    "max_abs_h_log": max_h,
                    "l2_norm_h_log": l2_h,
                    "max_abs_h_recomputed": math.nan,
                    "l2_norm_h_recomputed": math.nan,
                    "max_abs_h_used": max_h,
                    "l2_norm_h_used": l2_h,
                    "lambda_inf_norm": math.nan,
                    "lambda_l2_norm": math.nan,
                    "load_supplied_pct_log": math.nan,
                    "load_supplied_pct_used": math.nan,
                    "has_complete_tables": False,
                }
            )
    return records


def extract_array_after(line: str, key: str) -> list[float] | None:
    if key == "x":
        marker = ", x=["
        start = line.find(marker)
        if start < 0:
            marker = " x=["
            start = line.find(marker)
    else:
        marker = f"{key}=["
        start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find("]", start)
    if end < 0:
        return None
    return [float(match.group(0)) for match in re.finditer(NUM, line[start:end])]


def metrics_from_vector(
    x_values: list[float] | np.ndarray | None,
    case: Any,
    bus: int,
    objective_log: float,
) -> dict[str, Any]:
    x = None if x_values is None else np.asarray(x_values, dtype=float).reshape(-1)
    max_h = math.nan
    l2_h = math.nan
    load_supplied = math.nan
    objective_recomputed = math.nan
    if x is not None and x.size == case.n_variables:
        try:
            h = case.h(x)
            max_h = float(np.max(np.abs(h)))
            l2_h = float(np.linalg.norm(h))
            load_supplied = float(case.active_load_supplied_percent(x))
            objective_recomputed = float(case.objective(x))
        except Exception:
            pass

    objective_used = objective_recomputed if math.isfinite(objective_recomputed) else objective_log
    ref_obj, _ref_source = REFERENCE_OBJECTIVES.get(bus, (math.nan, ""))
    objective_gap = objective_used - ref_obj if math.isfinite(ref_obj) and math.isfinite(objective_used) else math.nan
    objective_gap_pct = 100.0 * objective_gap / ref_obj if math.isfinite(ref_obj) and ref_obj != 0 else math.nan
    return {
        "objective_recomputed": objective_recomputed,
        "objective_used": objective_used,
        "objective_gap": objective_gap,
        "objective_gap_pct": objective_gap_pct,
        "max_abs_h_used": max_h,
        "l2_norm_h_used": l2_h,
        "load_supplied_pct_used": load_supplied,
        "has_vector": x is not None and x.size == case.n_variables,
        "vector_length": int(x.size) if x is not None else 0,
    }


def make_round_record(
    metadata: dict[str, Any],
    experiment: str,
    path: Path,
    bus: int,
    case: Any,
    iteration: int,
    stage: str,
    objective_log: float,
    x_values: list[float] | np.ndarray | None,
    *,
    record_index: int,
    rank: int | None = None,
    refined_applied: bool | None = None,
    lalm_energy: float = math.nan,
    note: str = "",
) -> dict[str, Any]:
    metrics = metrics_from_vector(x_values, case, bus, objective_log)
    return {
        **metadata,
        "experiment": experiment,
        "bus": bus,
        "log_name": path.name,
        "source_log": str(path.relative_to(ROOT)),
        "record_index": record_index,
        "iteration": iteration,
        "stage": stage,
        "is_main_record": True,
        "rank": rank if rank is not None else "",
        "refined_applied": "" if refined_applied is None else bool(refined_applied),
        "objective_log": objective_log,
        "lalm_energy": lalm_energy,
        "note": note,
        **metrics,
    }


def parse_step_round_records(path: Path, experiment: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse per-round coarse and post-refine/final vectors from step logs."""
    bus = parse_case_from_name(path)
    if bus not in BUS_LIST:
        return []
    case = load_case(bus, ROOT)
    records: list[dict[str, Any]] = []
    record_index = 0

    iter_pattern = re.compile(r"global_candidate iter=(\d+)")
    rank_pattern = re.compile(r"rank=(\d+)")
    energy_pattern = re.compile(rf"energy=({NUM})")
    coarse_obj_pattern = re.compile(rf"coarse_objective=({NUM})")
    obj_pattern = re.compile(rf"(?<!coarse_)objective=({NUM})")
    refined_pattern = re.compile(r"refined=(True|False)")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "global_candidate" not in line or "evaluation_choice=True" not in line:
                continue
            iter_match = iter_pattern.search(line)
            if not iter_match:
                continue
            coarse_x = extract_array_after(line, "coarse_x")
            final_x = extract_array_after(line, "x")
            coarse_obj = safe_float(coarse_obj_pattern.search(line).group(1) if coarse_obj_pattern.search(line) else None)
            final_obj = safe_float(obj_pattern.search(line).group(1) if obj_pattern.search(line) else None)
            energy = safe_float(energy_pattern.search(line).group(1) if energy_pattern.search(line) else None)
            rank = safe_int(rank_pattern.search(line).group(1) if rank_pattern.search(line) else None)
            refined_match = refined_pattern.search(line)
            refined_applied = refined_match.group(1) == "True" if refined_match else None
            iteration = int(iter_match.group(1))

            if coarse_x is not None:
                records.append(
                    make_round_record(
                        metadata,
                        experiment,
                        path,
                        bus,
                        case,
                        iteration,
                        "coarse_pre_refine",
                        coarse_obj,
                        coarse_x,
                        record_index=record_index,
                        rank=rank,
                        refined_applied=False,
                        lalm_energy=energy,
                        note="coarse vector before optional refinement",
                    )
                )
                record_index += 1
            if final_x is not None:
                records.append(
                    make_round_record(
                        metadata,
                        experiment,
                        path,
                        bus,
                        case,
                        iteration,
                        "post_refine_or_final",
                        final_obj,
                        final_x,
                        record_index=record_index,
                        rank=rank,
                        refined_applied=refined_applied,
                        lalm_energy=energy,
                        note="final vector after coarse plus refinement when available",
                    )
                )
                record_index += 1

    return records


def round_records_from_table_records(table_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fallback for older logs that only have printed solution tables."""
    round_records = []
    for idx, row in enumerate(table_records):
        answer_type = str(row.get("answer_type", ""))
        if answer_type == "coarse":
            stage = "coarse_pre_refine"
            refined = False
        elif answer_type in {"refined", "evaluation"}:
            stage = "post_refine_or_final"
            refined = answer_type == "refined"
        else:
            continue
        copied = dict(row)
        copied["stage"] = stage
        copied["rank"] = ""
        copied["refined_applied"] = refined
        copied["has_vector"] = copied.get("has_complete_tables", False)
        copied["vector_length"] = ""
        copied["record_index"] = idx
        round_records.append(copied)
    return round_records


def parse_log_file(path: Path, experiment: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bus = parse_case_from_name(path)
    if bus not in BUS_LIST:
        return [], {}

    case = load_case(bus, ROOT)
    created_at = ""
    solver_used = ""
    run_start_text = ""
    run_config: dict[str, str] = {}
    step_times: list[datetime] = []
    log_datetime = parse_datetime_from_name(path)

    metadata: dict[str, Any] = {
        "log_created_at": "",
        "log_datetime": log_datetime.isoformat(sep=" ") if log_datetime else "",
        "solver_used": "",
        "run_start": "",
        "N": "",
        "beam_refine": "",
        "refine_method": "",
        "refine": "",
        "alpha_mode": "",
        "alpha": "",
        "rho": "",
        "rho_max": "",
        "bound_shrink_factor": "",
        "bound_min_factor": "",
        "bound_start_iter": "",
        "run_duration_minutes": math.nan,
        "file_size_bytes": path.stat().st_size,
        "file_mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" "),
    }

    records: list[dict[str, Any]] = []
    current_iteration: int | None = None
    record: dict[str, Any] | None = None
    mode: str | None = None
    record_counter = 0

    def finish_record() -> None:
        nonlocal record, record_counter
        if record is None or "objective" not in record:
            record = None
            return
        computed = add_computed_metrics(record, case, bus)
        output_record = {
            **metadata,
            "experiment": experiment,
            "bus": bus,
            "log_name": path.name,
            "source_log": str(path.relative_to(ROOT)),
            "record_index": record_counter,
            "iteration": record.get("iteration"),
            "objective_log": record.get("objective", math.nan),
            "lalm_energy": record.get("lalm_energy", math.nan),
            "feasible": record.get("feasible", ""),
            "note": record.get("note", ""),
            "max_abs_h_log": record.get("max_abs_h_log", math.nan),
            "l2_norm_h_log": record.get("l2_norm_h_log", math.nan),
            "lambda_inf_norm": record.get("lambda_inf_norm", math.nan),
            "lambda_l2_norm": record.get("lambda_l2_norm", math.nan),
            "load_supplied_pct_log": record.get("load_supplied_pct_log", math.nan),
            **computed,
        }
        records.append(output_record)
        record_counter += 1
        record = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            step_match = re.match(r"\[step (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
            if step_match:
                try:
                    step_times.append(datetime.strptime(step_match.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    pass

            if line.startswith("Created at:"):
                created_at = parse_created_at(line)
                metadata["log_created_at"] = created_at
                continue
            if line.startswith("Solver used:"):
                solver_used = line.split(":", 1)[1].strip()
                metadata["solver_used"] = solver_used
                continue
            if "] run_start " in line:
                run_start_text = line.split("] run_start ", 1)[1]
                run_config = parse_run_start(line)
                metadata["run_start"] = run_start_text
                for key in [
                    "N",
                    "beam_refine",
                    "refine_method",
                    "refine",
                    "alpha_mode",
                    "alpha",
                    "rho",
                    "rho_max",
                    "bound_shrink_factor",
                    "bound_min_factor",
                    "bound_start_iter",
                ]:
                    metadata[key] = run_config.get(key, "")
                continue

            if line.startswith("Iteration:"):
                finish_record()
                current_iteration = safe_int(line.split(":", 1)[1])
                mode = None
                continue

            if line.startswith("objective_value:"):
                finish_record()
                record = {
                    "iteration": current_iteration,
                    "objective": safe_float(line.split(":", 1)[1]),
                    "buses": {},
                    "branches": {},
                }
                mode = None
                continue

            if record is None:
                continue

            if line.startswith("lalm_energy:"):
                record["lalm_energy"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("feasible:"):
                record["feasible"] = line.split(":", 1)[1].strip()
            elif line.startswith("note:"):
                record["note"] = line.split(":", 1)[1].strip()
            elif line.startswith("max_abs_h:"):
                record["max_abs_h_log"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("l2_norm_h:"):
                record["l2_norm_h_log"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("lambda_inf_norm:"):
                record["lambda_inf_norm"] = safe_float(line.split(":", 1)[1])
            elif line.startswith("lambda_l2_norm:"):
                record["lambda_l2_norm"] = safe_float(line.split(":", 1)[1])
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
                        "V_R": safe_float(parts[1]),
                        "V_I": safe_float(parts[2]),
                        "Vmag": safe_float(parts[3]),
                        "Pg": safe_float(parts[4]),
                        "Qg": safe_float(parts[5]),
                        "Pl": safe_float(parts[6]),
                        "Ql": safe_float(parts[7]),
                    }
            elif mode == "branch":
                parts = re.split(r"\s+", line)
                if len(parts) >= 11 and parts[0].isdigit():
                    line_id = int(parts[0])
                    record["branches"][line_id] = {
                        "From": safe_int(parts[1]),
                        "To": safe_int(parts[2]),
                        "Pik": safe_float(parts[3]),
                        "Pki": safe_float(parts[4]),
                        "Qik": safe_float(parts[5]),
                        "Qki": safe_float(parts[6]),
                        "Sik_sq": safe_float(parts[7]),
                        "Ski_sq": safe_float(parts[8]),
                        "LossP": safe_float(parts[9]),
                        "LossQ": safe_float(parts[10]),
                    }
            elif mode == "summary":
                if line.startswith("Total Load Supplied"):
                    record["load_supplied_pct_log"] = safe_float(line.split(":", 1)[1])

    finish_record()

    if step_times:
        duration = (max(step_times) - min(step_times)).total_seconds() / 60.0
        metadata["run_duration_minutes"] = duration
        for rec in records:
            rec["run_duration_minutes"] = duration

    if not records:
        fallback = parse_step_evaluations(path, experiment, bus, metadata)
        records.extend(fallback)

    return records, metadata


def fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_iteration_le(records: list[dict[str, Any]], threshold: float) -> int | None:
    candidates = [
        int(row["iteration"])
        for row in records
        if row.get("iteration") is not None
        and math.isfinite(float(row.get("l2_norm_h_used", math.nan)))
        and float(row["l2_norm_h_used"]) <= threshold
    ]
    return min(candidates) if candidates else None


def summarize_log(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    main = [row for row in records if row.get("is_main_record")]
    if not main:
        main = records

    finite_l2 = [row for row in main if math.isfinite(float(row.get("l2_norm_h_used", math.nan)))]
    finite_max = [row for row in main if math.isfinite(float(row.get("max_abs_h_used", math.nan)))]
    finite_obj = [row for row in main if math.isfinite(float(row.get("objective_used", math.nan)))]
    if not finite_l2:
        return None

    best_l2 = min(finite_l2, key=lambda row: float(row["l2_norm_h_used"]))
    best_max = min(finite_max, key=lambda row: float(row["max_abs_h_used"])) if finite_max else best_l2
    best_obj = min(finite_obj, key=lambda row: float(row["objective_used"])) if finite_obj else best_l2
    final = sorted(main, key=lambda row: (int(row.get("iteration") or -1), int(row.get("record_index") or -1)))[-1]
    first = sorted(main, key=lambda row: (int(row.get("iteration") or -1), int(row.get("record_index") or -1)))[0]

    base = {key: records[0].get(key, "") for key in [
        "experiment",
        "bus",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "run_start",
        "N",
        "beam_refine",
        "refine_method",
        "refine",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "file_size_bytes",
        "file_mtime",
    ]}
    ref_obj, ref_source = REFERENCE_OBJECTIVES.get(int(base["bus"]), (math.nan, ""))
    distinct_iterations = sorted({int(row["iteration"]) for row in main if row.get("iteration") is not None})
    answer_types = sorted({str(row.get("answer_type", "")) for row in main if row.get("answer_type")})

    summary = {
        **base,
        "reference_objective": ref_obj,
        "reference_source": ref_source,
        "main_record_count": len(main),
        "all_record_count": len(records),
        "distinct_iteration_count": len(distinct_iterations),
        "first_iteration": distinct_iterations[0] if distinct_iterations else None,
        "last_iteration": distinct_iterations[-1] if distinct_iterations else None,
        "answer_types": "|".join(answer_types),
        "first_objective": first.get("objective_used", math.nan),
        "first_l2_norm_h": first.get("l2_norm_h_used", math.nan),
        "best_l2_iteration": best_l2.get("iteration"),
        "best_l2_objective": best_l2.get("objective_used", math.nan),
        "best_l2_objective_gap": best_l2.get("objective_gap", math.nan),
        "best_l2_objective_gap_pct": best_l2.get("objective_gap_pct", math.nan),
        "best_l2_norm_h": best_l2.get("l2_norm_h_used", math.nan),
        "best_l2_max_abs_h": best_l2.get("max_abs_h_used", math.nan),
        "best_l2_load_supplied_pct": best_l2.get("load_supplied_pct_used", math.nan),
        "best_l2_note": best_l2.get("note", ""),
        "best_max_h_iteration": best_max.get("iteration"),
        "best_max_h_objective": best_max.get("objective_used", math.nan),
        "best_max_abs_h": best_max.get("max_abs_h_used", math.nan),
        "best_max_h_l2_norm_h": best_max.get("l2_norm_h_used", math.nan),
        "best_objective_iteration": best_obj.get("iteration"),
        "best_objective": best_obj.get("objective_used", math.nan),
        "best_objective_l2_norm_h": best_obj.get("l2_norm_h_used", math.nan),
        "best_objective_gap_pct": best_obj.get("objective_gap_pct", math.nan),
        "final_iteration": final.get("iteration"),
        "final_objective": final.get("objective_used", math.nan),
        "final_objective_gap": final.get("objective_gap", math.nan),
        "final_objective_gap_pct": final.get("objective_gap_pct", math.nan),
        "final_l2_norm_h": final.get("l2_norm_h_used", math.nan),
        "final_max_abs_h": final.get("max_abs_h_used", math.nan),
        "final_load_supplied_pct": final.get("load_supplied_pct_used", math.nan),
        "first_l2_le_1e_2_iter": first_iteration_le(main, 1e-2),
        "first_l2_le_1e_3_iter": first_iteration_le(main, 1e-3),
        "first_l2_le_1e_4_iter": first_iteration_le(main, 1e-4),
        "first_l2_le_1e_5_iter": first_iteration_le(main, 1e-5),
    }
    return summary


def summarize_round_stage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    finite_l2 = [row for row in records if math.isfinite(float(row.get("l2_norm_h_used", math.nan)))]
    finite_max = [row for row in records if math.isfinite(float(row.get("max_abs_h_used", math.nan)))]
    finite_obj = [row for row in records if math.isfinite(float(row.get("objective_used", math.nan)))]
    if not finite_l2:
        return None

    best_l2 = min(finite_l2, key=lambda row: float(row["l2_norm_h_used"]))
    best_max = min(finite_max, key=lambda row: float(row["max_abs_h_used"])) if finite_max else best_l2
    best_obj = min(finite_obj, key=lambda row: float(row["objective_used"])) if finite_obj else best_l2
    final = sorted(records, key=lambda row: (int(row.get("iteration") or -1), int(row.get("record_index") or -1)))[-1]
    first = sorted(records, key=lambda row: (int(row.get("iteration") or -1), int(row.get("record_index") or -1)))[0]
    base = {key: records[0].get(key, "") for key in [
        "experiment",
        "bus",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "run_start",
        "N",
        "beam_refine",
        "refine_method",
        "refine",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "file_size_bytes",
        "file_mtime",
    ]}
    ref_obj, ref_source = REFERENCE_OBJECTIVES.get(int(base["bus"]), (math.nan, ""))
    distinct_iterations = sorted({int(row["iteration"]) for row in records if row.get("iteration") is not None})
    return {
        **base,
        "stage": records[0].get("stage", ""),
        "reference_objective": ref_obj,
        "reference_source": ref_source,
        "record_count": len(records),
        "distinct_iteration_count": len(distinct_iterations),
        "first_iteration": distinct_iterations[0] if distinct_iterations else None,
        "last_iteration": distinct_iterations[-1] if distinct_iterations else None,
        "first_objective": first.get("objective_used", math.nan),
        "first_l2_norm_h": first.get("l2_norm_h_used", math.nan),
        "best_l2_iteration": best_l2.get("iteration"),
        "best_l2_objective": best_l2.get("objective_used", math.nan),
        "best_l2_objective_gap": best_l2.get("objective_gap", math.nan),
        "best_l2_objective_gap_pct": best_l2.get("objective_gap_pct", math.nan),
        "best_l2_norm_h": best_l2.get("l2_norm_h_used", math.nan),
        "best_l2_max_abs_h": best_l2.get("max_abs_h_used", math.nan),
        "best_l2_load_supplied_pct": best_l2.get("load_supplied_pct_used", math.nan),
        "best_max_h_iteration": best_max.get("iteration"),
        "best_max_h_objective": best_max.get("objective_used", math.nan),
        "best_max_abs_h": best_max.get("max_abs_h_used", math.nan),
        "best_max_h_l2_norm_h": best_max.get("l2_norm_h_used", math.nan),
        "best_objective_iteration": best_obj.get("iteration"),
        "best_objective": best_obj.get("objective_used", math.nan),
        "best_objective_l2_norm_h": best_obj.get("l2_norm_h_used", math.nan),
        "best_objective_gap_pct": best_obj.get("objective_gap_pct", math.nan),
        "final_iteration": final.get("iteration"),
        "final_objective": final.get("objective_used", math.nan),
        "final_objective_gap": final.get("objective_gap", math.nan),
        "final_objective_gap_pct": final.get("objective_gap_pct", math.nan),
        "final_l2_norm_h": final.get("l2_norm_h_used", math.nan),
        "final_max_abs_h": final.get("max_abs_h_used", math.nan),
        "final_load_supplied_pct": final.get("load_supplied_pct_used", math.nan),
        "first_l2_le_1e_2_iter": first_iteration_le(records, 1e-2),
        "first_l2_le_1e_3_iter": first_iteration_le(records, 1e-3),
        "first_l2_le_1e_4_iter": first_iteration_le(records, 1e-4),
        "first_l2_le_1e_5_iter": first_iteration_le(records, 1e-5),
    }


def latest_key(summary: dict[str, Any]) -> tuple[str, str]:
    log_dt = summary.get("log_datetime") or ""
    mtime = summary.get("file_mtime") or ""
    return str(log_dt), str(mtime)


def select_latest(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["experiment"], int(row["bus"]))].append(row)
    selected = []
    for key in sorted(grouped):
        selected.append(max(grouped[key], key=latest_key))
    return selected


def select_latest_stage(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["experiment"], int(row["bus"]), str(row.get("stage", "")))].append(row)
    selected = []
    for key in sorted(grouped):
        selected.append(max(grouped[key], key=latest_key))
    return selected


def select_best_by_l2_stage(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["experiment"], int(row["bus"]), str(row.get("stage", "")))].append(row)
    selected = []
    for key in sorted(grouped):
        selected.append(min(grouped[key], key=lambda row: float(row.get("best_l2_norm_h", math.inf))))
    return selected


def select_best_by_l2(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["experiment"], int(row["bus"]))].append(row)
    selected = []
    for key in sorted(grouped):
        selected.append(min(grouped[key], key=lambda row: float(row.get("best_l2_norm_h", math.inf))))
    return selected


def plot_best_l2(selected: list[dict[str, Any]], path: Path) -> None:
    buses = BUS_LIST
    experiments = [name for name, _ in EXPERIMENTS if any(row["experiment"] == name for row in selected)]
    x = np.arange(len(buses), dtype=float)
    width = 0.8 / max(1, len(experiments))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for idx, exp in enumerate(experiments):
        values = []
        for bus in buses:
            row = next((item for item in selected if item["experiment"] == exp and int(item["bus"]) == bus), None)
            values.append(float(row["best_l2_norm_h"]) if row else np.nan)
        ax.bar(x + (idx - (len(experiments) - 1) / 2) * width, values, width=width, label=exp)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bus}-bus" for bus in buses])
    ax.set_ylabel("Best L2 constraint residual")
    ax.set_title("Best residual by experiment")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_objective_gap(selected: list[dict[str, Any]], path: Path) -> None:
    buses = BUS_LIST
    experiments = [name for name, _ in EXPERIMENTS if any(row["experiment"] == name for row in selected)]
    x = np.arange(len(buses), dtype=float)
    width = 0.8 / max(1, len(experiments))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for idx, exp in enumerate(experiments):
        values = []
        for bus in buses:
            row = next((item for item in selected if item["experiment"] == exp and int(item["bus"]) == bus), None)
            values.append(float(row["best_l2_objective_gap_pct"]) if row else np.nan)
        ax.bar(x + (idx - (len(experiments) - 1) / 2) * width, values, width=width, label=exp)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bus}-bus" for bus in buses])
    ax.set_ylabel("Objective gap at best residual (%)")
    ax.set_title("Objective gap at best-residual iteration")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_threshold_iterations(selected: list[dict[str, Any]], path: Path) -> None:
    buses = BUS_LIST
    experiments = [name for name, _ in EXPERIMENTS if any(row["experiment"] == name for row in selected)]
    x = np.arange(len(buses), dtype=float)
    width = 0.8 / max(1, len(experiments))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for idx, exp in enumerate(experiments):
        values = []
        for bus in buses:
            row = next((item for item in selected if item["experiment"] == exp and int(item["bus"]) == bus), None)
            value = row.get("first_l2_le_1e_4_iter") if row else None
            values.append(float(value) if value not in (None, "") else np.nan)
        ax.bar(x + (idx - (len(experiments) - 1) / 2) * width, values, width=width, label=exp)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bus}-bus" for bus in buses])
    ax.set_ylabel("First iteration with L2 residual <= 1e-4")
    ax.set_title("Iterations to 1e-4 residual threshold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_paired_stage_rows(selected_stage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected_stage:
        by_key[(row["experiment"], int(row["bus"]))][str(row.get("stage", ""))] = row

    paired = []
    for (experiment, bus), stages in sorted(by_key.items(), key=lambda item: (item[0][1], item[0][0])):
        coarse = stages.get("coarse_pre_refine")
        post = stages.get("post_refine_or_final")
        if not coarse and not post:
            continue
        coarse_l2 = float(coarse.get("best_l2_norm_h", math.nan)) if coarse else math.nan
        post_l2 = float(post.get("best_l2_norm_h", math.nan)) if post else math.nan
        improvement = coarse_l2 / post_l2 if math.isfinite(coarse_l2) and math.isfinite(post_l2) and post_l2 > 0 else math.nan
        paired.append(
            {
                "experiment": experiment,
                "bus": bus,
                "log_name": (post or coarse).get("log_name", ""),
                "N": (post or coarse).get("N", ""),
                "coarse_best_iteration": coarse.get("best_l2_iteration", "") if coarse else "",
                "coarse_best_l2_norm_h": coarse_l2,
                "coarse_best_objective_gap_pct": coarse.get("best_l2_objective_gap_pct", math.nan) if coarse else math.nan,
                "post_best_iteration": post.get("best_l2_iteration", "") if post else "",
                "post_best_l2_norm_h": post_l2,
                "post_best_objective_gap_pct": post.get("best_l2_objective_gap_pct", math.nan) if post else math.nan,
                "post_final_l2_norm_h": post.get("final_l2_norm_h", math.nan) if post else math.nan,
                "refine_improvement_ratio": improvement,
                "refine_improvement_pct": 100.0 * (1.0 - post_l2 / coarse_l2)
                if math.isfinite(coarse_l2) and math.isfinite(post_l2) and coarse_l2 > 0
                else math.nan,
            }
        )
    return paired


def plot_coarse_vs_post(paired_rows: list[dict[str, Any]], path: Path) -> None:
    buses = BUS_LIST
    experiments = [name for name, _ in EXPERIMENTS if any(row["experiment"] == name for row in paired_rows)]
    x = np.arange(len(buses), dtype=float)
    width = 0.8 / max(1, len(experiments))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for idx, exp in enumerate(experiments):
        values = []
        for bus in buses:
            row = next((item for item in paired_rows if item["experiment"] == exp and int(item["bus"]) == bus), None)
            values.append(float(row["refine_improvement_ratio"]) if row else np.nan)
        ax.bar(x + (idx - (len(experiments) - 1) / 2) * width, values, width=width, label=exp)
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bus}-bus" for bus in buses])
    ax.set_ylabel("Best coarse L2 / best post-refine L2")
    ax.set_title("Residual improvement from refine stage")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_convergence_by_bus(selected_rows: list[dict[str, Any]], iteration_rows: list[dict[str, Any]]) -> None:
    selected_logs = {(row["experiment"], int(row["bus"]), row["log_name"]) for row in selected_rows}
    selected_log_stages = {
        (row["experiment"], int(row["bus"]), row["log_name"], str(row.get("stage", "")))
        for row in selected_rows
        if row.get("stage")
    }
    filter_stage = bool(selected_log_stages)
    for bus in BUS_LIST:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for exp, _path in EXPERIMENTS:
            matching = [
                row
                for row in iteration_rows
                if row.get("is_main_record")
                and int(row["bus"]) == bus
                and (row["experiment"], int(row["bus"]), row["log_name"]) in selected_logs
                and row["experiment"] == exp
                and (
                    not filter_stage
                    or (row["experiment"], int(row["bus"]), row["log_name"], str(row.get("stage", ""))) in selected_log_stages
                )
                and math.isfinite(float(row.get("l2_norm_h_used", math.nan)))
            ]
            if not matching:
                continue
            matching.sort(key=lambda row: (int(row.get("iteration") or 0), int(row.get("record_index") or 0)))
            ax.plot(
                [int(row["iteration"]) for row in matching],
                [float(row["l2_norm_h_used"]) for row in matching],
                linewidth=1.6,
                label=exp,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("L2 constraint residual")
        ax.set_title(f"{bus}-bus convergence")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(fontsize=9)
        fig.savefig(OUTPUT_DIR / f"convergence_l2_{bus}bus.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for exp, _path in EXPERIMENTS:
            matching = [
                row
                for row in iteration_rows
                if row.get("is_main_record")
                and int(row["bus"]) == bus
                and (row["experiment"], int(row["bus"]), row["log_name"]) in selected_logs
                and row["experiment"] == exp
                and (
                    not filter_stage
                    or (row["experiment"], int(row["bus"]), row["log_name"], str(row.get("stage", ""))) in selected_log_stages
                )
                and math.isfinite(float(row.get("objective_gap_pct", math.nan)))
            ]
            if not matching:
                continue
            matching.sort(key=lambda row: (int(row.get("iteration") or 0), int(row.get("record_index") or 0)))
            ax.plot(
                [int(row["iteration"]) for row in matching],
                [float(row["objective_gap_pct"]) for row in matching],
                linewidth=1.6,
                label=exp,
            )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective gap (%)")
        ax.set_title(f"{bus}-bus objective gap")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(fontsize=9)
        fig.savefig(OUTPUT_DIR / f"convergence_objective_gap_{bus}bus.png", dpi=220)
        plt.close(fig)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No data._\n"
    header = "| " + " | ".join(label for _key, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value) if value is not None else "")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_analysis(
    selected_latest: list[dict[str, Any]],
    best_across: list[dict[str, Any]],
    all_summaries: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    coverage_rows = []
    for exp, _path in EXPERIMENTS:
        for bus in BUS_LIST:
            logs = [row for row in all_summaries if row["experiment"] == exp and int(row["bus"]) == bus]
            selected = next((row for row in selected_latest if row["experiment"] == exp and int(row["bus"]) == bus), None)
            coverage_rows.append(
                {
                    "experiment": exp,
                    "bus": bus,
                    "log_count": len(logs),
                    "selected_log": selected["log_name"] if selected else "",
                    "selected_records": (
                        selected.get("main_record_count", selected.get("record_count", ""))
                        if selected
                        else ""
                    ),
                }
            )

    comparison_rows = []
    for row in sorted(selected_latest, key=lambda item: (int(item["bus"]), item["experiment"])):
        comparison_rows.append(
            {
                "bus": f"{row['bus']}-bus",
                "experiment": row["experiment"],
                "N": row.get("N", ""),
                "best_iter": row.get("best_l2_iteration", ""),
                "best_l2": float(row.get("best_l2_norm_h", math.nan)),
                "best_max_h": float(row.get("best_l2_max_abs_h", math.nan)),
                "best_obj_gap_pct": float(row.get("best_l2_objective_gap_pct", math.nan)),
                "final_l2": float(row.get("final_l2_norm_h", math.nan)),
                "first_1e4": row.get("first_l2_le_1e_4_iter", ""),
                "log": row["log_name"],
            }
        )

    best_lines = []
    for bus in BUS_LIST:
        bus_rows = [row for row in selected_latest if int(row["bus"]) == bus]
        if not bus_rows:
            continue
        best = min(bus_rows, key=lambda row: float(row.get("best_l2_norm_h", math.inf)))
        best_lines.append(
            f"- {bus}-bus: best latest-log residual is {fmt(best['best_l2_norm_h'])} "
            f"from `{best['experiment']}` at iteration {best['best_l2_iteration']} "
            f"(objective gap {fmt(best['best_l2_objective_gap_pct'])}%)."
        )

    pair_lines = []
    for bus in BUS_LIST:
        single = next((row for row in selected_latest if int(row["bus"]) == bus and row["experiment"] == "single_beam"), None)
        multi = next((row for row in selected_latest if int(row["bus"]) == bus and row["experiment"] == "multi_beam"), None)
        if not single or not multi:
            continue
        single_l2 = float(single["best_l2_norm_h"])
        multi_l2 = float(multi["best_l2_norm_h"])
        winner = "multi_beam" if multi_l2 < single_l2 else "single_beam"
        ratio = max(single_l2, multi_l2) / max(min(single_l2, multi_l2), 1e-300)
        pair_lines.append(
            f"- {bus}-bus: `{winner}` has the lower best residual; the gap is about {fmt(ratio, 2)}x "
            f"({fmt(multi_l2)} for multi_beam vs {fmt(single_l2)} for single_beam)."
        )

    reference_rows = [
        {"bus": f"{bus}-bus", "objective": obj, "source": source}
        for bus, (obj, source) in REFERENCE_OBJECTIVES.items()
        if bus in BUS_LIST
    ]

    analysis = [
        "# QHD-LALM-SB Log Analysis for 2/3/5/9-Bus Cases",
        "",
        "This report combines the requested log folders and excludes 14-bus results. "
        "All detailed source data for the plots is saved as CSV in this same `output` folder.",
        "",
        "## Data Scope",
        markdown_table(
            coverage_rows,
            [
                ("experiment", "Experiment"),
                ("bus", "Bus"),
                ("log_count", "Logs"),
                ("selected_log", "Latest selected log"),
                ("selected_records", "Main records"),
            ],
        ),
        "## Reference Objectives",
        markdown_table(reference_rows, [("bus", "Bus"), ("objective", "Reference objective"), ("source", "Source")]),
        "## Latest-Log Post-Refine/Final Comparison",
        "This table uses the per-round result after the coarse QHD/SB solve plus refinement when refinement is present. "
        "If no refinement vector is present for a round, the final coarse vector is used.",
        "",
        markdown_table(
            comparison_rows,
            [
                ("bus", "Bus"),
                ("experiment", "Experiment"),
                ("N", "N"),
                ("best_iter", "Best iter"),
                ("best_l2", "Best L2 residual"),
                ("best_max_h", "Max |h| at best L2"),
                ("best_obj_gap_pct", "Obj gap %"),
                ("final_l2", "Final L2"),
                ("first_1e4", "First <=1e-4"),
                ("log", "Log"),
            ],
        ),
        "## Coarse vs Post-Refine Paired Comparison",
        markdown_table(
            [
                {
                    "bus": f"{row['bus']}-bus",
                    "experiment": row["experiment"],
                    "N": row.get("N", ""),
                    "coarse_iter": row.get("coarse_best_iteration", ""),
                    "coarse_l2": row.get("coarse_best_l2_norm_h", math.nan),
                    "post_iter": row.get("post_best_iteration", ""),
                    "post_l2": row.get("post_best_l2_norm_h", math.nan),
                    "improve_ratio": row.get("refine_improvement_ratio", math.nan),
                    "improve_pct": row.get("refine_improvement_pct", math.nan),
                }
                for row in paired_rows
            ],
            [
                ("bus", "Bus"),
                ("experiment", "Experiment"),
                ("N", "N"),
                ("coarse_iter", "Coarse best iter"),
                ("coarse_l2", "Coarse best L2"),
                ("post_iter", "Post best iter"),
                ("post_l2", "Post/refined best L2"),
                ("improve_ratio", "Coarse/Post ratio"),
                ("improve_pct", "Residual reduction %"),
            ],
        ),
        "## Key Observations",
        "\n".join(best_lines) if best_lines else "- No selected logs were available.",
        "",
        "\n".join(pair_lines) if pair_lines else "- No single/multi beam pairs were available.",
        "",
        "- The post-refine/final stage is now compared explicitly against the coarse-pre-refine stage for each round.",
        "- The best-residual iteration is often better than the final iteration; this is especially visible where later iterations drift after reaching a low residual.",
        "- The residual columns use recomputed ACOPF residuals when the bus/branch tables are complete; otherwise they fall back to the values printed in the log.",
        "",
        "## Generated Files",
        "- `iteration_metrics.csv`: all parsed per-iteration records.",
        "- `round_stage_metrics.csv`: paired coarse and post-refine/final source data parsed from `coarse_x` and final `x` vectors.",
        "- `round_stage_summary.csv`: one row per log/stage.",
        "- `selected_latest_round_stage_summary.csv`: latest valid log for each experiment/bus/stage.",
        "- `paired_coarse_post_latest.csv`: coarse-vs-post-refine comparison for the latest selected logs.",
        "- `plot_source_selected_convergence.csv`: post-refine/final per-iteration source data used by the convergence plots.",
        "- `all_runs_summary.csv`: one row per parsed log.",
        "- `selected_latest_summary.csv`: latest valid log for each experiment/bus pair.",
        "- `best_across_logs_summary.csv`: best historical log for each experiment/bus pair by L2 residual.",
        "- `best_l2_residual_by_experiment.png`, `objective_gap_pct_by_experiment.png`, `iterations_to_1e4_by_experiment.png`: post-refine/final cross-case comparison plots.",
        "- `coarse_vs_post_refine_improvement.png`: paired residual improvement from refine.",
        "- `convergence_l2_*bus.png` and `convergence_objective_gap_*bus.png`: bus-specific convergence plots.",
        "",
    ]

    (OUTPUT_DIR / "analysis.md").write_text("\n".join(analysis), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_records: list[dict[str, Any]] = []
    all_round_records: list[dict[str, Any]] = []

    for experiment, directory in EXPERIMENTS:
        files = sorted(path for path in directory.glob("Buses-*.txt") if parse_case_from_name(path) in BUS_LIST)
        if not files:
            continue
        for path in files:
            records, _metadata = parse_log_file(path, experiment)
            all_records.extend(records)
            round_records = parse_step_round_records(path, experiment, _metadata)
            if not round_records:
                round_records = round_records_from_table_records(records)
            all_round_records.extend(round_records)

    fieldnames = [
        "experiment",
        "bus",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "N",
        "beam_refine",
        "refine",
        "refine_method",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "record_index",
        "iteration",
        "answer_type",
        "is_main_record",
        "objective_log",
        "objective_recomputed",
        "objective_used",
        "objective_gap",
        "objective_gap_pct",
        "lalm_energy",
        "feasible",
        "note",
        "max_abs_h_log",
        "l2_norm_h_log",
        "max_abs_h_recomputed",
        "l2_norm_h_recomputed",
        "max_abs_h_used",
        "l2_norm_h_used",
        "lambda_inf_norm",
        "lambda_l2_norm",
        "load_supplied_pct_log",
        "load_supplied_pct_used",
        "has_complete_tables",
        "file_size_bytes",
        "file_mtime",
    ]
    write_csv(OUTPUT_DIR / "iteration_metrics.csv", all_records, fieldnames)

    round_fieldnames = [
        "experiment",
        "bus",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "N",
        "beam_refine",
        "refine",
        "refine_method",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "record_index",
        "iteration",
        "stage",
        "rank",
        "refined_applied",
        "is_main_record",
        "objective_log",
        "objective_recomputed",
        "objective_used",
        "objective_gap",
        "objective_gap_pct",
        "lalm_energy",
        "note",
        "max_abs_h_used",
        "l2_norm_h_used",
        "load_supplied_pct_used",
        "has_vector",
        "vector_length",
        "file_size_bytes",
        "file_mtime",
    ]
    write_csv(OUTPUT_DIR / "round_stage_metrics.csv", all_round_records, round_fieldnames)

    by_log: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_records:
        by_log[(row["experiment"], row["log_name"])].append(row)

    summaries = []
    for records in by_log.values():
        summary = summarize_log(records)
        if summary is not None:
            summaries.append(summary)

    summary_fields = [
        "experiment",
        "bus",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "N",
        "beam_refine",
        "refine",
        "refine_method",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "reference_objective",
        "reference_source",
        "main_record_count",
        "all_record_count",
        "distinct_iteration_count",
        "first_iteration",
        "last_iteration",
        "answer_types",
        "first_objective",
        "first_l2_norm_h",
        "best_l2_iteration",
        "best_l2_objective",
        "best_l2_objective_gap",
        "best_l2_objective_gap_pct",
        "best_l2_norm_h",
        "best_l2_max_abs_h",
        "best_l2_load_supplied_pct",
        "best_l2_note",
        "best_max_h_iteration",
        "best_max_h_objective",
        "best_max_abs_h",
        "best_max_h_l2_norm_h",
        "best_objective_iteration",
        "best_objective",
        "best_objective_l2_norm_h",
        "best_objective_gap_pct",
        "final_iteration",
        "final_objective",
        "final_objective_gap",
        "final_objective_gap_pct",
        "final_l2_norm_h",
        "final_max_abs_h",
        "final_load_supplied_pct",
        "first_l2_le_1e_2_iter",
        "first_l2_le_1e_3_iter",
        "first_l2_le_1e_4_iter",
        "first_l2_le_1e_5_iter",
        "file_size_bytes",
        "file_mtime",
    ]
    summaries.sort(key=lambda row: (int(row["bus"]), row["experiment"], row["log_name"]))
    write_csv(OUTPUT_DIR / "all_runs_summary.csv", summaries, summary_fields)

    selected_latest = select_latest(summaries)
    selected_latest.sort(key=lambda row: (int(row["bus"]), row["experiment"]))
    write_csv(OUTPUT_DIR / "selected_latest_summary.csv", selected_latest, summary_fields)

    best_across = select_best_by_l2(summaries)
    best_across.sort(key=lambda row: (int(row["bus"]), row["experiment"]))
    write_csv(OUTPUT_DIR / "best_across_logs_summary.csv", best_across, summary_fields)

    by_log_stage: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_round_records:
        by_log_stage[(row["experiment"], row["log_name"], row["stage"])].append(row)

    stage_summaries = []
    for records in by_log_stage.values():
        summary = summarize_round_stage(records)
        if summary is not None:
            stage_summaries.append(summary)

    stage_summary_fields = [
        "experiment",
        "bus",
        "stage",
        "log_name",
        "source_log",
        "log_created_at",
        "log_datetime",
        "solver_used",
        "N",
        "beam_refine",
        "refine",
        "refine_method",
        "alpha_mode",
        "alpha",
        "rho",
        "rho_max",
        "bound_shrink_factor",
        "bound_min_factor",
        "bound_start_iter",
        "run_duration_minutes",
        "reference_objective",
        "reference_source",
        "record_count",
        "distinct_iteration_count",
        "first_iteration",
        "last_iteration",
        "first_objective",
        "first_l2_norm_h",
        "best_l2_iteration",
        "best_l2_objective",
        "best_l2_objective_gap",
        "best_l2_objective_gap_pct",
        "best_l2_norm_h",
        "best_l2_max_abs_h",
        "best_l2_load_supplied_pct",
        "best_max_h_iteration",
        "best_max_h_objective",
        "best_max_abs_h",
        "best_max_h_l2_norm_h",
        "best_objective_iteration",
        "best_objective",
        "best_objective_l2_norm_h",
        "best_objective_gap_pct",
        "final_iteration",
        "final_objective",
        "final_objective_gap",
        "final_objective_gap_pct",
        "final_l2_norm_h",
        "final_max_abs_h",
        "final_load_supplied_pct",
        "first_l2_le_1e_2_iter",
        "first_l2_le_1e_3_iter",
        "first_l2_le_1e_4_iter",
        "first_l2_le_1e_5_iter",
        "file_size_bytes",
        "file_mtime",
    ]
    stage_summaries.sort(key=lambda row: (int(row["bus"]), row["experiment"], row["stage"], row["log_name"]))
    write_csv(OUTPUT_DIR / "round_stage_summary.csv", stage_summaries, stage_summary_fields)

    selected_latest_stage = select_latest_stage(stage_summaries)
    selected_latest_stage.sort(key=lambda row: (int(row["bus"]), row["experiment"], row["stage"]))
    write_csv(OUTPUT_DIR / "selected_latest_round_stage_summary.csv", selected_latest_stage, stage_summary_fields)

    best_across_stage = select_best_by_l2_stage(stage_summaries)
    best_across_stage.sort(key=lambda row: (int(row["bus"]), row["experiment"], row["stage"]))
    write_csv(OUTPUT_DIR / "best_across_round_stage_summary.csv", best_across_stage, stage_summary_fields)

    selected_latest_post = [
        row for row in selected_latest_stage if row.get("stage") == "post_refine_or_final"
    ]
    paired_latest = build_paired_stage_rows(selected_latest_stage)
    paired_fields = [
        "experiment",
        "bus",
        "log_name",
        "N",
        "coarse_best_iteration",
        "coarse_best_l2_norm_h",
        "coarse_best_objective_gap_pct",
        "post_best_iteration",
        "post_best_l2_norm_h",
        "post_best_objective_gap_pct",
        "post_final_l2_norm_h",
        "refine_improvement_ratio",
        "refine_improvement_pct",
    ]
    write_csv(OUTPUT_DIR / "paired_coarse_post_latest.csv", paired_latest, paired_fields)

    selected_keys = {
        (row["experiment"], int(row["bus"]), row["log_name"], row["stage"])
        for row in selected_latest_post
    }
    selected_iteration_rows = [
        row
        for row in all_round_records
        if row.get("is_main_record")
        and (row["experiment"], int(row["bus"]), row["log_name"], row.get("stage")) in selected_keys
    ]
    write_csv(OUTPUT_DIR / "plot_source_selected_convergence.csv", selected_iteration_rows, round_fieldnames)

    plot_best_l2(selected_latest_post, OUTPUT_DIR / "best_l2_residual_by_experiment.png")
    plot_objective_gap(selected_latest_post, OUTPUT_DIR / "objective_gap_pct_by_experiment.png")
    plot_threshold_iterations(selected_latest_post, OUTPUT_DIR / "iterations_to_1e4_by_experiment.png")
    plot_coarse_vs_post(paired_latest, OUTPUT_DIR / "coarse_vs_post_refine_improvement.png")
    plot_convergence_by_bus(selected_latest_post, all_round_records)

    write_analysis(selected_latest_post, best_across, summaries, all_round_records, paired_latest)

    print(f"Parsed records: {len(all_records)}")
    print(f"Parsed round-stage records: {len(all_round_records)}")
    print(f"Parsed summaries: {len(summaries)}")
    print(f"Parsed round-stage summaries: {len(stage_summaries)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
