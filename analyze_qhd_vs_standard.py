#!/usr/bin/env python
"""Compare QHD/Simbi ACOPF logs against standard answer files.

The script parses the repository's answer text files and QHD iteration logs,
computes Euclidean distances over bus variables and branch-flow variables, and
writes plots/CSV summaries under logs/.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"


CASES = {
    "5bus": {
        "standard": ROOT / "5bus-answer.txt",
        "qhd_log": LOG_DIR / "Buses-5_06-07-2026_00-05-09.txt",
    },
    "9bus": {
        "standard": ROOT / "9bus-answer.txt",
        "qhd_log": LOG_DIR / "Buses-9_06-08-2026_19-48-09.txt",
    },
    "14bus": {
        "standard": ROOT / "14bus-answer.txt",
        "qhd_log": LOG_DIR / "Buses-14_06-08-2026_11-02-24.txt",
    },
}


@dataclass
class StandardAnswer:
    objective: float
    buses: dict[int, dict[str, float]]
    branches: dict[tuple[int, int], dict[str, float]]


@dataclass
class IterationRecord:
    iteration: int
    objective: float | None = None
    max_abs_h: float | None = None
    l2_norm_h: float | None = None
    total_pg: float | None = None
    total_load_p: float | None = None
    load_supplied: float | None = None
    buses: dict[int, dict[str, float]] = field(default_factory=dict)
    branches: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    bus_distance: float | None = None
    branch_distance: float | None = None
    combined_distance: float | None = None
    objective_diff: float | None = None
    objective_abs_diff: float | None = None
    objective_pct_diff: float | None = None


def _floats_from_line(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)]


def parse_standard_answer(path: Path) -> StandardAnswer:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    objective = None
    buses: dict[int, dict[str, float]] = {}
    branches: dict[tuple[int, int], dict[str, float]] = {}

    mode = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("Objective Function Value"):
            objective = float(stripped.split(":", 1)[1].strip())
            continue

        if stripped.startswith("BusID"):
            mode = "bus"
            continue

        if stripped.startswith("Busi"):
            mode = "branch"
            continue

        if stripped.startswith("TOTAL") or stripped.startswith("Total "):
            mode = None
            continue

        if mode == "bus":
            parts = stripped.split()
            if len(parts) >= 5 and parts[0].isdigit():
                bus_id = int(parts[0])
                buses[bus_id] = {
                    "VR": float(parts[1]),
                    "VI": float(parts[2]),
                    "Pg": float(parts[3]),
                    "Qg": float(parts[4]),
                }
            continue

        if mode == "branch":
            parts = stripped.split()
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                i, j = int(parts[0]), int(parts[1])
                branches[(i, j)] = {
                    "Pik": float(parts[2]),
                    "Pki": float(parts[3]),
                    "Qik": float(parts[4]),
                    "Qki": float(parts[5]),
                }

    if objective is None:
        raise ValueError(f"Could not parse objective from {path}")
    if not buses:
        raise ValueError(f"Could not parse bus table from {path}")
    if not branches:
        raise ValueError(f"Could not parse branch table from {path}")
    return StandardAnswer(objective=objective, buses=buses, branches=branches)


def parse_qhd_log(path: Path) -> list[IterationRecord]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[IterationRecord] = []
    current: IterationRecord | None = None
    mode = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Iteration:"):
            if current is not None:
                records.append(current)
            current = IterationRecord(iteration=int(stripped.split(":", 1)[1].strip()))
            mode = None
            continue

        if current is None or not stripped:
            continue

        if stripped.startswith("objective_value:"):
            current.objective = float(stripped.split(":", 1)[1].strip())
            continue
        if stripped.startswith("max_abs_h:"):
            current.max_abs_h = float(stripped.split(":", 1)[1].strip())
            continue
        if stripped.startswith("l2_norm_h:"):
            current.l2_norm_h = float(stripped.split(":", 1)[1].strip())
            continue

        if stripped.startswith("BusID"):
            mode = "bus"
            continue
        if stripped.startswith("LineID"):
            mode = "branch"
            continue
        if stripped.startswith("Summary"):
            mode = "summary"
            continue

        if mode == "bus":
            parts = stripped.split()
            if len(parts) >= 8 and parts[0].isdigit():
                bus_id = int(parts[0])
                current.buses[bus_id] = {
                    "VR": float(parts[1]),
                    "VI": float(parts[2]),
                    "Pg": float(parts[4]),
                    "Qg": float(parts[5]),
                }
            continue

        if mode == "branch":
            parts = stripped.split()
            if len(parts) >= 9 and parts[0].isdigit():
                i, j = int(parts[1]), int(parts[2])
                current.branches[(i, j)] = {
                    "Pik": float(parts[3]),
                    "Pki": float(parts[4]),
                    "Qik": float(parts[5]),
                    "Qki": float(parts[6]),
                }
            continue

        if mode == "summary":
            if stripped.startswith("Total Pg"):
                current.total_pg = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Total Load P"):
                current.total_load_p = float(stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Total Load Supplied"):
                current.load_supplied = float(stripped.split(":", 1)[1].strip().rstrip("%"))

    if current is not None:
        records.append(current)

    return [record for record in records if record.objective is not None and record.buses]


def compute_distances(records: list[IterationRecord], standard: StandardAnswer) -> None:
    for record in records:
        bus_sq = 0.0
        for bus_id, std_bus in standard.buses.items():
            qhd_bus = record.buses.get(bus_id)
            if qhd_bus is None:
                raise ValueError(f"Missing QHD bus {bus_id} at iteration {record.iteration}")
            for key in ("VR", "VI", "Pg", "Qg"):
                bus_sq += (qhd_bus[key] - std_bus[key]) ** 2

        branch_sq = 0.0
        for key, std_branch in standard.branches.items():
            qhd_branch = record.branches.get(key)
            if qhd_branch is None:
                reverse = (key[1], key[0])
                qhd_reverse = record.branches.get(reverse)
                if qhd_reverse is None:
                    raise ValueError(f"Missing QHD branch {key} at iteration {record.iteration}")
                qhd_branch = {
                    "Pik": qhd_reverse["Pki"],
                    "Pki": qhd_reverse["Pik"],
                    "Qik": qhd_reverse["Qki"],
                    "Qki": qhd_reverse["Qik"],
                }
            for flow_key in ("Pik", "Pki", "Qik", "Qki"):
                branch_sq += (qhd_branch[flow_key] - std_branch[flow_key]) ** 2

        record.bus_distance = math.sqrt(bus_sq)
        record.branch_distance = math.sqrt(branch_sq)
        record.combined_distance = math.sqrt(bus_sq + branch_sq)
        record.objective_diff = record.objective - standard.objective
        record.objective_abs_diff = abs(record.objective_diff)
        record.objective_pct_diff = 100.0 * record.objective_diff / standard.objective


def best_by_combined(records: list[IterationRecord]) -> IterationRecord:
    return min(records, key=lambda item: item.combined_distance if item.combined_distance is not None else math.inf)


def best_by_l2(records: list[IterationRecord]) -> IterationRecord:
    return min(records, key=lambda item: item.l2_norm_h if item.l2_norm_h is not None else math.inf)


def plot_case(case_name: str, records: list[IterationRecord], out_path: Path) -> None:
    iterations = [r.iteration for r in records]
    objective_diff = [r.objective_diff for r in records]
    bus_dist = [r.bus_distance for r in records]
    branch_dist = [r.branch_distance for r in records]
    combined_dist = [r.combined_distance for r in records]
    max_abs_h = [r.max_abs_h for r in records]
    l2_norm_h = [r.l2_norm_h for r in records]
    load_supplied = [r.load_supplied for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{case_name}: QHD/Simbi vs standard answer", fontsize=14)

    ax = axes[0, 0]
    ax.plot(iterations, objective_diff, color="tab:blue", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Objective difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("QHD objective - standard objective")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iterations, bus_dist, label="Bus state", color="tab:orange", linewidth=1.5)
    ax.plot(iterations, branch_dist, label="Branch flow", color="tab:green", linewidth=1.5)
    ax.plot(iterations, combined_dist, label="Combined", color="tab:red", linewidth=1.5)
    ax.set_title("Euclidean distance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 distance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(iterations, max_abs_h, label="max |h|", color="tab:purple", linewidth=1.5)
    ax.semilogy(iterations, l2_norm_h, label="||h||2", color="tab:brown", linewidth=1.5)
    ax.set_title("Constraint residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(iterations, load_supplied, color="tab:cyan", linewidth=1.5)
    ax.axhline(100.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Active load supplied")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_summary(best_records: dict[str, IterationRecord], out_path: Path) -> None:
    cases = list(best_records.keys())
    distances = [best_records[name].combined_distance for name in cases]
    objective_pct = [best_records[name].objective_pct_diff for name in cases]
    residuals = [best_records[name].max_abs_h for name in cases]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    axes[0].bar(cases, distances, color="tab:red")
    axes[0].set_title("Combined L2 distance")
    axes[0].set_ylabel("Distance")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(cases, objective_pct, color="tab:blue")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Objective gap")
    axes[1].set_ylabel("Percent")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(cases, residuals, color="tab:purple")
    axes[2].set_yscale("log")
    axes[2].set_title("max |h|")
    axes[2].set_ylabel("Residual")
    axes[2].grid(True, axis="y", alpha=0.3, which="both")

    fig.suptitle("Best-by-distance QHD/Simbi comparison with standard answers", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    summary_rows = []
    best_distance_records = {}

    for case_name, paths in CASES.items():
        standard = parse_standard_answer(paths["standard"])
        records = parse_qhd_log(paths["qhd_log"])
        if not records:
            raise ValueError(f"No records parsed from {paths['qhd_log']}")

        compute_distances(records, standard)
        best_distance = best_by_combined(records)
        best_residual = best_by_l2(records)
        best_distance_records[case_name] = best_distance

        plot_case(
            case_name,
            records,
            LOG_DIR / f"{case_name}_qhd_vs_standard_comparison.png",
        )

        for tag, record in (("best_distance", best_distance), ("best_residual", best_residual)):
            summary_rows.append(
                {
                    "case": case_name,
                    "selected_by": tag,
                    "iteration": record.iteration,
                    "standard_objective": f"{standard.objective:.12g}",
                    "qhd_objective": f"{record.objective:.12g}",
                    "objective_diff": f"{record.objective_diff:.12g}",
                    "objective_abs_diff": f"{record.objective_abs_diff:.12g}",
                    "objective_pct_diff": f"{record.objective_pct_diff:.12g}",
                    "bus_l2_distance": f"{record.bus_distance:.12g}",
                    "branch_l2_distance": f"{record.branch_distance:.12g}",
                    "combined_l2_distance": f"{record.combined_distance:.12g}",
                    "max_abs_h": f"{record.max_abs_h:.12g}",
                    "l2_norm_h": f"{record.l2_norm_h:.12g}",
                    "load_supplied_pct": "" if record.load_supplied is None else f"{record.load_supplied:.12g}",
                    "source_log": str(paths["qhd_log"].relative_to(ROOT)),
                    "standard_file": str(paths["standard"].relative_to(ROOT)),
                }
            )

    plot_summary(best_distance_records, LOG_DIR / "qhd_vs_standard_summary.png")

    csv_path = LOG_DIR / "qhd_standard_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {csv_path}")
    for row in summary_rows:
        print(
            row["case"],
            row["selected_by"],
            "iter",
            row["iteration"],
            "combined_l2",
            row["combined_l2_distance"],
            "obj_diff",
            row["objective_diff"],
            "obj_pct",
            row["objective_pct_diff"],
        )


if __name__ == "__main__":
    main()
