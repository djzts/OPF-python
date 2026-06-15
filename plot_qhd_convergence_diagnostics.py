#!/usr/bin/env python
# coding: utf-8

"""
Build convergence diagnostics for OPF/QHD/Simbi ALM logs.

This script is intentionally self-contained so it can be run even in an
environment without Pyomo/Gurobi. It parses the QHD log, reconstructs the
decision vectors in the same order used by SympyACOPFModel, solves a reference
ACOPF with SciPy/SLSQP, writes iteration metrics to CSV, and creates a 2x2
diagnostic plot.

Existing solver locations used by this script:
- Iteration loop: Sympy_OPF_LALM_mu_final_3bus.py::run_linear_alm(),
  inside `for k in range(config.max_outer)`.
- Objective values: `coarse_objective_value` and `objective_value`.
- Constraint residuals: `coarse_h_val = h_func(x_coarse)` and
  `h_val = h_func(x_new)`.
- Voltage variables: the decision vector is
  [P_G, Q_G, V_R, V_I, V_sq, P_ij, Q_ij, S_tot_sq].
- Branch flow variables: P_ij/Q_ij are directed arcs ordered by
  (line,+1), (line,-1).
- Active load supplied: printed in the log summary and recomputed here as
  100 * (sum(P_G) - sum(line active losses)) / sum(P_D).

For online recording inside the ALM loop, call `record_iteration_metrics(...)`
after both coarse and refined solutions have been evaluated, right after
`objective_value = evaluate_objective(model, x_new)` and before the existing
`metric_history.append(...)` block.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize


DEFAULT_LOG = Path(__file__).resolve().parent / "logs" / "Buses-3_06-13-2026_01-17-16.txt"
DEFAULT_LEGEND_FONTSIZE = 14
DEFAULT_AXIS_FONTSIZE = 14


@dataclass
class OPFCase:
    n_bus: int
    sbase: float
    buses: dict[int, list[float]]
    lines: dict[int, list[float]]
    gens: dict[int, list[float]]

    def __post_init__(self) -> None:
        self.bus_ids = sorted(self.buses)
        self.line_ids = sorted(self.lines)
        self.gen_ids = sorted(self.gens)
        self.n_buses = len(self.bus_ids)
        self.n_lines = len(self.line_ids)
        self.n_gens = len(self.gen_ids)
        self.bus_index = {bid: i for i, bid in enumerate(self.bus_ids)}
        self.gen_index = {gid: i for i, gid in enumerate(self.gen_ids)}
        self.gen_indices_by_bus = {bid: [] for bid in self.bus_ids}
        for gid in self.gen_ids:
            bus_id = int(self.gens[gid][0])
            self.gen_indices_by_bus[bus_id].append(self.gen_index[gid])

        self.arc_ids: list[tuple[int, int]] = []
        self.arc_collection: list[tuple[int, int]] = []
        self.arc_to_line: list[int] = []
        for lid in self.line_ids:
            from_bus, to_bus = int(self.lines[lid][0]), int(self.lines[lid][1])
            i = self.bus_index[from_bus]
            j = self.bus_index[to_bus]
            self.arc_ids.append((lid, +1))
            self.arc_collection.append((i, j))
            self.arc_to_line.append(lid)
            self.arc_ids.append((lid, -1))
            self.arc_collection.append((j, i))
            self.arc_to_line.append(lid)
        self.arc_index = {arc_id: i for i, arc_id in enumerate(self.arc_ids)}
        self.n_arcs = len(self.arc_ids)

        self._build_network_matrices()
        self.P_D = np.array([self.buses[bid][6] for bid in self.bus_ids], dtype=float)
        self.Q_D = np.array([self.buses[bid][7] for bid in self.bus_ids], dtype=float)

    @property
    def n_variables(self) -> int:
        return 2 * self.n_gens + 3 * self.n_buses + 3 * self.n_arcs

    def _build_network_matrices(self) -> None:
        nb = self.n_buses
        nl = self.n_lines
        ybus = np.zeros((nb, nb), dtype=np.complex128)
        g_series = np.zeros(nl)
        b_series = np.zeros(nl)
        branch_b = np.zeros((nb, nb))
        for ell, lid in enumerate(self.line_ids):
            from_bus, to_bus, r, x, bsh, tap, _rate = self.lines[lid]
            i = self.bus_index[int(from_bus)]
            j = self.bus_index[int(to_bus)]
            z = complex(r, x)
            y = 1.0 / z if abs(z) > 0.0 else 1e6 + 0j
            bs = 1j * bsh
            a = tap if tap != 0 else 1.0
            ybus[i, i] += y / (a**2)
            ybus[j, j] += y / (a**2)
            ybus[i, i] += bs
            ybus[j, j] += bs
            ybus[i, j] -= y / a
            ybus[j, i] -= y / a
            g_series[ell] = y.real
            b_series[ell] = y.imag
            branch_b[i, j] = bs.imag
            branch_b[j, i] = bs.imag
        self.Ybus = ybus
        self.G_mat = ybus.real
        self.B_mat = ybus.imag
        self.g_series = g_series
        self.b_series = b_series
        self.branch_b = branch_b

    def unpack(self, x: Iterable[float]) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_variables:
            raise ValueError(f"Expected vector length {self.n_variables}, got {x.size}")
        idx = 0
        ng, nb, na = self.n_gens, self.n_buses, self.n_arcs
        out = {}
        out["P_G"] = x[idx:idx + ng]; idx += ng
        out["Q_G"] = x[idx:idx + ng]; idx += ng
        out["V_R"] = x[idx:idx + nb]; idx += nb
        out["V_I"] = x[idx:idx + nb]; idx += nb
        out["V_sq"] = x[idx:idx + nb]; idx += nb
        out["P_ij"] = x[idx:idx + na]; idx += na
        out["Q_ij"] = x[idx:idx + na]; idx += na
        out["S_tot_sq"] = x[idx:idx + na]
        return out

    def objective(self, x: Iterable[float]) -> float:
        vals = self.unpack(x)
        obj = 0.0
        for gi, gid in enumerate(self.gen_ids):
            gdata = self.gens[gid]
            obj += float(gdata[5]) * vals["P_G"][gi] ** 2
            obj += float(gdata[6]) * vals["P_G"][gi]
            obj += float(gdata[7])
        return float(obj)

    def branch_flow_expr(self, arc_idx: int, V_R: np.ndarray, V_I: np.ndarray) -> tuple[float, float]:
        i, j = self.arc_collection[arc_idx]
        Gij = self.G_mat[i, j]
        Bij = self.B_mat[i, j]
        bij = self.branch_b[i, j]
        vi_sq = V_R[i] ** 2 + V_I[i] ** 2
        w_real = V_R[i] * V_R[j] + V_I[i] * V_I[j]
        w_imag = V_I[i] * V_R[j] - V_R[i] * V_I[j]
        p_expr = (-Gij) * vi_sq + Gij * w_real + Bij * w_imag
        q_expr = (Bij - bij) * vi_sq - Bij * w_real + Gij * w_imag
        return float(p_expr), float(q_expr)

    def h(self, x: Iterable[float], ref_bus_id: int | None = None) -> np.ndarray:
        vals = self.unpack(x)
        P_G = vals["P_G"]
        Q_G = vals["Q_G"]
        V_R = vals["V_R"]
        V_I = vals["V_I"]
        V_sq = vals["V_sq"]
        P_ij = vals["P_ij"]
        Q_ij = vals["Q_ij"]
        S_tot_sq = vals["S_tot_sq"]

        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        outgoing = {i: [] for i in range(self.n_buses)}
        for a, (i, _j) in enumerate(self.arc_collection):
            outgoing[i].append(a)

        residuals = []
        for i, bus_id in enumerate(self.bus_ids):
            pg_sum = sum(P_G[gi] for gi in self.gen_indices_by_bus[bus_id])
            residuals.append(pg_sum - self.P_D[i] - sum(P_ij[a] for a in outgoing[i]))

        for i, bus_id in enumerate(self.bus_ids):
            qg_sum = sum(Q_G[gi] for gi in self.gen_indices_by_bus[bus_id])
            residuals.append(qg_sum - self.Q_D[i] - sum(Q_ij[a] for a in outgoing[i]))

        for a in range(self.n_arcs):
            p_expr, q_expr = self.branch_flow_expr(a, V_R, V_I)
            residuals.append(P_ij[a] - p_expr)
            residuals.append(Q_ij[a] - q_expr)

        for i in range(self.n_buses):
            residuals.append(V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2))

        for a in range(self.n_arcs):
            residuals.append(S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2))

        residuals.append(V_I[ref_idx])
        residuals.append(V_R[ref_idx] - 1.0)
        return np.asarray(residuals, dtype=float)

    def bounds(self) -> Bounds:
        lb: list[float] = []
        ub: list[float] = []
        for gid in self.gen_ids:
            lb.append(float(self.gens[gid][1]))
            ub.append(float(self.gens[gid][2]))
        for gid in self.gen_ids:
            lb.append(float(self.gens[gid][3]))
            ub.append(float(self.gens[gid][4]))
        lb.extend([-1.1] * self.n_buses)
        ub.extend([1.1] * self.n_buses)
        lb.extend([-1.1] * self.n_buses)
        ub.extend([1.1] * self.n_buses)
        lb.extend([0.9**2] * self.n_buses)
        ub.extend([1.1**2] * self.n_buses)
        for lid in self.arc_to_line:
            rate = float(self.lines[lid][6])
            lb.append(-rate)
            ub.append(rate)
        for lid in self.arc_to_line:
            rate = float(self.lines[lid][6])
            lb.append(-rate)
            ub.append(rate)
        for lid in self.arc_to_line:
            rate = float(self.lines[lid][6])
            lb.append(0.0)
            ub.append(rate**2)
        return Bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))

    def build_initial_x0(self) -> np.ndarray:
        total_pd = float(np.sum(self.P_D))
        pmin = np.array([self.gens[gid][1] for gid in self.gen_ids], dtype=float)
        pmax = np.array([self.gens[gid][2] for gid in self.gen_ids], dtype=float)
        if total_pd > 1e-10 and np.sum(pmax) > 1e-10:
            P_G = total_pd * pmax / np.sum(pmax)
            P_G = np.clip(P_G, pmin, pmax)
        else:
            P_G = pmin.copy()

        total_qd = float(np.sum(self.Q_D))
        qmin = np.array([self.gens[gid][3] for gid in self.gen_ids], dtype=float)
        qmax = np.array([self.gens[gid][4] for gid in self.gen_ids], dtype=float)
        if abs(total_qd) > 1e-10 and np.sum(np.maximum(P_G, 0.0)) > 1e-10:
            Q_G = total_qd * np.maximum(P_G, 0.0) / np.sum(np.maximum(P_G, 0.0))
            Q_G = np.clip(Q_G, qmin, qmax)
        else:
            Q_G = np.clip(np.zeros(self.n_gens), qmin, qmax)

        V_R = np.zeros(self.n_buses)
        V_I = np.zeros(self.n_buses)
        for bid in self.bus_ids:
            bdata = self.buses[bid]
            i = self.bus_index[bid]
            vm = float(bdata[2])
            va = math.radians(float(bdata[3]))
            V_R[i] = vm * math.cos(va)
            V_I[i] = vm * math.sin(va)
        V_sq = V_R**2 + V_I**2
        P_ij = np.zeros(self.n_arcs)
        Q_ij = np.zeros(self.n_arcs)
        S_tot_sq = np.zeros(self.n_arcs)
        return np.concatenate([P_G, Q_G, V_R, V_I, V_sq, P_ij, Q_ij, S_tot_sq])

    def bus_state_vector(self, x: Iterable[float]) -> np.ndarray:
        vals = self.unpack(x)
        return np.concatenate([vals["V_R"], vals["V_I"]])

    def branch_flow_vector(self, x: Iterable[float]) -> np.ndarray:
        vals = self.unpack(x)
        pieces = []
        for lid in self.line_ids:
            a_fwd = self.arc_index[(lid, +1)]
            a_rev = self.arc_index[(lid, -1)]
            pieces.extend([
                vals["P_ij"][a_fwd],
                vals["Q_ij"][a_fwd],
                vals["P_ij"][a_rev],
                vals["Q_ij"][a_rev],
            ])
        return np.asarray(pieces, dtype=float)

    def active_load_supplied_percent(self, x: Iterable[float]) -> float:
        vals = self.unpack(x)
        total_load = float(np.sum(self.P_D))
        if abs(total_load) <= 1e-12:
            return float("nan")
        total_pg = float(np.sum(vals["P_G"]))
        total_loss = 0.0
        for lid in self.line_ids:
            a_fwd = self.arc_index[(lid, +1)]
            a_rev = self.arc_index[(lid, -1)]
            total_loss += float(vals["P_ij"][a_fwd] + vals["P_ij"][a_rev])
        return 100.0 * (total_pg - total_loss) / total_load


def load_case(n_bus: int, base_dir: Path) -> OPFCase:
    if n_bus == 2:
        sbase = 10.0
        buses = {
            1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.3, 0.1],
        }
        lines = {1: [1, 2, 0.0452, 0.1852, 0.0204, 1.0, 30.0 / sbase]}
        gens = {1: [1, 0.0 / sbase, 20.0 / sbase, -20.0 / sbase, 100.0 / sbase, 0.00375, 2.0, 0.0]}
        return OPFCase(n_bus, sbase, buses, lines, gens)

    if n_bus == 3:
        sbase = 10.0
        buses = {
            1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            3: [3, 2, 1.00, 0.0, 0.0, 0.0, 0.3, 0.1],
        }
        lines = {
            1: [1, 2, 0.0192, 0.0575, 0.0264, 1.0, 30.0 / sbase],
            2: [1, 3, 0.0452, 0.1852, 0.0204, 1.0, 30.0 / sbase],
            3: [2, 3, 0.0570, 0.1737, 0.0184, 1.0, 30.0 / sbase],
        }
        gens = {
            1: [1, 0.0 / sbase, 20.0 / sbase, -20.0 / sbase, 100.0 / sbase, 0.00375, 2.0, 0.0],
            2: [2, 0.0 / sbase, 20.0 / sbase, -20.0 / sbase, 100.0 / sbase, 0.0175, 1.75, 0.0],
        }
        return OPFCase(n_bus, sbase, buses, lines, gens)

    json_path = base_dir / f"case{n_bus}_custom.json"
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    sbase = float(data["Sbase"])
    buses = {int(k.replace("k", "")): v for k, v in data["buses"].items()}
    lines = {int(k.replace("k", "")): v for k, v in data["lines"].items()}
    gens = {int(k.replace("k", "")): v for k, v in data["gens"].items()}
    return OPFCase(n_bus, sbase, buses, lines, gens)


def parse_float_after_colon(line: str) -> float:
    return float(line.split(":", 1)[1].strip().replace("%", ""))


def parse_log(log_path: Path, case: OPFCase) -> list[dict]:
    records: list[dict] = []
    current_iteration: int | None = None
    record: dict | None = None
    mode: str | None = None

    def finish_record() -> None:
        nonlocal record
        if record is not None and record.get("note") in {
            "coarse_solution_before_refine",
            "refined_solution_TNC_orig",
            "refined_solution_ipopt_orig",
            "refined_solution_GurobiALM",
            "refined_solution_GurobiOrig",
            "refined_solution_none",
        }:
            records.append(record)
        record = None

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
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
        elif mode == "summary":
            if line.startswith("Total Load Supplied:"):
                record["log_active_load_supplied_percent"] = parse_float_after_colon(line)

    finish_record()
    for rec in records:
        rec["x"] = record_to_decision_vector(rec, case)
        rec["answer_type"] = "coarse" if rec["note"] == "coarse_solution_before_refine" else "refined"
    return records


def record_to_decision_vector(record: dict, case: OPFCase) -> np.ndarray:
    buses = record["buses"]
    branches = record["branches"]
    P_G = []
    Q_G = []
    for gid in case.gen_ids:
        bus_id = int(case.gens[gid][0])
        P_G.append(buses[bus_id]["Pg"])
        Q_G.append(buses[bus_id]["Qg"])

    V_R = np.array([buses[bid]["V_R"] for bid in case.bus_ids], dtype=float)
    V_I = np.array([buses[bid]["V_I"] for bid in case.bus_ids], dtype=float)
    V_sq = np.array([buses[bid]["Vmag"] ** 2 for bid in case.bus_ids], dtype=float)

    P_ij = []
    Q_ij = []
    S_tot_sq = []
    for lid in case.line_ids:
        br = branches[lid]
        P_ij.extend([br["Pik"], br["Pki"]])
        Q_ij.extend([br["Qik"], br["Qki"]])
        S_tot_sq.extend([br["Sik_sq"], br["Ski_sq"]])

    return np.concatenate([
        np.asarray(P_G, dtype=float),
        np.asarray(Q_G, dtype=float),
        V_R,
        V_I,
        V_sq,
        np.asarray(P_ij, dtype=float),
        np.asarray(Q_ij, dtype=float),
        np.asarray(S_tot_sq, dtype=float),
    ])


def clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(x, dtype=float), bounds.lb), bounds.ub)


def solve_standard_acopf(case: OPFCase, warm_starts: list[np.ndarray]) -> dict:
    bounds = case.bounds()
    h_dim = case.h(case.build_initial_x0()).size
    eq_constraint = NonlinearConstraint(case.h, np.zeros(h_dim), np.zeros(h_dim))
    candidates = [case.build_initial_x0(), *warm_starts]
    best = None
    for i, x0_raw in enumerate(candidates):
        x0 = clip_to_bounds(x0_raw, bounds)
        result = minimize(
            case.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[eq_constraint],
            options={
                "ftol": 1e-12,
                "maxiter": 3000,
                "disp": False,
            },
        )
        max_h = float(np.max(np.abs(case.h(result.x)))) if result.x is not None else float("inf")
        candidate = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(case.objective(result.x)),
            "max_abs_h": max_h,
            "x": np.asarray(result.x, dtype=float),
            "start_index": i,
        }
        if best is None:
            best = candidate
            continue
        if candidate["max_abs_h"] < best["max_abs_h"] - 1e-10:
            best = candidate
        elif candidate["max_abs_h"] <= max(best["max_abs_h"], 1e-10) and candidate["objective"] < best["objective"]:
            best = candidate

    if best is None:
        raise RuntimeError("No standard ACOPF candidate was attempted.")
    return best


def record_iteration_metrics(record: dict, standard_solution: dict, case: OPFCase) -> dict:
    x = np.asarray(record["x"], dtype=float)
    x_ref = np.asarray(standard_solution["x"], dtype=float)
    h = case.h(x)
    bus_dist = float(np.linalg.norm(case.bus_state_vector(x) - case.bus_state_vector(x_ref)))
    branch_dist = float(np.linalg.norm(case.branch_flow_vector(x) - case.branch_flow_vector(x_ref)))
    combined_dist = float(np.linalg.norm(np.concatenate([
        case.bus_state_vector(x) - case.bus_state_vector(x_ref),
        case.branch_flow_vector(x) - case.branch_flow_vector(x_ref),
    ])))
    qhd_objective = float(case.objective(x))
    standard_objective = float(standard_solution["objective"])
    return {
        "iteration": int(record["iteration"]),
        "answer_type": record["answer_type"],
        "qhd_objective": qhd_objective,
        "standard_objective": standard_objective,
        "objective_difference": qhd_objective - standard_objective,
        "bus_state_l2_distance": bus_dist,
        "branch_flow_l2_distance": branch_dist,
        "combined_l2_distance": combined_dist,
        "max_constraint_residual": float(np.max(np.abs(h))),
        "l2_constraint_residual": float(np.linalg.norm(h)),
        "active_load_supplied_percent": float(case.active_load_supplied_percent(x)),
        "log_objective": float(record["objective"]),
        "log_max_constraint_residual": float(record.get("log_max_abs_h", np.nan)),
        "log_l2_constraint_residual": float(record.get("log_l2_norm_h", np.nan)),
        "log_active_load_supplied_percent": float(record.get("log_active_load_supplied_percent", np.nan)),
    }


def write_history_csv(history: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "answer_type",
        "qhd_objective",
        "standard_objective",
        "objective_difference",
        "bus_state_l2_distance",
        "branch_flow_l2_distance",
        "combined_l2_distance",
        "max_constraint_residual",
        "l2_constraint_residual",
        "active_load_supplied_percent",
        "log_objective",
        "log_max_constraint_residual",
        "log_l2_constraint_residual",
        "log_active_load_supplied_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def series(history: list[dict], answer_type: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in history if row["answer_type"] == answer_type]
    rows.sort(key=lambda item: item["iteration"])
    return (
        np.array([row["iteration"] for row in rows], dtype=int),
        np.array([row[key] for row in rows], dtype=float),
    )


def positive_for_log(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.maximum(y, 1e-14)


PLOT_COLORS = {
    "coarse": "tab:orange",
    "refined": "tab:blue",
    "coarse_alt": "tab:red",
    "refined_alt": "tab:green",
}


def align_yaxis_values(ax_left, left_value: float, ax_right, right_value: float) -> None:
    left_min, left_max = ax_left.get_ylim()
    if left_max == left_min:
        return

    position = (left_value - left_min) / (left_max - left_min)
    position = min(max(position, 1e-6), 1.0 - 1e-6)

    right_min, right_max = ax_right.get_ylim()
    lower_span = (right_value - right_min) / position
    upper_span = (right_max - right_value) / (1.0 - position)
    span = max(right_max - right_min, lower_span, upper_span, 1e-12)
    ax_right.set_ylim(
        right_value - position * span,
        right_value + (1.0 - position) * span,
    )


def draw_objective_panel(ax, history: list[dict]) -> None:
    ax2 = ax.twinx()
    for answer_type, color in [("coarse", PLOT_COLORS["coarse"]), ("refined", PLOT_COLORS["refined"])]:
        x, y = series(history, answer_type, "objective_difference")
        ax.plot(x, y, color=color, lw=1.6, label=f"{answer_type} obj diff")
        x, supplied = series(history, answer_type, "active_load_supplied_percent")
        ax2.plot(x, supplied, color=color, lw=1.2, ls="--", alpha=0.75, label=f"{answer_type} load supplied")
    ax.axhline(0.0, color="0.25", lw=1.0, ls=":")
    align_yaxis_values(ax, 0.0, ax2, 100.0)
    ax.set_title("Objective difference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("QHD objective - standard objective")
    ax2.set_ylabel("Active load supplied (%)")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")


def draw_distance_panel(ax, history: list[dict]) -> None:
    distance_specs = [
        ("bus_state_l2_distance", "bus", "-"),
        ("branch_flow_l2_distance", "branch", "--"),
        ("combined_l2_distance", "combined", "-."),
    ]
    for answer_type, color in [("coarse", PLOT_COLORS["coarse"]), ("refined", PLOT_COLORS["refined"])]:
        for key, label, ls in distance_specs:
            x, y = series(history, answer_type, key)
            ax.plot(x, positive_for_log(y), color=color, ls=ls, lw=1.35, label=f"{answer_type} {label}")
    ax.set_yscale("log")
    ax.set_title("Euclidean distance")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 distance to standard")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(ncol=2)


def draw_residual_panel(ax, history: list[dict]) -> None:
    for answer_type, color in [("coarse", PLOT_COLORS["coarse"]), ("refined", PLOT_COLORS["refined"])]:
        x, max_h = series(history, answer_type, "max_constraint_residual")
        ax.plot(x, positive_for_log(max_h), color=color, lw=1.45, label=f"{answer_type} max |h|")
        x, l2_h = series(history, answer_type, "l2_constraint_residual")
        ax.plot(x, positive_for_log(l2_h), color=color, ls="--", lw=1.45, label=f"{answer_type} ||h||2")
    ax.set_yscale("log")
    ax.set_title("Constraint residual")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend(ncol=2)


def draw_refinement_contraction_panel(ax, history: list[dict]) -> None:
    xc, coarse_combined = series(history, "coarse", "combined_l2_distance")
    xr, refined_combined = series(history, "refined", "combined_l2_distance")
    _xc, coarse_resid = series(history, "coarse", "l2_constraint_residual")
    _xr, refined_resid = series(history, "refined", "l2_constraint_residual")
    common_n = min(len(xc), len(xr), len(coarse_combined), len(refined_combined))
    if common_n:
        x_common = xr[:common_n]
        dist_ratio = refined_combined[:common_n] / np.maximum(coarse_combined[:common_n], 1e-14)
        resid_ratio = refined_resid[:common_n] / np.maximum(coarse_resid[:common_n], 1e-14)
        ax.plot(x_common, positive_for_log(dist_ratio), color="tab:purple", lw=1.7, label="refined/coarse distance")
        ax.plot(x_common, positive_for_log(resid_ratio), color="tab:brown", lw=1.7, ls="--", label="refined/coarse residual")
    ax.axhline(1.0, color="0.25", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_title("Refinement contraction")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Ratio, lower is better")
    ax.grid(True, which="both", ls=":", alpha=0.55)
    ax.legend()


def save_single_panel_plots(history: list[dict], case_name: str, output_dir: Path) -> list[Path]:
    stem = case_name.replace(" ", "_").replace("/", "_")
    panels = [
        ("objective_difference", draw_objective_panel),
        ("euclidean_distance", draw_distance_panel),
        ("constraint_residual", draw_residual_panel),
        ("refinement_contraction", draw_refinement_contraction_panel),
    ]
    output_paths: list[Path] = []
    for suffix, draw_func in panels:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        draw_func(ax, history)
        ax.margins(x=0.01)
        png_path = output_dir / f"{stem}_panel_{suffix}.png"
        pdf_path = output_dir / f"{stem}_panel_{suffix}.pdf"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        output_paths.extend([png_path, pdf_path])
    return output_paths


def plot_convergence_diagnostics(
    history: list[dict],
    standard_solution: dict,
    case_name: str,
    output_dir: Path,
    legend_fontsize: float = DEFAULT_LEGEND_FONTSIZE,
    axis_fontsize: float = DEFAULT_AXIS_FONTSIZE,
) -> tuple[Path, Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": axis_fontsize,
        "xtick.labelsize": axis_fontsize,
        "ytick.labelsize": axis_fontsize,
        "legend.fontsize": legend_fontsize,
        "figure.titlesize": 15,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    #fig.suptitle(f"{case_name}: QHD coarse and refined answer", fontweight="bold")

    draw_objective_panel(axes[0, 0], history)
    draw_distance_panel(axes[0, 1], history)
    draw_residual_panel(axes[1, 0], history)
    draw_refinement_contraction_panel(axes[1, 1], history)

    for ax in axes.ravel():
        ax.margins(x=0.01)

    stem = case_name.replace(" ", "_").replace("/", "_")
    png_path = output_dir / f"{stem}_qhd_convergence_diagnostics.png"
    pdf_path = output_dir / f"{stem}_qhd_convergence_diagnostics.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    panel_paths = save_single_panel_plots(history, case_name, output_dir)
    return png_path, pdf_path, panel_paths


def parse_case_name_from_log(log_path: Path) -> tuple[int, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Number of buses:\s*(\d+)", text)
    if not match:
        raise ValueError(f"Could not find 'Number of buses' in {log_path}")
    n_bus = int(match.group(1))
    return n_bus, f"{n_bus}-bus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create QHD/Simbi/ALM convergence diagnostic plots.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="QHD ACOPF log text file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV/PNG/PDF outputs")
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=DEFAULT_LEGEND_FONTSIZE,
        help="Legend font size for all generated plots",
    )
    parser.add_argument(
        "--axis-fontsize",
        type=float,
        default=DEFAULT_AXIS_FONTSIZE,
        help="Font size for all x/y axis labels and tick labels",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    base_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve() if args.output_dir else log_path.parent / f"{log_path.stem}_diagnostics"

    n_bus, case_name = parse_case_name_from_log(log_path)
    case = load_case(n_bus, base_dir)
    records = parse_log(log_path, case)
    if not records:
        raise RuntimeError(f"No coarse/refined QHD records parsed from {log_path}")

    refined_records = [rec for rec in records if rec["answer_type"] == "refined"]
    refined_records.sort(key=lambda rec: rec.get("log_l2_norm_h", float("inf")))
    warm_starts = [rec["x"] for rec in refined_records[:8]]
    if refined_records:
        warm_starts.append(refined_records[-1]["x"])

    standard_solution = solve_standard_acopf(case, warm_starts=warm_starts)
    history = [record_iteration_metrics(rec, standard_solution, case) for rec in records]
    history.sort(key=lambda row: (row["iteration"], row["answer_type"] != "coarse"))

    csv_path = output_dir / f"{log_path.stem}_diagnostic_metrics.csv"
    write_history_csv(history, csv_path)
    png_path, pdf_path, panel_paths = plot_convergence_diagnostics(
        history,
        standard_solution,
        case_name,
        output_dir,
        legend_fontsize=args.legend_fontsize,
        axis_fontsize=args.axis_fontsize,
    )

    print(f"Parsed records: {len(records)} ({len(records)//2} iterations with coarse/refined pairs)")
    print(
        "Standard solve: "
        f"success={standard_solution['success']}, "
        f"status={standard_solution['status']}, "
        f"objective={standard_solution['objective']:.12g}, "
        f"max|h|={standard_solution['max_abs_h']:.3e}, "
        f"start_index={standard_solution['start_index']}"
    )
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("Single-panel plots:")
    for path in panel_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
