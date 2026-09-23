"""Continuous-only SciPy/TNC benchmark for the audited width-1 SB runs.

The benchmark uses the same rectangular ACOPF equations, logged initial centers,
and reference-bus bounds. TNC minimizes a staged equality-penalty objective; it
does not call QHD, simulated bifurcation, or the logged TNC refinement results.
"""

from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, minimize

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot_qhd_convergence_diagnostics import load_case  # noqa: E402

OUT = Path(__file__).resolve().parent / "tnc_continuous_baseline"
OUT.mkdir(exist_ok=True)
LOGS = {
    2: "Buses-2_07-07-2026_21-37-55.txt",
    3: "Buses-3_07-07-2026_21-37-55.txt",
    5: "Buses-5_07-07-2026_21-37-59.txt",
    9: "Buses-9_07-07-2026_21-38-05.txt",
    14: "Buses-14_07-07-2026_21-38-10.txt",
}
PENALTIES = (128.0, 512.0, 2048.0, 8192.0, 32768.0, 131072.0, 524288.0, 2097152.0)
TOL = 1e-5


def logged_start(path: Path, n: int) -> np.ndarray:
    match = re.search(r"qhd_start iter=0, parent=1/1, center=\[(.*?)\]", path.read_text(errors="replace"))
    if match is None:
        raise ValueError(f"No logged first center in {path}")
    x = np.fromstring(match.group(1).replace(",", " "), sep=" ")
    if x.size != n:
        raise ValueError(f"Logged center has {x.size} variables; expected {n}")
    return x


def logged_best(path: Path) -> dict[str, float]:
    pattern = re.compile(
        r"evaluation iter=(\d+), rank=\d+, objective=([-+\deE.]+), .*?"
        r"l2_norm_h=([-+\deE.]+), max_abs_h=([-+\deE.]+)"
    )
    rows = [
        (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for m in pattern.finditer(path.read_text(errors="replace"))
    ]
    if not rows:
        raise ValueError(f"No evaluations in {path}")
    iteration, objective, l2, max_abs = min(rows, key=lambda row: (row[3], row[2]))
    return {"iter": iteration, "objective": objective, "l2": l2, "max_abs": max_abs}


def reference_bounds(case) -> Bounds:
    bounds = case.bounds()
    lb, ub = bounds.lb.copy(), bounds.ub.copy()
    vr = 2 * case.n_gens
    vi = vr + case.n_buses
    lb[vr], ub[vr] = 0.999999, 1.000001
    lb[vi], ub[vi] = -0.000001, 0.000001
    return Bounds(lb, ub)


def objective_grad(case, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    for i, gid in enumerate(case.gen_ids):
        coeff = case.gens[gid]
        grad[i] = 2.0 * float(coeff[5]) * x[i] + float(coeff[6])
    return grad


def residual_jac(case, x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian in the exact residual order of ``OPFCase.h``."""
    ng, nb, na = case.n_gens, case.n_buses, case.n_arcs
    p_g, q_g = 0, ng
    vr, vi, vsq = 2 * ng, 2 * ng + nb, 2 * ng + 2 * nb
    p_arc, q_arc, ssq = 2 * ng + 3 * nb, 2 * ng + 3 * nb + na, 2 * ng + 3 * nb + 2 * na
    jac = np.zeros((3 * nb + 3 * na + 2, x.size))

    for i, bid in enumerate(case.bus_ids):
        for gi in case.gen_indices_by_bus[bid]:
            jac[i, p_g + gi] = 1.0
            jac[nb + i, q_g + gi] = 1.0
    for a, (i, j) in enumerate(case.arc_collection):
        jac[i, p_arc + a] = -1.0
        jac[nb + i, q_arc + a] = -1.0
        g, b, charge = case.G_mat[i, j], case.B_mat[i, j], case.branch_b[i, j]
        ri, ii, rj, ij = x[vr + i], x[vi + i], x[vr + j], x[vi + j]
        rp, rq = 2 * nb + 2 * a, 2 * nb + 2 * a + 1
        jac[rp, p_arc + a] = 1.0
        jac[rq, q_arc + a] = 1.0
        jac[rp, vr + i] = -(-2 * g * ri + g * rj - b * ij)
        jac[rp, vi + i] = -(-2 * g * ii + g * ij + b * rj)
        jac[rp, vr + j] = -(g * ri + b * ii)
        jac[rp, vi + j] = -(g * ii - b * ri)
        jac[rq, vr + i] = -(2 * (b - charge) * ri - b * rj - g * ij)
        jac[rq, vi + i] = -(2 * (b - charge) * ii - b * ij + g * rj)
        jac[rq, vr + j] = -(-b * ri + g * ii)
        jac[rq, vi + j] = -(-b * ii - g * ri)
    voltage_row = 2 * nb + 2 * na
    for i in range(nb):
        jac[voltage_row + i, vr + i] = -2 * x[vr + i]
        jac[voltage_row + i, vi + i] = -2 * x[vi + i]
        jac[voltage_row + i, vsq + i] = 1.0
    apparent_row = voltage_row + nb
    for a in range(na):
        jac[apparent_row + a, p_arc + a] = -2 * x[p_arc + a]
        jac[apparent_row + a, q_arc + a] = -2 * x[q_arc + a]
        jac[apparent_row + a, ssq + a] = 1.0
    jac[-2, vi] = 1.0
    jac[-1, vr] = 1.0
    return jac


def metrics(case, x: np.ndarray) -> tuple[float, float, float]:
    h = case.h(x)
    return case.objective(x), float(np.linalg.norm(h)), float(np.max(np.abs(h)))


def run_case(bus: int) -> tuple[dict, list[dict]]:
    case = load_case(bus, ROOT)
    source = ROOT / "logs" / "single_beam" / LOGS[bus]
    bounds = reference_bounds(case)
    x = np.clip(logged_start(source, case.n_variables), bounds.lb, bounds.ub)
    np.save(OUT / f"case{bus}_start.npy", x)
    start_objective, start_l2, start_max = metrics(case, x)
    best_x, best = x.copy(), (start_max, start_l2)
    qhd = logged_best(source)
    t0 = time.perf_counter()
    history: list[dict] = []
    nfev = 0
    total_iter = 0
    messages = []

    for penalty in PENALTIES:
        def fun_and_grad(z: np.ndarray):
            h = case.h(z)
            value = case.objective(z) + 0.5 * penalty * float(h @ h)
            grad = objective_grad(case, z) + penalty * residual_jac(case, z).T @ h
            return value, grad

        def callback(z: np.ndarray):
            nonlocal total_iter, best_x, best
            total_iter += 1
            objective, l2, max_abs = metrics(case, z)
            elapsed = time.perf_counter() - t0
            history.append({
                "bus": bus, "stage_rho": penalty, "iteration": total_iter,
                "seconds": elapsed, "objective": objective, "l2": l2, "max_abs": max_abs,
            })
            if (max_abs, l2) < best:
                best, best_x = (max_abs, l2), z.copy()

        result = minimize(
            fun_and_grad, x, method="TNC", jac=True,
            bounds=list(zip(bounds.lb, bounds.ub)), callback=callback,
            options={"maxfun": 1800, "ftol": 1e-12, "gtol": 1e-8, "xtol": 1e-10},
        )
        x = np.clip(result.x, bounds.lb, bounds.ub)
        nfev += int(result.nfev)
        objective, l2, max_abs = metrics(case, x)
        if (max_abs, l2) < best:
            best, best_x = (max_abs, l2), x.copy()
        messages.append(f"rho={penalty:g}: status={result.status}, nfev={result.nfev}, max={max_abs:.3e}")
        print(f"{bus}-bus {messages[-1]}", flush=True)
        if best[0] <= TOL:
            break

    elapsed = time.perf_counter() - t0
    best_obj, best_l2, best_max = metrics(case, best_x)
    np.save(OUT / f"case{bus}_tnc_best.npy", best_x)
    end_obj, end_l2, end_max = metrics(case, x)
    row = {
        "bus": bus, "n_variables": case.n_variables, "scipy_version": scipy.__version__,
        "start_objective": start_objective, "start_l2": start_l2, "start_max_abs": start_max,
        "tnc_best_objective": best_obj, "tnc_best_l2": best_l2, "tnc_best_max_abs": best_max,
        "tnc_final_objective": end_obj, "tnc_final_l2": end_l2, "tnc_final_max_abs": end_max,
        "tnc_seconds": elapsed, "tnc_iterations": total_iter, "tnc_nfev": nfev,
        "tnc_feasible_1e5": best_max <= TOL, "tnc_messages": " | ".join(messages),
        "sb_best_iteration": qhd["iter"], "sb_best_objective": qhd["objective"],
        "sb_best_l2": qhd["l2"], "sb_best_max_abs": qhd["max_abs"],
        "sb_log": str(source),
    }
    return row, history


def main() -> None:
    rows, histories = [], []
    for bus in LOGS:
        row, history = run_case(bus)
        rows.append(row)
        histories.extend(history)
        with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if histories:
            with (OUT / "history.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(histories[0]))
                writer.writeheader()
                writer.writerows(histories)
    print(f"Wrote {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
