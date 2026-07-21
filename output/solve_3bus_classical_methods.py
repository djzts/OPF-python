#!/usr/bin/env python
# coding: utf-8

"""
Solve the repository's 3-bus ACOPF with classical continuous optimizers and
compare them with the coarse-only QCE log from 2026-06-22 05:08:44.

Outputs are written next to this script:
- 3bus_method_start_metrics.csv
- 3bus_method_best_summary.csv
- 3bus_objective_comparison.csv
- 3bus_qce_iteration_metrics.csv
- 3bus_solver_convergence_history.csv
- 3bus_classical_vs_qce_analysis.md
- 3bus_objective_gap_comparison.png
- 3bus_best_residual_comparison.png
- 3bus_runtime_comparison.png
- 3bus_convergence_residual.png
- 3bus_initial_sensitivity.png
"""

from __future__ import annotations

import csv
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

import sys

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_qhd_convergence_diagnostics import OPFCase, load_case  # noqa: E402


QCE_LOG = ROOT / "logs" / "QCE_result" / "Buses-3_06-22-2026_05-08-44.txt"
QCE_PRIOR_ANALYSIS = ROOT / "logs" / "QCE_result" / "Buses-3_06-22-2026_two_log_analysis.md"
QCE_SERVER = ROOT / "Sympy_OPF_LALM_mu_final_3bus_QCE_server.py"
EPS = 1e-12


@dataclass
class SolveResult:
    method: str
    start_name: str
    success: bool
    message: str
    x: np.ndarray
    runtime_s: float
    iterations: int
    function_evals: int | None
    source: str


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def fmt_float(value: float, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not math.isfinite(float(value)):
        return "nan"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def objective_gradient(case: OPFCase, x: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    vals = case.unpack(x)
    for i, gid in enumerate(case.gen_ids):
        coeff = case.gens[gid]
        grad[i] = 2.0 * float(coeff[5]) * vals["P_G"][i] + float(coeff[6])
    return grad


def constraint_jacobian(case: OPFCase, x: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = case.h(x)
    jac = np.empty((f0.size, x.size), dtype=float)
    for j in range(x.size):
        step = 1e-6 * max(1.0, abs(float(x[j])))
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        jac[:, j] = (case.h(xp) - case.h(xm)) / (2.0 * step)
    return jac


def clip_to_bounds(x: Iterable[float], bounds: Bounds, margin: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    lb = bounds.lb + margin
    ub = bounds.ub - margin
    return np.minimum(np.maximum(x, lb), ub)


def qce_x_from_log_line(line: str) -> np.ndarray | None:
    match = re.search(r"x=\[(.*)\]\s*$", line)
    if not match:
        return None
    return np.fromstring(match.group(1).replace(",", " "), sep=" ")


def parse_qce_log(case: OPFCase, log_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, object]] = []
    start_ts: datetime | None = None
    end_ts: datetime | None = None

    run_start = re.search(r"\[step ([^\]]+)\] run_start", text)
    run_end = re.search(r"\[step ([^\]]+)\] run_end", text)
    if run_start:
        start_ts = datetime.strptime(run_start.group(1), "%Y-%m-%d %H:%M:%S")
    if run_end:
        end_ts = datetime.strptime(run_end.group(1), "%Y-%m-%d %H:%M:%S")

    pattern = re.compile(
        r"^\[step (?P<ts>[^\]]+)\] evaluation "
        r"iter=(?P<iter>\d+), rank=(?P<rank>\d+), "
        r"objective=(?P<objective>[-+0-9.eE]+), "
        r"lalm_energy=(?P<energy>[-+0-9.eE]+), "
        r"l2_norm_h=(?P<l2>[-+0-9.eE]+), "
        r"max_abs_h=(?P<max>[-+0-9.eE]+), "
        r"step_norm=(?P<step>[-+0-9.eE]+), "
        r"feasible=(?P<feasible>True|False), x=\[(?P<x>.*)\]$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        x = np.fromstring(match.group("x").replace(",", " "), sep=" ")
        if x.size != case.n_variables:
            continue
        h = case.h(x)
        rows.append(
            {
                "method": "QCE coarse-only",
                "start_name": "QCE beam",
                "iteration": int(match.group("iter")),
                "rank": int(match.group("rank")),
                "objective": case.objective(x),
                "logged_objective": to_float(match.group("objective")),
                "lalm_energy": to_float(match.group("energy")),
                "l2_h": float(np.linalg.norm(h)),
                "logged_l2_h": to_float(match.group("l2")),
                "max_abs_h": float(np.max(np.abs(h))),
                "logged_max_abs_h": to_float(match.group("max")),
                "step_norm": to_float(match.group("step")),
                "feasible": match.group("feasible") == "True",
                "load_supplied_pct": case.active_load_supplied_percent(x),
                "x": x,
            }
        )

    metadata = {
        "start_ts": start_ts.isoformat(sep=" ") if start_ts else "",
        "end_ts": end_ts.isoformat(sep=" ") if end_ts else "",
        "runtime_s": (end_ts - start_ts).total_seconds() if start_ts and end_ts else float("nan"),
        "raw_path": str(log_path),
    }
    return rows, metadata


def base_metrics(case: OPFCase, x: Iterable[float], reference_x: np.ndarray, reference_obj: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    h = case.h(x)
    vals = case.unpack(x)
    ref_vals = case.unpack(reference_x)
    bus_l2 = float(np.linalg.norm(case.bus_state_vector(x) - case.bus_state_vector(reference_x)))
    branch_l2 = float(np.linalg.norm(case.branch_flow_vector(x) - case.branch_flow_vector(reference_x)))
    return {
        "objective": case.objective(x),
        "objective_gap_pct": 100.0 * (case.objective(x) / reference_obj - 1.0),
        "l2_h": float(np.linalg.norm(h)),
        "max_abs_h": float(np.max(np.abs(h))),
        "load_supplied_pct": case.active_load_supplied_percent(x),
        "bus_l2_to_reference": bus_l2,
        "branch_l2_to_reference": branch_l2,
        "combined_l2_to_reference": float(math.hypot(bus_l2, branch_l2)),
        "pg_l2_to_reference": float(np.linalg.norm(vals["P_G"] - ref_vals["P_G"])),
        "qg_l2_to_reference": float(np.linalg.norm(vals["Q_G"] - ref_vals["Q_G"])),
    }


def record_history(
    history: list[dict[str, object]],
    case: OPFCase,
    method: str,
    start_name: str,
    iteration: int,
    x: Iterable[float],
    stage: str = "",
) -> None:
    x = np.asarray(x, dtype=float)
    h = case.h(x)
    history.append(
        {
            "method": method,
            "start_name": start_name,
            "iteration": iteration,
            "stage": stage,
            "objective": case.objective(x),
            "l2_h": float(np.linalg.norm(h)),
            "max_abs_h": float(np.max(np.abs(h))),
            "load_supplied_pct": case.active_load_supplied_percent(x),
        }
    )


def solve_sqp(case: OPFCase, x0: np.ndarray, start_name: str, history: list[dict[str, object]]) -> SolveResult:
    bounds = case.bounds()
    zeros = np.zeros(case.h(x0).size)
    constraint = NonlinearConstraint(
        case.h,
        zeros,
        zeros,
        jac=lambda x: constraint_jacobian(case, x),
    )
    iter_count = 0

    def callback(xk: np.ndarray) -> None:
        nonlocal iter_count
        iter_count += 1
        record_history(history, case, "SQP (SLSQP)", start_name, iter_count, xk)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            case.objective,
            clip_to_bounds(x0, bounds),
            method="SLSQP",
            jac=lambda x: objective_gradient(case, x),
            bounds=bounds,
            constraints=[constraint],
            callback=callback,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
    runtime = time.perf_counter() - t0
    return SolveResult(
        method="SQP (SLSQP)",
        start_name=start_name,
        success=bool(res.success),
        message=str(res.message),
        x=clip_to_bounds(res.x, bounds),
        runtime_s=runtime,
        iterations=int(getattr(res, "nit", iter_count)),
        function_evals=int(getattr(res, "nfev", 0)) if hasattr(res, "nfev") else None,
        source="scipy.optimize.minimize(method='SLSQP')",
    )


def solve_interior_point(case: OPFCase, x0: np.ndarray, start_name: str, history: list[dict[str, object]]) -> SolveResult:
    bounds = case.bounds()
    zeros = np.zeros(case.h(x0).size)
    constraint = NonlinearConstraint(
        case.h,
        zeros,
        zeros,
        jac=lambda x: constraint_jacobian(case, x),
    )
    iter_count = 0

    def callback(xk: np.ndarray, state=None) -> bool:
        nonlocal iter_count
        iter_count += 1
        record_history(history, case, "Interior-point (trust-constr)", start_name, iter_count, xk)
        return False

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            case.objective,
            clip_to_bounds(x0, bounds),
            method="trust-constr",
            jac=lambda x: objective_gradient(case, x),
            bounds=bounds,
            constraints=[constraint],
            callback=callback,
            options={
                "gtol": 1e-10,
                "xtol": 1e-10,
                "barrier_tol": 1e-10,
                "maxiter": 400,
                "verbose": 0,
            },
        )
    runtime = time.perf_counter() - t0
    return SolveResult(
        method="Interior-point (trust-constr)",
        start_name=start_name,
        success=bool(res.success),
        message=str(res.message),
        x=clip_to_bounds(res.x, bounds),
        runtime_s=runtime,
        iterations=int(getattr(res, "nit", iter_count)),
        function_evals=int(getattr(res, "nfev", 0)) if hasattr(res, "nfev") else None,
        source="scipy.optimize.minimize(method='trust-constr')",
    )


def solve_truncated_newton(
    case: OPFCase,
    x0: np.ndarray,
    start_name: str,
    history: list[dict[str, object]],
) -> SolveResult:
    bounds = case.bounds()
    x = clip_to_bounds(x0, bounds)
    total_nfev = 0
    total_iter = 0
    success = False
    message = ""
    t0 = time.perf_counter()
    stages = [1e2, 1e4, 1e6, 1e8]

    for rho in stages:
        stage_name = f"rho={rho:.0e}"

        def penalty_fun(z: np.ndarray) -> float:
            h = case.h(z)
            return case.objective(z) + 0.5 * rho * float(h @ h)

        def penalty_grad(z: np.ndarray) -> np.ndarray:
            h = case.h(z)
            return objective_gradient(case, z) + rho * constraint_jacobian(case, z).T @ h

        def callback(xk: np.ndarray) -> None:
            nonlocal total_iter
            total_iter += 1
            record_history(history, case, "Truncated Newton (TNC penalty)", start_name, total_iter, xk, stage_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                penalty_fun,
                x,
                method="TNC",
                jac=penalty_grad,
                bounds=list(zip(bounds.lb, bounds.ub)),
                callback=callback,
                options={
                    "maxfun": 2200,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "xtol": 1e-10,
                    "disp": False,
                },
            )
        x = clip_to_bounds(res.x, bounds)
        total_nfev += int(getattr(res, "nfev", 0))
        success = bool(res.success) or float(np.max(np.abs(case.h(x)))) <= 1e-6
        message = f"{stage_name}: {res.message}"

    runtime = time.perf_counter() - t0
    return SolveResult(
        method="Truncated Newton (TNC penalty)",
        start_name=start_name,
        success=success,
        message=message,
        x=x,
        runtime_s=runtime,
        iterations=total_iter,
        function_evals=total_nfev,
        source="scipy.optimize.minimize(method='TNC') on staged equality-penalty objective",
    )


def active_kkt_components(case: OPFCase, bounds: Bounds, x: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_residual = np.asarray([x[0] - bounds.lb[0]], dtype=float)
    c = np.concatenate([case.h(x), active_residual])
    jac_h = constraint_jacobian(case, x)
    active_jac = np.zeros((1, x.size), dtype=float)
    active_jac[0, 0] = 1.0
    jac = np.vstack([jac_h, active_jac])
    grad_lag = objective_gradient(case, x) + jac.T @ lam
    return c, jac, grad_lag


def lagrangian_hessian_fd(case: OPFCase, bounds: Bounds, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
    n = x.size
    hess = np.empty((n, n), dtype=float)

    def grad_lag_at(z: np.ndarray) -> np.ndarray:
        return active_kkt_components(case, bounds, z, lam)[2]

    for j in range(n):
        step = 1e-5 * max(1.0, abs(float(x[j])))
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        xp = clip_to_bounds(xp, bounds, margin=1e-10)
        xm = clip_to_bounds(xm, bounds, margin=1e-10)
        hess[:, j] = (grad_lag_at(xp) - grad_lag_at(xm)) / max(float(np.linalg.norm(xp - xm)), EPS)
    return 0.5 * (hess + hess.T)


def solve_newton_raphson(
    case: OPFCase,
    x0: np.ndarray,
    start_name: str,
    history: list[dict[str, object]],
    max_iter: int = 16,
) -> SolveResult:
    bounds = case.bounds()
    x = clip_to_bounds(x0, bounds, margin=1e-10)
    n = x.size
    m = case.h(x).size + 1
    lam = np.zeros(m, dtype=float)
    try:
        _c0, jac0, _grad_lag0 = active_kkt_components(case, bounds, x, lam)
        lam = np.linalg.lstsq(jac0.T, -objective_gradient(case, x), rcond=None)[0]
    except np.linalg.LinAlgError:
        lam = np.zeros(m, dtype=float)
    success = False
    message = "maximum iterations reached"
    t0 = time.perf_counter()

    def kkt_residual(z: np.ndarray, mult: np.ndarray) -> np.ndarray:
        c, _jac, grad_lag = active_kkt_components(case, bounds, z, mult)
        return np.concatenate([grad_lag, c])

    for it in range(1, max_iter + 1):
        c, jac, grad_lag = active_kkt_components(case, bounds, x, lam)
        residual = np.concatenate([grad_lag, c])
        residual_norm = float(np.linalg.norm(residual))
        record_history(history, case, "Newton-Raphson active-set KKT", start_name, it, x)
        if residual_norm <= 1e-8 and float(np.max(np.abs(case.h(x)))) <= 1e-8:
            success = True
            message = "KKT residual tolerance reached"
            break

        hess = lagrangian_hessian_fd(case, bounds, x, lam)
        kkt = np.block(
            [
                [hess, jac.T],
                [jac, np.zeros((m, m), dtype=float)],
            ]
        )
        try:
            step = np.linalg.solve(kkt, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(kkt, -residual, rcond=None)[0]
        dx = step[:n]
        dlam = step[n:]

        best_norm = residual_norm
        accepted = False
        for scale in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
            trial_x = clip_to_bounds(x + scale * dx, bounds, margin=1e-10)
            trial_lam = lam + scale * dlam
            trial_norm = float(np.linalg.norm(kkt_residual(trial_x, trial_lam)))
            if trial_norm < best_norm:
                x = trial_x
                lam = trial_lam
                accepted = True
                break
        if not accepted:
            message = "line search failed to reduce KKT residual"
            break

    runtime = time.perf_counter() - t0
    final_residual = float(np.linalg.norm(kkt_residual(x, lam)))
    if not success and final_residual <= 1e-7 and float(np.max(np.abs(case.h(x)))) <= 1e-8:
        success = True
        message = "KKT residual tolerance reached"
    return SolveResult(
        method="Newton-Raphson active-set KKT",
        start_name=start_name,
        success=success,
        message=message,
        x=clip_to_bounds(x, bounds),
        runtime_s=runtime,
        iterations=it,
        function_evals=None,
        source="custom damped Newton solve of active-bound KKT system",
    )


def threshold_crossings(history_rows: list[dict[str, object]], method: str, start_name: str) -> dict[str, int | str]:
    rows = [r for r in history_rows if r["method"] == method and r["start_name"] == start_name]
    out: dict[str, int | str] = {}
    for threshold in [1e-2, 1e-4, 1e-6]:
        key = f"first_l2_le_{threshold:g}"
        crossing = [int(r["iteration"]) for r in rows if float(r["l2_h"]) <= threshold]
        out[key] = min(crossing) if crossing else ""
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, np.generic):
                    value = value.item()
                cleaned[key] = value
            writer.writerow(cleaned)


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(label for _key, label in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
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


def plot_outputs(
    best_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    history: list[dict[str, object]],
    objective_rows: list[dict[str, object]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    colors = {
        "Newton-Raphson active-set KKT": "#4C78A8",
        "Truncated Newton (TNC penalty)": "#F58518",
        "Interior-point (trust-constr)": "#54A24B",
        "SQP (SLSQP)": "#B279A2",
        "QCE coarse-only": "#8C8C8C",
    }

    objective_labels = [str(r.get("plot_label", r["label"])) for r in objective_rows]
    objective_values = [float(r["objective"]) for r in objective_rows]
    objective_gaps = [float(r["objective_gap_pct"]) for r in objective_rows]
    objective_colors = [colors.get(str(r["method"]), "#4C78A8") for r in objective_rows]
    reference_obj = next(
        (float(r["objective"]) for r in objective_rows if str(r["method"]) == "SQP (SLSQP)"),
        min(objective_values),
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    ax1.bar(range(len(objective_rows)), objective_values, color=objective_colors)
    ax1.axhline(reference_obj, color="#333333", linewidth=1.2, linestyle="--", label="Reference objective")
    ymin = min(objective_values + [reference_obj])
    ymax = max(objective_values + [reference_obj])
    pad = max((ymax - ymin) * 0.18, 1e-5)
    ax1.set_ylim(ymin - pad, ymax + pad)
    ax1.set_ylabel("Objective")
    ax1.set_title("3-bus objective comparison")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.bar(range(len(objective_rows)), objective_gaps, color=objective_colors)
    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.set_ylabel("Objective gap (%)")
    ax2.set_xticks(range(len(objective_rows)))
    ax2.set_xticklabels(objective_labels, rotation=35, ha="right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3bus_objective_gap_comparison.png", dpi=220)
    plt.close(fig)

    labels = [r["method"] for r in best_rows]
    residuals = [max(float(r["max_abs_h"]), 1e-14) for r in best_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, residuals, color=[colors.get(label, "#4C78A8") for label in labels])
    ax.set_yscale("log")
    ax.set_ylabel("Max equality residual")
    ax.set_title("3-bus best constraint violation by method")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3bus_best_residual_comparison.png", dpi=220)
    plt.close(fig)

    runtimes = [max(float(r["runtime_s"]), 1e-6) for r in best_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, runtimes, color=[colors.get(label, "#4C78A8") for label in labels])
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (s, log scale)")
    ax.set_title("3-bus runtime comparison")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3bus_runtime_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method in labels:
        method_rows = [r for r in history if r["method"] == method]
        if not method_rows:
            continue
        starts = sorted({str(r["start_name"]) for r in method_rows})
        preferred = "flat_start" if "flat_start" in starts else starts[0]
        series = [r for r in method_rows if r["start_name"] == preferred]
        series = sorted(series, key=lambda r: int(r["iteration"]))
        ax.plot(
            [int(r["iteration"]) for r in series],
            [max(float(r["l2_h"]), 1e-14) for r in series],
            label=f"{method} ({preferred})",
            color=colors.get(method),
            linewidth=1.8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 equality residual")
    ax.set_title("3-bus residual convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3bus_convergence_residual.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    sens_rows = [r for r in all_rows if r["method"] != "QCE coarse-only" and r["start_name"] != "sqp_solution_warm_start"]
    labels2 = [f"{r['method']}\n{r['start_name']}" for r in sens_rows]
    vals = [max(float(r["max_abs_h"]), 1e-14) for r in sens_rows]
    ax.bar(range(len(sens_rows)), vals, color=[colors.get(str(r["method"]), "#4C78A8") for r in sens_rows])
    ax.set_xticks(range(len(sens_rows)))
    ax.set_xticklabels(labels2, rotation=70, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Final max equality residual")
    ax.set_title("Sensitivity to initial conditions")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3bus_initial_sensitivity.png", dpi=220)
    plt.close(fig)


def build_report(
    best_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    objective_rows: list[dict[str, object]],
    qce_rows: list[dict[str, object]],
    qce_meta: dict[str, object],
    reference: dict[str, object],
) -> str:
    solver_best = [r for r in best_rows if r["method"] != "QCE coarse-only"]
    qce_best = next(r for r in best_rows if r["method"] == "QCE coarse-only")
    sqp = next(r for r in best_rows if r["method"] == "SQP (SLSQP)")
    ip = next(r for r in best_rows if r["method"] == "Interior-point (trust-constr)")
    tnc = next(r for r in best_rows if r["method"] == "Truncated Newton (TNC penalty)")
    nr = next(r for r in best_rows if r["method"] == "Newton-Raphson active-set KKT")

    qce_final = qce_rows[-1]
    qce_min_l2 = min(qce_rows, key=lambda r: float(r["l2_h"]))
    qce_min_max = min(qce_rows, key=lambda r: float(r["max_abs_h"]))
    qce_first_1e2 = min((r for r in qce_rows if float(r["l2_h"]) <= 1e-2), key=lambda r: int(r["iteration"]), default=None)
    qce_first_1e3 = min((r for r in qce_rows if float(r["l2_h"]) <= 1e-3), key=lambda r: int(r["iteration"]), default=None)
    qce_tail_step_zero = 0
    for row in reversed(qce_rows):
        if abs(float(row["step_norm"])) <= 1e-14:
            qce_tail_step_zero += 1
        else:
            break

    method_table = markdown_table(
        best_rows,
        [
            ("method", "Method"),
            ("start_name", "Best start"),
            ("success", "Success"),
            ("objective", "Objective"),
            ("objective_gap_pct", "Obj gap %"),
            ("l2_h", "L2 h"),
            ("max_abs_h", "Max abs h"),
            ("runtime_s", "Runtime s"),
            ("iterations", "Iters"),
            ("combined_l2_to_reference", "State/flow dist"),
        ],
    )

    sensitivity_table = markdown_table(
        all_rows,
        [
            ("method", "Method"),
            ("start_name", "Start"),
            ("success", "Success"),
            ("objective_gap_pct", "Obj gap %"),
            ("max_abs_h", "Max abs h"),
            ("runtime_s", "Runtime s"),
            ("iterations", "Iters"),
        ],
    )

    objective_table = markdown_table(
        objective_rows,
        [
            ("label", "Selection"),
            ("objective", "Objective"),
            ("objective_gap_pct", "Obj gap %"),
            ("l2_h", "L2 h"),
            ("max_abs_h", "Max abs h"),
            ("feasible_for_obj_compare", "Feasible/near-feasible"),
        ],
    )

    qce_table = markdown_table(
        [
            {
                "selection": "Best L2 residual",
                "iteration": int(qce_min_l2["iteration"]),
                "objective": float(qce_min_l2["objective"]),
                "l2_h": float(qce_min_l2["l2_h"]),
                "max_abs_h": float(qce_min_l2["max_abs_h"]),
                "load_supplied_pct": float(qce_min_l2["load_supplied_pct"]),
            },
            {
                "selection": "Best max residual",
                "iteration": int(qce_min_max["iteration"]),
                "objective": float(qce_min_max["objective"]),
                "l2_h": float(qce_min_max["l2_h"]),
                "max_abs_h": float(qce_min_max["max_abs_h"]),
                "load_supplied_pct": float(qce_min_max["load_supplied_pct"]),
            },
            {
                "selection": "Final main-loop record",
                "iteration": int(qce_final["iteration"]),
                "objective": float(qce_final["objective"]),
                "l2_h": float(qce_final["l2_h"]),
                "max_abs_h": float(qce_final["max_abs_h"]),
                "load_supplied_pct": float(qce_final["load_supplied_pct"]),
            },
        ],
        [
            ("selection", "QCE selection"),
            ("iteration", "Iteration"),
            ("objective", "Objective"),
            ("l2_h", "L2 h"),
            ("max_abs_h", "Max abs h"),
            ("load_supplied_pct", "Load %"),
        ],
    )

    nr_rows = [r for r in all_rows if r["method"] == "Newton-Raphson active-set KKT"]
    nr_success = sum(1 for r in nr_rows if str(r["success"]) == "True")

    lines = [
        "# 3-bus classical solver comparison against QCE coarse-only",
        "",
        "## Technical summary",
        "",
        f"- The continuous constrained optimizers solve the same 3-bus ACOPF to near machine precision: SQP reaches max |h| = {fmt_float(float(sqp['max_abs_h']))}, interior-point reaches {fmt_float(float(ip['max_abs_h']))}, and the staged truncated-Newton penalty reaches {fmt_float(float(tnc['max_abs_h']))}.",
        f"- The requested QCE run is coarse-only (`refine_method='none'`, `qhd_refine=False`) and stops at max |h| = {fmt_float(float(qce_best['max_abs_h']))} for its best residual record. That is still about {fmt_float(float(qce_best['max_abs_h']) / 1e-5)}x above the configured 1e-5 feasibility tolerance.",
        f"- Newton-Raphson active-set KKT succeeds for {nr_success} of {len(nr_rows)} tested starts when the correct active generator bound is supplied. This is useful for local KKT solving, but it is less general than SQP or interior-point because the active set is assumed rather than discovered.",
        f"- The QCE log ran from {qce_meta['start_ts']} to {qce_meta['end_ts']} ({fmt_float(float(qce_meta['runtime_s']))} s), while the local SciPy continuous solves completed in seconds or less on this machine. Runtime is therefore directionally useful but not a hardware-normalized benchmark.",
        "",
        "## Method results",
        "",
        method_table,
        "",
        "## Objective comparison",
        "",
        objective_table,
        "",
        f"Using the SQP solution as the reference objective ({fmt_float(float(sqp['objective']), digits=12)}), the continuous constrained methods are essentially objective-equivalent. TNC is slightly higher at +{fmt_float(float(tnc['objective_gap_pct']))}%, while the QCE best-residual record is +{fmt_float(float(qce_best['objective_gap_pct']))}% and the QCE final record is higher still. The QCE `best_iteration_by_objective` summary in the raw log has objective 0, but it is infeasible and is therefore excluded from this objective-quality comparison.",
        "",
        "## QCE baseline behavior",
        "",
        qce_table,
        "",
        f"The 05-08-44 QCE trajectory has {len(qce_rows)} main-loop records. It first reaches L2 h <= 1e-2 at iteration {int(qce_first_1e2['iteration']) if qce_first_1e2 else 'not reached'} and L2 h <= 1e-3 at iteration {int(qce_first_1e3['iteration']) if qce_first_1e3 else 'not reached'}. The final {qce_tail_step_zero} records have zero logged step norm, which matches the plateau behavior described in the existing two-log analysis file.",
        "",
        "The two-log analysis establishes the reference ACOPF objective as 0.531710655973 and notes that the earlier 05-08-43 QCE run also stayed infeasible, with best L2 h around 7.25e-4 and a 100-iteration plateau. The 05-08-44 run improves the minimum residual slightly, but the parsed final point has a large reactive-power split: Q_G is approximately [-0.409, 0.395] rather than the reference dispatch near [-0.0136, -0.00639]. This explains why a small equality residual can still correspond to a noticeably different reactive operating branch.",
        "",
        "## Sensitivity to initial conditions",
        "",
        sensitivity_table,
        "",
        "SQP is robust from the flat and QCE starts but can fail from a larger arbitrary perturbation. Interior-point is the most robust across the tested starts. The truncated-Newton penalty method is also fairly robust, but it trades constraint accuracy for a longer staged penalty solve and depends on large penalty weights. The active-set Newton-Raphson implementation converges on these starts after the correct active bound is supplied, but it remains sensitive to that modeling assumption and should not be treated as a generic inequality-constrained solver without active-set detection.",
        "",
        "## Convergence interpretation",
        "",
        "QCE makes broad global search progress early but becomes quantized by the coarse grid, bound shrinkage, alpha floor, and rho cap. The continuous methods do not have that quantization barrier: SQP and interior-point use local constraint models to drive the equality residual to numerical precision, and the penalty TNC method reduces violation as rho increases. The price is that these continuous methods rely on smooth local derivatives and do not provide the same beam-search exploration that QCE provides.",
        "",
        "## Implementation notes from the QCE server",
        "",
        f"- Source file: `{QCE_SERVER.name}`.",
        "- The server is configured as coarse-only QCE: `refine_method='none'` and `qhd_refine=False`.",
        "- The beam width is `n_linearization_points=10`, with adaptive `alpha` and `rho` capped at `rho_max=256`.",
        "- `simbi_agents` and `simbi_max_steps` are annotated as integers but assigned with `/4`, producing floats. Python accepts this, but casting to `int` would make the solver configuration less ambiguous.",
        "",
        "## Recommended next steps",
        "",
        "1. Use SQP or interior-point as the deterministic continuous reference for 3-bus result validation.",
        "2. Use QCE best-residual points as candidate warm starts, then add a continuous local refinement stage if the goal is feasibility below 1e-5.",
        "3. Treat Newton-Raphson KKT as an active-set method: it works here after assuming generator 1 is at Pmin, but a general implementation should detect or update the active set.",
        "4. For future QCE runs, log both coarse and post-refine metrics when refinement is enabled, so the comparison is made after each full coarse+refine round.",
        "",
        "## Output files",
        "",
        "- `3bus_method_start_metrics.csv`: method-by-start metrics.",
        "- `3bus_method_best_summary.csv`: best result by method plus QCE baseline.",
        "- `3bus_objective_comparison.csv`: objective and objective-gap comparison source data.",
        "- `3bus_qce_iteration_metrics.csv`: parsed QCE iteration metrics from the 05-08-44 log.",
        "- `3bus_solver_convergence_history.csv`: source data for convergence plots.",
        "- `3bus_objective_gap_comparison.png`, `3bus_best_residual_comparison.png`, `3bus_runtime_comparison.png`, `3bus_convergence_residual.png`, `3bus_initial_sensitivity.png`: generated figures.",
        "",
        "## Sources",
        "",
        f"- `{QCE_LOG.relative_to(ROOT)}`",
        f"- `{QCE_PRIOR_ANALYSIS.relative_to(ROOT)}`",
        f"- `{QCE_SERVER.relative_to(ROOT)}`",
        "- `plot_qhd_convergence_diagnostics.py` for the shared 3-bus case equations and metric helpers.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    case = load_case(3, ROOT)
    bounds = case.bounds()
    x_flat = clip_to_bounds(case.build_initial_x0(), bounds)

    qce_rows, qce_meta = parse_qce_log(case, QCE_LOG)
    if not qce_rows:
        raise RuntimeError(f"No QCE evaluation rows parsed from {QCE_LOG}")
    qce_best_l2 = min(qce_rows, key=lambda r: float(r["l2_h"]))
    qce_final = qce_rows[-1]

    rng = np.random.default_rng(1234)
    perturb = rng.normal(loc=0.0, scale=0.04, size=x_flat.size)
    x_random = clip_to_bounds(x_flat + perturb, bounds)

    starts = {
        "flat_start": x_flat,
        "qce_best_l2_iter": np.asarray(qce_best_l2["x"], dtype=float),
        "qce_final_iter": np.asarray(qce_final["x"], dtype=float),
        "random_perturbed_flat": x_random,
    }

    history: list[dict[str, object]] = []
    results: list[SolveResult] = []

    for start_name, x0 in starts.items():
        results.append(solve_sqp(case, x0, start_name, history))
    reference_result = min(
        (r for r in results if r.success),
        key=lambda r: (float(np.max(np.abs(case.h(r.x)))), case.objective(r.x)),
    )
    reference_x = reference_result.x
    reference_obj = case.objective(reference_x)

    starts_with_warm = dict(starts)
    starts_with_warm["sqp_solution_warm_start"] = reference_x

    for start_name, x0 in starts.items():
        results.append(solve_interior_point(case, x0, start_name, history))
    for start_name, x0 in starts.items():
        results.append(solve_truncated_newton(case, x0, start_name, history))
    for start_name, x0 in starts_with_warm.items():
        results.append(solve_newton_raphson(case, x0, start_name, history))

    metric_rows: list[dict[str, object]] = []
    for result in results:
        metrics = base_metrics(case, result.x, reference_x, reference_obj)
        row = {
            "method": result.method,
            "start_name": result.start_name,
            "success": result.success,
            "message": result.message,
            "runtime_s": result.runtime_s,
            "iterations": result.iterations,
            "function_evals": result.function_evals if result.function_evals is not None else "",
            "source": result.source,
        }
        row.update(metrics)
        row.update(threshold_crossings(history, result.method, result.start_name))
        metric_rows.append(row)

    qce_metric_rows: list[dict[str, object]] = []
    for row in qce_rows:
        metrics = base_metrics(case, np.asarray(row["x"], dtype=float), reference_x, reference_obj)
        qce_metric_rows.append(
            {
                "method": "QCE coarse-only",
                "start_name": "QCE beam",
                "iteration": row["iteration"],
                "rank": row["rank"],
                "objective": metrics["objective"],
                "objective_gap_pct": metrics["objective_gap_pct"],
                "l2_h": metrics["l2_h"],
                "max_abs_h": metrics["max_abs_h"],
                "load_supplied_pct": metrics["load_supplied_pct"],
                "bus_l2_to_reference": metrics["bus_l2_to_reference"],
                "branch_l2_to_reference": metrics["branch_l2_to_reference"],
                "combined_l2_to_reference": metrics["combined_l2_to_reference"],
                "pg_l2_to_reference": metrics["pg_l2_to_reference"],
                "qg_l2_to_reference": metrics["qg_l2_to_reference"],
                "logged_l2_h": row["logged_l2_h"],
                "logged_max_abs_h": row["logged_max_abs_h"],
                "step_norm": row["step_norm"],
                "feasible": row["feasible"],
            }
        )

    best_qce = min(qce_metric_rows, key=lambda r: float(r["max_abs_h"]))
    qce_best_summary = {
        "method": "QCE coarse-only",
        "start_name": f"best_max_iter_{best_qce['iteration']}",
        "success": False,
        "message": "coarse-only log never reached configured feasibility tolerance",
        "runtime_s": qce_meta["runtime_s"],
        "iterations": len(qce_rows),
        "function_evals": "",
        "source": "parsed QCE log",
        **{k: best_qce[k] for k in [
            "objective",
            "objective_gap_pct",
            "l2_h",
            "max_abs_h",
            "load_supplied_pct",
            "bus_l2_to_reference",
            "branch_l2_to_reference",
            "combined_l2_to_reference",
            "pg_l2_to_reference",
            "qg_l2_to_reference",
        ]},
        "first_l2_le_0.01": min((int(r["iteration"]) for r in qce_metric_rows if float(r["l2_h"]) <= 1e-2), default=""),
        "first_l2_le_0.0001": min((int(r["iteration"]) for r in qce_metric_rows if float(r["l2_h"]) <= 1e-4), default=""),
        "first_l2_le_1e-06": min((int(r["iteration"]) for r in qce_metric_rows if float(r["l2_h"]) <= 1e-6), default=""),
    }

    best_rows: list[dict[str, object]] = []
    for method in sorted({r["method"] for r in metric_rows}):
        method_rows = [r for r in metric_rows if r["method"] == method]
        best = min(method_rows, key=lambda r: (float(r["max_abs_h"]), abs(float(r["objective_gap_pct"]))))
        best_rows.append(best)
    best_rows.append(qce_best_summary)
    method_order = [
        "SQP (SLSQP)",
        "Interior-point (trust-constr)",
        "Truncated Newton (TNC penalty)",
        "Newton-Raphson active-set KKT",
        "QCE coarse-only",
    ]
    best_rows.sort(key=lambda r: method_order.index(str(r["method"])) if str(r["method"]) in method_order else 99)

    qce_final_metric = qce_metric_rows[-1]
    objective_rows: list[dict[str, object]] = []
    for row in best_rows:
        if row["method"] == "QCE coarse-only":
            label = f"QCE best residual iter {best_qce['iteration']}"
            plot_label = f"QCE best residual\niter {best_qce['iteration']}"
            iteration = best_qce["iteration"]
        else:
            label = str(row["method"])
            plot_label = str(row["method"]).replace(" (", "\n(")
            iteration = ""
        objective_rows.append(
            {
                "method": row["method"],
                "label": label,
                "plot_label": plot_label,
                "selection": row["start_name"],
                "iteration": iteration,
                "objective": row["objective"],
                "objective_gap_pct": row["objective_gap_pct"],
                "l2_h": row["l2_h"],
                "max_abs_h": row["max_abs_h"],
                "load_supplied_pct": row["load_supplied_pct"],
                "feasible_for_obj_compare": bool(row["success"]) and float(row["max_abs_h"]) <= 1e-5,
            }
        )
    objective_rows.append(
        {
            "method": "QCE coarse-only",
            "label": f"QCE final iter {qce_final_metric['iteration']}",
            "plot_label": f"QCE final\niter {qce_final_metric['iteration']}",
            "selection": "final_main_loop_record",
            "iteration": qce_final_metric["iteration"],
            "objective": qce_final_metric["objective"],
            "objective_gap_pct": qce_final_metric["objective_gap_pct"],
            "l2_h": qce_final_metric["l2_h"],
            "max_abs_h": qce_final_metric["max_abs_h"],
            "load_supplied_pct": qce_final_metric["load_supplied_pct"],
            "feasible_for_obj_compare": False,
        }
    )

    history_for_csv = [
        {
            "method": r["method"],
            "start_name": r["start_name"],
            "iteration": r["iteration"],
            "stage": r.get("stage", ""),
            "objective": r["objective"],
            "l2_h": r["l2_h"],
            "max_abs_h": r["max_abs_h"],
            "load_supplied_pct": r["load_supplied_pct"],
        }
        for r in history
    ]
    for row in qce_metric_rows:
        history_for_csv.append(
            {
                "method": "QCE coarse-only",
                "start_name": "QCE beam",
                "iteration": row["iteration"],
                "stage": "",
                "objective": row["objective"],
                "l2_h": row["l2_h"],
                "max_abs_h": row["max_abs_h"],
                "load_supplied_pct": row["load_supplied_pct"],
            }
        )

    metric_fields = [
        "method",
        "start_name",
        "success",
        "message",
        "objective",
        "objective_gap_pct",
        "l2_h",
        "max_abs_h",
        "load_supplied_pct",
        "runtime_s",
        "iterations",
        "function_evals",
        "bus_l2_to_reference",
        "branch_l2_to_reference",
        "combined_l2_to_reference",
        "pg_l2_to_reference",
        "qg_l2_to_reference",
        "first_l2_le_0.01",
        "first_l2_le_0.0001",
        "first_l2_le_1e-06",
        "source",
    ]
    qce_fields = [
        "method",
        "start_name",
        "iteration",
        "rank",
        "objective",
        "objective_gap_pct",
        "l2_h",
        "max_abs_h",
        "load_supplied_pct",
        "bus_l2_to_reference",
        "branch_l2_to_reference",
        "combined_l2_to_reference",
        "pg_l2_to_reference",
        "qg_l2_to_reference",
        "logged_l2_h",
        "logged_max_abs_h",
        "step_norm",
        "feasible",
    ]
    objective_fields = [
        "method",
        "label",
        "selection",
        "iteration",
        "objective",
        "objective_gap_pct",
        "l2_h",
        "max_abs_h",
        "load_supplied_pct",
        "feasible_for_obj_compare",
    ]
    history_fields = ["method", "start_name", "iteration", "stage", "objective", "l2_h", "max_abs_h", "load_supplied_pct"]

    write_csv(OUT_DIR / "3bus_method_start_metrics.csv", metric_rows, metric_fields)
    write_csv(OUT_DIR / "3bus_method_best_summary.csv", best_rows, metric_fields)
    write_csv(OUT_DIR / "3bus_objective_comparison.csv", objective_rows, objective_fields)
    write_csv(OUT_DIR / "3bus_qce_iteration_metrics.csv", qce_metric_rows, qce_fields)
    write_csv(OUT_DIR / "3bus_solver_convergence_history.csv", history_for_csv, history_fields)

    plot_outputs(best_rows, metric_rows, history_for_csv, objective_rows)

    report = build_report(
        best_rows,
        metric_rows,
        objective_rows,
        qce_rows,
        qce_meta,
        {
            "x": reference_x,
            "objective": reference_obj,
            "source_method": reference_result.method,
            "source_start": reference_result.start_name,
        },
    )
    (OUT_DIR / "3bus_classical_vs_qce_analysis.md").write_text(report, encoding="utf-8")

    print("Wrote 3-bus classical solver comparison outputs to", OUT_DIR)
    print("Reference objective:", f"{reference_obj:.12f}")
    print("Best summary:")
    for row in best_rows:
        print(
            f"  {row['method']}: objective={float(row['objective']):.12f}, "
            f"max_abs_h={float(row['max_abs_h']):.3e}, runtime_s={float(row['runtime_s']):.3f}, "
            f"start={row['start_name']}"
        )


if __name__ == "__main__":
    main()
