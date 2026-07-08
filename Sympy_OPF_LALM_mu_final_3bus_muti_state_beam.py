Sympy_OPF_LALM_mu_final_3bus.py#!/usr/bin/env python
# coding: utf-8

import json
import time
from dataclasses import dataclass
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from pyomo.environ import *
from qhdopt import QHD
from scipy.optimize import Bounds, minimize

from Sympy_OPF_LALM_class import (
    PrintQHDACOPFResults,
    SympyACOPFModel,
    extract_qhd_solution_vector,
    initialize_qhd_acopf_log,
    solve_with_gurobi_from_sympy,
)


@dataclass
class SolverConfig:
    n_bus: int = 3
    max_outer: int = 2000
    tol: float = 1e-5
    option: int = 1  # 1: QHD, 2: Gurobi
    qhd_solver: str = "simbi"  # simbi / openjij / gurobi
    refine_method: str = "TNC_orig"  # none / ipopt_orig / TNC_orig / GurobiALM / GurobiOrig
    rho: float = 128.0
    alpha: float = 5.0
    mu_prox: float = 1 #2e-2
    alpha_mode: str = "adaptive"  # adaptive / fixed
    alpha_min: float = 1e-2
    alpha_max: float = 8.0
    rho_min: float = 1e-2
    rho_max: float = 512.0
    plateau_window: int = 4
    worsen_window: int = 2
    stable_window: int = 4
    improve_tol: float = 0.005
    worsen_tol: float = 0.03
    qhd_refine: bool = True
    simbi_resolution: int = 25
    simbi_shots: int = 128
    simbi_agents: int = 4096
    simbi_max_steps: int = 42000
    simbi_seed: int | None = 42
    simbi_best_only: bool = False
    simbi_ballistic: bool = False
    simbi_heated: bool | None = None
    early_stop_patience: int = 100
    tnc_maxfun: int | None = None
    ipopt_max_iter: int = 300
    gurobi_time_limit: float | None = 60.0
    gurobi_threads: int = 0
    gurobi_log_to_console: bool = False
    coarse_beam_search: bool = True
    n_linearization_points: int = 10
    candidate_distinct_atol: float = 1e-9
    beam_refine_candidates: bool = True
    beam_refine_keep: int = 10
    coarse_repeat_limit: int = 5
    bound_shrink_factor: float = 0.5
    bound_shrink_min_factor: float = 2.0 ** -10
    bound_shrink_start_iter: int = 2
    bound_shrink_require_residual_improvement: bool = True
    coarse_repeat_atol: float = 1e-10
    return_best_solution: bool = True
    print_to_console: bool = True
    show_plot: bool = True
    log_folder: str = "logs"
    max_runtime_seconds: float | None = 23 * 60 * 60


REFINE_METHODS = {"none", "ipopt_orig", "TNC_orig", "GurobiALM", "GurobiOrig"}
REFINE_METHOD_ALIASES = {method.lower(): method for method in REFINE_METHODS}
REF_VR_BOUND = (0.999999, 1.000001)
REF_VI_BOUND = (-0.000001, 0.000001)


def load_matpower_json(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sbase = float(data["Sbase"])
    buses = {int(k.replace("k", "")): v for k, v in data["buses"].items()}
    lines = {int(k.replace("k", "")): v for k, v in data["lines"].items()}
    gens = {int(k.replace("k", "")): v for k, v in data["gens"].items()}
    return sbase, buses, lines, gens


def build_model(n_bus: int) -> SympyACOPFModel:
    if n_bus == 2:
        sbase = 10.0
        buses = {
            1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.3, 0.1],
        }
        lines = {
            1: [1, 2, 0.0452, 0.1852, 0.0204, 1.0, 30.0 / sbase],
        }
        gens = {
            1: [1, 0.0 / sbase, 20.0 / sbase, -20.0 / sbase, 100.0 / sbase, 0.00375, 2.0, 0.0],
        }
        return SympyACOPFModel(Sbase=sbase, buses=buses, lines=lines, gens=gens)

    if n_bus == 3:
        return SympyACOPFModel()

    sbase, buses, lines, gens = load_matpower_json(f"case{n_bus}_custom.json")
    return SympyACOPFModel(Sbase=sbase, buses=buses, lines=lines, gens=gens)


def apply_reference_voltage_bounds(
    model: SympyACOPFModel,
    ref_bus_id: int | None = None,
    vr_bound=REF_VR_BOUND,
    vi_bound=REF_VI_BOUND,
) -> int:
    if ref_bus_id is None:
        ref_bus_id = model.bus_ids[0]

    ref_idx = model.bus_index[ref_bus_id]
    vr_pos = model.variable_list.index(model.V_R[ref_idx])
    vi_pos = model.variable_list.index(model.V_I[ref_idx])
    model.Var_bound_list[vr_pos] = [float(vr_bound[0]), float(vr_bound[1])]
    model.Var_bound_list[vi_pos] = [float(vi_bound[0]), float(vi_bound[1])]
    return ref_bus_id


def reference_voltage_bound_indices(model: SympyACOPFModel, ref_bus_id: int | None = None) -> set[int]:
    if ref_bus_id is None:
        ref_bus_id = model.bus_ids[0]

    ref_idx = model.bus_index[ref_bus_id]
    return {
        model.variable_list.index(model.V_R[ref_idx]),
        model.variable_list.index(model.V_I[ref_idx]),
    }


def shrink_bounds_around_refined_solution(
    original_bounds,
    current_bounds,
    refined_x,
    shrink_factor: float,
    min_shrink_factor: float,
    fixed_indices: set[int],
):
    refined_x = np.asarray(refined_x, dtype=float).reshape(-1)
    new_bounds = []

    for idx, (orig_bound, current_bound, center) in enumerate(
        zip(original_bounds, current_bounds, refined_x)
    ):
        orig_lb, orig_ub = float(orig_bound[0]), float(orig_bound[1])
        current_lb, current_ub = float(current_bound[0]), float(current_bound[1])
        if idx in fixed_indices:
            new_bounds.append([current_lb, current_ub])
            continue

        orig_width = orig_ub - orig_lb
        current_width = current_ub - current_lb
        if orig_width <= 0.0 or current_width <= 0.0:
            new_bounds.append([current_lb, current_ub])
            continue

        min_width = orig_width * float(min_shrink_factor)
        new_width = min(orig_width, max(min_width, current_width * float(shrink_factor)))
        half_width = 0.5 * new_width
        lb = float(center) - half_width
        ub = float(center) + half_width

        if lb < orig_lb:
            lb = orig_lb
            ub = orig_lb + new_width
        if ub > orig_ub:
            ub = orig_ub
            lb = orig_ub - new_width

        new_bounds.append([max(orig_lb, lb), min(orig_ub, ub)])

    return new_bounds


def coarse_solution_unchanged(x_current, x_previous, ignored_indices: set[int], atol: float) -> bool:
    x_current = np.asarray(x_current, dtype=float).reshape(-1)
    x_previous = np.asarray(x_previous, dtype=float).reshape(-1)
    if x_current.shape != x_previous.shape:
        return False

    if ignored_indices:
        mask = np.ones(x_current.size, dtype=bool)
        mask[list(ignored_indices)] = False
        x_current = x_current[mask]
        x_previous = x_previous[mask]

    return bool(np.allclose(x_current, x_previous, rtol=0.0, atol=atol))


def candidate_points_equal(x_a, x_b, ignored_indices: set[int], atol: float) -> bool:
    x_a = np.asarray(x_a, dtype=float).reshape(-1)
    x_b = np.asarray(x_b, dtype=float).reshape(-1)
    if x_a.shape != x_b.shape:
        return False

    if ignored_indices:
        mask = np.ones(x_a.size, dtype=bool)
        mask[list(ignored_indices)] = False
        x_a = x_a[mask]
        x_b = x_b[mask]

    return bool(np.allclose(x_a, x_b, rtol=0.0, atol=atol))


def select_lowest_energy_distinct_candidates(
    candidates,
    limit: int,
    ignored_indices: set[int],
    atol: float,
):
    """Return at most ``limit`` finite-energy, distinct candidates."""
    ordered = sorted(
        (item for item in candidates if np.isfinite(item["lalm_energy"])),
        key=lambda item: item["lalm_energy"],
    )
    selected = []
    for item in ordered:
        if any(
            candidate_points_equal(item["x"], kept["x"], ignored_indices, atol)
            for kept in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def validate_config(config: SolverConfig) -> None:
    if config.alpha_mode not in {"adaptive", "fixed"}:
        raise ValueError("alpha_mode must be 'adaptive' or 'fixed'.")
    if config.alpha <= 0:
        raise ValueError("alpha must be positive.")
    if config.rho <= 0:
        raise ValueError("rho must be positive.")
    if config.option not in {1, 2}:
        raise ValueError("option must be 1 (QHD) or 2 (Gurobi).")
    if config.qhd_solver not in {"simbi", "openjij", "gurobi"}:
        raise ValueError("qhd_solver must be 'simbi', 'openjij', or 'gurobi'.")
    canonical_refine_method(config.refine_method)
    if config.max_outer <= 0:
        raise ValueError("max_outer must be positive.")
    if config.simbi_resolution <= 0:
        raise ValueError("simbi_resolution must be positive.")
    if config.simbi_shots <= 0:
        raise ValueError("simbi_shots must be positive.")
    if config.ipopt_max_iter <= 0:
        raise ValueError("ipopt_max_iter must be positive.")
    if config.tnc_maxfun is not None and config.tnc_maxfun <= 0:
        raise ValueError("tnc_maxfun must be positive or None.")
    if config.max_runtime_seconds is not None and config.max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive or None.")
    if config.n_linearization_points <= 0:
        raise ValueError("n_linearization_points must be positive.")
    if config.candidate_distinct_atol < 0.0:
        raise ValueError("candidate_distinct_atol must be nonnegative.")
    if config.beam_refine_keep <= 0:
        raise ValueError("beam_refine_keep must be positive.")
    if config.coarse_repeat_limit <= 0:
        raise ValueError("coarse_repeat_limit must be positive.")
    if not 0.0 < config.bound_shrink_factor <= 1.0:
        raise ValueError("bound_shrink_factor must be in (0, 1].")
    if not 0.0 < config.bound_shrink_min_factor <= 1.0:
        raise ValueError("bound_shrink_min_factor must be in (0, 1].")
    if config.bound_shrink_start_iter < 0:
        raise ValueError("bound_shrink_start_iter must be nonnegative.")
    if config.coarse_repeat_atol < 0.0:
        raise ValueError("coarse_repeat_atol must be nonnegative.")
    if config.rho_max < config.rho_min:
        raise ValueError("rho_max must be greater than or equal to rho_min.")
    if config.alpha_max < config.alpha_min:
        raise ValueError("alpha_max must be greater than or equal to alpha_min.")


def canonical_refine_method(method: str) -> str:
    key = str(method).lower()
    if key not in REFINE_METHOD_ALIASES:
        allowed = ", ".join(sorted(REFINE_METHODS))
        raise ValueError(f"refine_method must be one of: {allowed}.")
    return REFINE_METHOD_ALIASES[key]


def validate_bounds(variable_list, var_bound_list) -> None:
    bad_bounds = []
    for i, (var, bnd) in enumerate(zip(variable_list, var_bound_list)):
        lb, ub = float(bnd[0]), float(bnd[1])
        if ub < lb:
            bad_bounds.append((i, str(var), lb, ub))

    if bad_bounds:
        for item in bad_bounds:
            print("Invalid bound:", item)
        raise ValueError("Var_bound_list contains invalid bounds (ub < lb).")


def solve_subproblem_qhd(
    lagrangian,
    variable_list,
    var_bound_list,
    model: SympyACOPFModel,
    x_center,
    rho: float,
    config: SolverConfig,
    refine_var_bound_list=None,
    return_coarse_samples: bool = False,
):
    qhd_model = QHD.SymPy(lagrangian, variable_list, var_bound_list)
    refine_method = canonical_refine_method(config.refine_method)
    qhd_post_processing_method = (
        "TNC" if return_coarse_samples or refine_method == "none" else refine_method
    )

    if config.qhd_solver == "simbi":
        qhd_model.simbi_setup(
            resolution=config.simbi_resolution,
            shots=config.simbi_shots,
            agents=config.simbi_agents,
            max_steps=config.simbi_max_steps,
            embedding_scheme="unary",
            post_processing_method=qhd_post_processing_method,
            best_only=config.simbi_best_only,
            seed=config.simbi_seed,
            ballistic=config.simbi_ballistic,
            heated=config.simbi_heated,
            verbose=True,
        )
    elif config.qhd_solver == "openjij":
        qhd_model.openjij_setup(
            resolution=6,
            shots=2048,
            sampler_name="SQASampler",
            post_processing_method=qhd_post_processing_method,
            seed=42,
            debug=False,
            sampler_init_kwargs={},
            sample_kwargs={
                "beta": 5.0,
                "gamma": 1.0,
                "trotter": 8,
                "num_sweeps": 10000,
                "reinitialize_state": True,
            },
        )
    elif config.qhd_solver == "gurobi":
        qhd_model.gurobi_setup(
            resolution=4,
            shots=20,
            embedding_scheme="unary",
            solver_mode="ising",
            time_limit=30,
            threads=0,
            log_to_console=False,
            post_processing_method=qhd_post_processing_method,
        )
    else:
        raise ValueError(f"Unsupported qhd_solver={config.qhd_solver!r}.")

    should_refine = bool(
        not return_coarse_samples and config.qhd_refine and refine_method != "none"
    )
    if should_refine:
        tnc_options = {}
        if config.tnc_maxfun is not None:
            tnc_options["maxfun"] = config.tnc_maxfun
        qhd_model.set_acopf_refine_problem(
            objective=model._build_objective_expr(),
            constraints=model.build_h_symbolic(ref_bus_id=None),
            lambda_vec=np.asarray(model.lambda_vec, dtype=float),
            rho=rho,
            x_center=x_center,
            mu_prox=config.mu_prox,
            best_only=True,
            tnc_options=tnc_options,
            ipopt_options={
                "tol": config.tol,
                "max_iter": config.ipopt_max_iter,
                "hessian_approximation": "limited-memory",
            },
            gurobi_options={
                "time_limit": config.gurobi_time_limit,
                "threads": config.gurobi_threads,
                "log_to_console": config.gurobi_log_to_console,
            },
            refine_bounds=refine_var_bound_list,
        )

    response = qhd_model.optimize(refine=should_refine, verbose=0)
    if return_coarse_samples:
        samples = []
        for candidate in getattr(response, "coarse_samples", None) or []:
            if candidate is None:
                continue
            vector = np.asarray(candidate, dtype=float).reshape(-1)
            if vector.size == len(variable_list) and np.all(np.isfinite(vector)):
                samples.append(vector)

        try:
            samples.append(
                extract_qhd_solution_vector(
                    response,
                    prefer_refined=False,
                    expected_len=len(variable_list),
                )
            )
        except ValueError:
            pass

        if not samples:
            raise RuntimeError("QHD/Simbi returned no usable coarse samples.")
        return samples

    x_coarse = extract_qhd_solution_vector(
        response,
        prefer_refined=False,
        expected_len=len(variable_list),
    )
    x_new = extract_qhd_solution_vector(
        response,
        prefer_refined=True,
        expected_len=len(variable_list),
    )
    return x_coarse, x_new


def solve_subproblem_gurobi(lagrangian, variable_list, var_bound_list):
    return solve_with_gurobi_from_sympy(
        L_sym=lagrangian,
        variable_list=variable_list,
        Var_bound_list=var_bound_list,
        verbose=False,
    )


def evaluate_objective(model: SympyACOPFModel, x_vec) -> float:
    subs_dict = {var: val for var, val in zip(model.variable_list, x_vec)}
    return float(sp.N(model.objective.subs(subs_dict)))


def evaluate_sympy_expression(expr, variable_list, x_vec) -> float:
    subs_dict = {var: val for var, val in zip(variable_list, x_vec)}
    return float(sp.N(expr.subs(subs_dict)))


def build_scalar_evaluator(expr, variable_list):
    """Compile a SymPy scalar once for efficient evaluation of many samples."""
    raw_func = sp.lambdify(variable_list, expr, modules="numpy")

    def evaluate(x_vec):
        x_vec = np.asarray(x_vec, dtype=float).reshape(-1)
        return float(raw_func(*x_vec))

    return evaluate


def append_step_log(log_file: str, message: str) -> None:
    """Append one algorithm step immediately so partial runs remain auditable."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as stream:
        stream.write(f"[step {timestamp}] {message.rstrip()}\n")
        stream.flush()


def format_step_vector(x_vec) -> str:
    return np.array2string(
        np.asarray(x_vec, dtype=float).reshape(-1),
        precision=10,
        separator=",",
        max_line_width=1_000_000,
    )


def set_model_lambda_vec(model: SympyACOPFModel, lambda_vec):
    lambda_vec = np.asarray(lambda_vec, dtype=float).reshape(-1)
    model.lambda_vec = lambda_vec.tolist()

    nb = model.n_buses
    na = model.n_arcs
    idx = 0
    model.lambda_P_bal = lambda_vec[idx:idx + nb].tolist()
    idx += nb
    model.lambda_Q_bal = lambda_vec[idx:idx + nb].tolist()
    idx += nb
    model.lambda_P_flow = lambda_vec[idx:idx + na].tolist()
    idx += na
    model.lambda_Q_flow = lambda_vec[idx:idx + na].tolist()
    idx += na
    model.lambda_Vsq = lambda_vec[idx:idx + nb].tolist()
    idx += nb
    model.lambda_Ssq = lambda_vec[idx:idx + na].tolist()
    idx += na
    model.lambda_ref_VI = float(lambda_vec[idx])
    idx += 1
    model.lambda_ref_VR = float(lambda_vec[idx])
    idx += 1

    if idx != lambda_vec.size:
        raise ValueError(
            f"lambda length mismatch while setting model state: used {idx}, "
            f"got {lambda_vec.size}"
        )
    return lambda_vec


def _clip_to_bounds(x, var_bound_list):
    x = np.asarray(x, dtype=float).reshape(-1)
    lb = np.asarray([b[0] for b in var_bound_list], dtype=float)
    ub = np.asarray([b[1] for b in var_bound_list], dtype=float)
    return np.minimum(np.maximum(x, lb), ub)


def _scipy_bounds(var_bound_list):
    lb = np.asarray([b[0] for b in var_bound_list], dtype=float)
    ub = np.asarray([b[1] for b in var_bound_list], dtype=float)
    return Bounds(lb, ub)


def _lambdify_scalar_and_grad(expr, variable_list):
    expr = sp.expand(expr)
    grad_expr = [sp.diff(expr, var) for var in variable_list]
    f_raw = sp.lambdify(variable_list, expr, modules="numpy")
    g_raw = sp.lambdify(variable_list, grad_expr, modules="numpy")

    def fun(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(f_raw(*x))

    def jac(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.asarray(g_raw(*x), dtype=float).reshape(-1)

    return fun, jac


def _lambdify_constraints_and_jac(h_exprs, variable_list):
    h_vec = sp.Matrix(h_exprs)
    jac_expr = h_vec.jacobian(variable_list)
    h_raw = sp.lambdify(variable_list, h_exprs, modules="numpy")
    jac_raw = sp.lambdify(variable_list, jac_expr, modules="numpy")

    def cons(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.asarray(h_raw(*x), dtype=float).reshape(-1)

    def jac(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.asarray(jac_raw(*x), dtype=float)

    return cons, jac


def build_full_acopf_alm_expr(model: SympyACOPFModel, rho: float, x_center=None, mu_prox: float = 0.0):
    variable_list = model.variable_list
    obj = model._build_objective_expr()
    model.objective = obj
    h_exprs = model.build_h_symbolic(ref_bus_id=None)
    lam = np.asarray(model.lambda_vec, dtype=float).reshape(-1)

    if lam.size != len(h_exprs):
        raise ValueError(f"lambda size {lam.size} != number of ACOPF constraints {len(h_exprs)}")

    alm_expr = obj
    for lam_i, h_i in zip(lam, h_exprs):
        if lam_i != 0.0:
            alm_expr += sp.Float(lam_i) * h_i
        alm_expr += sp.Float(rho) * sp.Rational(1, 2) * h_i**2

    if mu_prox > 0.0 and x_center is not None:
        x0 = np.asarray(x_center, dtype=float).reshape(-1)
        if x0.size != len(variable_list):
            raise ValueError(f"x_center length mismatch: expected {len(variable_list)}, got {x0.size}")
        for var, val in zip(variable_list, x0):
            alm_expr += sp.Float(mu_prox) * sp.Rational(1, 2) * (var - sp.Float(val)) ** 2

    return sp.expand(alm_expr)


def solve_refine_tnc_orig(model: SympyACOPFModel, x0, x_center, rho: float, config: SolverConfig):
    alm_expr = build_full_acopf_alm_expr(
        model,
        rho=rho,
        x_center=x_center,
        mu_prox=config.mu_prox,
    )
    fun, jac = _lambdify_scalar_and_grad(alm_expr, model.variable_list)
    x0 = _clip_to_bounds(x0, model.Var_bound_list)
    options = {"gtol": 1e-6, "eps": 1e-9}
    if config.tnc_maxfun is not None:
        options["maxfun"] = config.tnc_maxfun

    result = minimize(
        fun,
        x0,
        method="TNC",
        jac=jac,
        bounds=_scipy_bounds(model.Var_bound_list),
        options=options,
    )
    print(f"[refine:TNC_orig] success={result.success}, status={result.status}, fun={float(result.fun):.9g}")
    if not result.success:
        print(f"[refine:TNC_orig] message={result.message}")
    return _clip_to_bounds(result.x, model.Var_bound_list)


def solve_refine_ipopt_orig(model: SympyACOPFModel, x0, config: SolverConfig):
    try:
        import cyipopt
    except ImportError as exc:
        raise RuntimeError("refine_method='ipopt_orig' requires cyipopt in the active Python environment.") from exc

    obj_expr = model._build_objective_expr()
    model.objective = obj_expr
    h_exprs = model.build_h_symbolic(ref_bus_id=None)
    fun, jac = _lambdify_scalar_and_grad(obj_expr, model.variable_list)
    cons_fun, cons_jac = _lambdify_constraints_and_jac(h_exprs, model.variable_list)
    x0 = _clip_to_bounds(x0, model.Var_bound_list)

    result = cyipopt.minimize_ipopt(
        fun,
        x0,
        jac=jac,
        bounds=_scipy_bounds(model.Var_bound_list),
        constraints={"type": "eq", "fun": cons_fun, "jac": cons_jac},
        options={
            "tol": config.tol,
            "max_iter": config.ipopt_max_iter,
            "hessian_approximation": "limited-memory",
        },
    )
    success = bool(getattr(result, "success", False))
    status = getattr(result, "status", None)
    fun_val = getattr(result, "fun", np.nan)
    print(f"[refine:ipopt_orig] success={success}, status={status}, objective={float(fun_val):.9g}")
    if not success:
        print(f"[refine:ipopt_orig] message={getattr(result, 'message', '')}")
    return _clip_to_bounds(result.x, model.Var_bound_list)


def _sympy_poly_to_gurobi_expr(expr, variable_list, gurobi_vars, max_degree=2):
    expanded = sp.expand(expr)
    poly = sp.Poly(expanded, variable_list)
    gurobi_expr = 0.0

    for monom, coeff in poly.terms():
        degree = sum(monom)
        if degree > max_degree:
            raise ValueError(
                f"Gurobi expression builder supports degree <= {max_degree}, "
                f"but found degree {degree} term {monom}."
            )

        term = float(coeff)
        for idx, power in enumerate(monom):
            for _ in range(power):
                term = term * gurobi_vars[idx]
        gurobi_expr += term

    return gurobi_expr


def _setup_gurobi_model(name: str, config: SolverConfig):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(f"refine_method='{name}' requires gurobipy in the active Python environment.") from exc

    model_g = gp.Model(name)
    model_g.Params.OutputFlag = 1 if config.gurobi_log_to_console else 0
    model_g.Params.NonConvex = 2
    if config.gurobi_time_limit is not None:
        model_g.Params.TimeLimit = float(config.gurobi_time_limit)
    if config.gurobi_threads is not None:
        model_g.Params.Threads = int(config.gurobi_threads)
    return gp, GRB, model_g


def _add_gurobi_decision_vars(model_g, variable_list, var_bound_list, prefix="x", start_values=None):
    if start_values is not None:
        start_values = _clip_to_bounds(start_values, var_bound_list)
    variables = []
    for idx, (sym, bounds) in enumerate(zip(variable_list, var_bound_list)):
        lb, ub = float(bounds[0]), float(bounds[1])
        var = model_g.addVar(lb=lb, ub=ub, name=f"{prefix}_{idx}_{sym}")
        if start_values is not None:
            var.Start = float(start_values[idx])
        variables.append(var)
    model_g.update()
    return variables


def solve_refine_gurobi_orig(model: SympyACOPFModel, x0, config: SolverConfig):
    _, _, model_g = _setup_gurobi_model("GurobiOrig_ACOPF_QCQP", config)
    x_vars = _add_gurobi_decision_vars(
        model_g,
        model.variable_list,
        model.Var_bound_list,
        start_values=x0,
    )

    obj_expr = model._build_objective_expr()
    model.objective = obj_expr
    model_g.setObjective(
        _sympy_poly_to_gurobi_expr(obj_expr, model.variable_list, x_vars, max_degree=2)
    )

    for idx, h_expr in enumerate(model.build_h_symbolic(ref_bus_id=None)):
        h_gurobi = _sympy_poly_to_gurobi_expr(h_expr, model.variable_list, x_vars, max_degree=2)
        model_g.addConstr(h_gurobi == 0.0, name=f"acopf_h_{idx}")

    model_g.optimize()
    if model_g.SolCount < 1:
        raise RuntimeError(f"GurobiOrig did not return a solution, status={model_g.Status}")

    print(f"[refine:GurobiOrig] status={model_g.Status}, objective={float(model_g.ObjVal):.9g}")
    return np.asarray([var.X for var in x_vars], dtype=float)


def solve_refine_gurobi_alm(model: SympyACOPFModel, x0, x_center, rho: float, config: SolverConfig):
    _, GRB, model_g = _setup_gurobi_model("GurobiALM_full_ACOPF_ALM", config)
    x0 = _clip_to_bounds(x0, model.Var_bound_list)
    x_vars = _add_gurobi_decision_vars(
        model_g,
        model.variable_list,
        model.Var_bound_list,
        start_values=x0,
    )

    obj_expr = model._build_objective_expr()
    model.objective = obj_expr
    gurobi_obj = _sympy_poly_to_gurobi_expr(obj_expr, model.variable_list, x_vars, max_degree=2)

    h_exprs = model.build_h_symbolic(ref_bus_id=None)
    lam = np.asarray(model.lambda_vec, dtype=float).reshape(-1)
    if lam.size != len(h_exprs):
        raise ValueError(f"lambda size {lam.size} != number of ACOPF constraints {len(h_exprs)}")

    h_func_start = np.asarray(model.build_h_func(ref_bus_id=None)(x0), dtype=float).reshape(-1)
    for idx, (lam_i, h_expr) in enumerate(zip(lam, h_exprs)):
        h_var = model_g.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"h_{idx}")
        h_var.Start = float(h_func_start[idx])
        h_gurobi = _sympy_poly_to_gurobi_expr(h_expr, model.variable_list, x_vars, max_degree=2)
        model_g.addConstr(h_var == h_gurobi, name=f"define_h_{idx}")
        if lam_i != 0.0:
            gurobi_obj += float(lam_i) * h_var
        gurobi_obj += 0.5 * float(rho) * h_var * h_var

    if config.mu_prox > 0.0 and x_center is not None:
        x0_center = np.asarray(x_center, dtype=float).reshape(-1)
        if x0_center.size != len(x_vars):
            raise ValueError(f"x_center length mismatch: expected {len(x_vars)}, got {x0_center.size}")
        for var, center_val in zip(x_vars, x0_center):
            gurobi_obj += 0.5 * float(config.mu_prox) * (var - float(center_val)) * (var - float(center_val))

    model_g.setObjective(gurobi_obj)
    model_g.optimize()
    if model_g.SolCount < 1:
        raise RuntimeError(f"GurobiALM did not return a solution, status={model_g.Status}")

    print(f"[refine:GurobiALM] status={model_g.Status}, ALM objective={float(model_g.ObjVal):.9g}")
    return np.asarray([var.X for var in x_vars], dtype=float)


def refine_acopf_solution(model: SympyACOPFModel, x_coarse, x_center, rho: float, config: SolverConfig):
    method = canonical_refine_method(config.refine_method)
    x_coarse = _clip_to_bounds(x_coarse, model.Var_bound_list)

    if method == "none":
        return x_coarse
    if method == "TNC_orig":
        return solve_refine_tnc_orig(model, x_coarse, x_center=x_center, rho=rho, config=config)
    if method == "ipopt_orig":
        return solve_refine_ipopt_orig(model, x_coarse, config=config)
    if method == "GurobiALM":
        return solve_refine_gurobi_alm(model, x_coarse, x_center=x_center, rho=rho, config=config)
    if method == "GurobiOrig":
        return solve_refine_gurobi_orig(model, x_coarse, config=config)

    raise ValueError(f"Unsupported refine_method={method!r}.")


def adapt_alpha_rho(
    norm_h: float,
    lambda_inf: float,
    prev_norm_h: float | None,
    prev_lambda_inf: float | None,
    alpha: float,
    rho: float,
    stable_count: int,
    plateau_count: int,
    worsen_count: int,
    config: SolverConfig,
    alpha_max: float | None = None,
):
    if config.alpha_mode != "adaptive":
        return alpha, rho, stable_count, plateau_count, worsen_count

    alpha_cap = config.alpha_max if alpha_max is None else float(alpha_max)

    if prev_norm_h is not None:
        ratio = norm_h / max(prev_norm_h, 1e-12)

        if ratio <= 1.0 - config.improve_tol:
            stable_count += 1
            plateau_count = 0
            worsen_count = 0
            if stable_count >= config.stable_window:
                alpha = min(alpha * 1.03, alpha_cap)
                stable_count = 0
                print(f"[adaptive] Stable improvement detected, alpha -> {alpha:.6e}")

        elif ratio >= 1.0 + config.worsen_tol:
            worsen_count += 1
            plateau_count = 0
            stable_count = 0
            alpha = max(alpha * 0.8, config.alpha_min)
            print(f"[adaptive] Residual worsened, alpha -> {alpha:.6e}")
            if worsen_count >= config.worsen_window:
                rho = min(rho * 1.15, config.rho_max)
                worsen_count = 0
                print(f"[adaptive] Repeated worsening, rho -> {rho:.6e}")

        else:
            plateau_count += 1
            stable_count = 0
            worsen_count = 0
            if plateau_count >= config.plateau_window:
                rho = min(rho * 1.2, config.rho_max)
                alpha = max(alpha * 0.9, config.alpha_min)
                plateau_count = 0
                print(f"[adaptive] Plateau detected, rho -> {rho:.6e}, alpha -> {alpha:.6e}")

    if prev_lambda_inf is not None and prev_lambda_inf > 1e-10 and lambda_inf > 1.4 * prev_lambda_inf:
        alpha = max(alpha * 0.85, config.alpha_min)
        stable_count = 0
        worsen_count = 0
        print(f"[adaptive] Lambda growth too fast, alpha -> {alpha:.6e}")

    return alpha, rho, stable_count, plateau_count, worsen_count


def save_objective_plot(
    objective_history, log_file: str, log_folder: str, show: bool = True
) -> None:
    valid_history = [item for item in objective_history if np.isfinite(item["objective"])]
    if not valid_history:
        print("Objective history plot skipped: no valid objective values recorded.")
        return

    iterations = [item["iter"] for item in valid_history]
    objective_values = [item["objective"] for item in valid_history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iterations, objective_values, color="tab:blue", linewidth=1.8)
    ax.set_title("Objective Function Value Iteration History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    plot_path = Path(log_folder) / f"{Path(log_file).stem}-Obj.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print("Objective history plot:", plot_path)
    if show:
        plt.show()
    plt.close(fig)


def run_coarse_beam_search_alm(model: SympyACOPFModel, config: SolverConfig):
    """Run multi-state beam search with one lambda vector per active beam point."""
    validate_config(config)

    h_func = model.build_h_func()
    model.reset_lambdas(0.0)
    objective_expr = model._build_objective_expr()
    model.objective = objective_expr
    objective_func = build_scalar_evaluator(objective_expr, model.variable_list)

    x_initial = np.asarray(model.build_initial_x0(), dtype=float)
    initial_lambda_vec = np.asarray(model.lambda_vec, dtype=float).reshape(-1)
    beam_states = [
        {"x": x_initial.copy(), "lambda_vec": initial_lambda_vec.copy()}
    ]
    previous_evaluation_x = x_initial.copy()

    alpha = config.alpha
    rho = config.rho
    alpha_max_current = config.alpha_max
    log_file = initialize_qhd_acopf_log(
        model,
        folder=config.log_folder,
        option=config.option,
        qhd_solver=config.qhd_solver,
    )

    def log_step(message: str) -> None:
        append_step_log(log_file, message)

    print("Log file:", log_file)
    print(f"Alpha mode: {config.alpha_mode}")
    if config.beam_refine_candidates:
        print(
            "Beam refine: enabled for selected candidates via "
            f"{canonical_refine_method(config.refine_method)}"
        )
        print(f"Refined objective keep M: {config.beam_refine_keep}")
    else:
        print("Beam refine: disabled (coarse answers only)")
    print(f"Linearization beam width N: {config.n_linearization_points}")
    print(
        f"Bound continuation: factor={config.bound_shrink_factor}, "
        f"minimum={config.bound_shrink_min_factor:.3%} of original width, "
        f"start iteration={config.bound_shrink_start_iter}, "
        "requires residual improvement="
        f"{config.bound_shrink_require_residual_improvement}"
    )
    if config.alpha_mode == "fixed":
        print(f"Fixed alpha: {alpha}")
    else:
        print(f"Adaptive alpha/rho start: alpha={alpha}, rho={rho}")

    log_step(
        "run_start "
        f"n_bus={config.n_bus}, N={config.n_linearization_points}, "
        f"beam_refine={config.beam_refine_candidates}, "
        f"refine_method={canonical_refine_method(config.refine_method)}, "
        f"beam_refine_keep={config.beam_refine_keep}, "
        f"alpha_mode={config.alpha_mode}, "
        f"alpha={alpha:.12g}, rho={rho:.12g}, rho_max={config.rho_max:.12g}, "
        f"bound_shrink_factor={config.bound_shrink_factor:.12g}, "
        f"bound_min_factor={config.bound_shrink_min_factor:.12g}, "
        f"bound_start_iter={config.bound_shrink_start_iter}, "
        "bound_requires_residual_improvement="
        f"{config.bound_shrink_require_residual_improvement}"
    )

    print("\n===== Start Multi-Point Linear ALM Beam Loop =====\n")
    start_time = time.monotonic()
    runtime_timeout = False
    converged = False
    xk = x_initial.copy()
    objective_history = []
    metric_history = []
    candidate_history = []
    plateau_count = 0
    worsen_count = 0
    stable_count = 0
    prev_norm_h = None
    prev_lambda_inf = None
    best_residual_iter = None
    best_residual = float("inf")

    original_var_bound_list = [
        [float(bound[0]), float(bound[1])] for bound in model.Var_bound_list
    ]
    qhd_var_bound_list = [bound.copy() for bound in original_var_bound_list]
    fixed_bound_indices = reference_voltage_bound_indices(model)
    bounds_shrink_count = 0

    best_record = {
        "iter": None,
        "metric": float("inf"),
        "x": None,
        "objective": float("inf"),
        "h_x": None,
        "lambda_vec": None,
        "feasible": False,
        "rho": None,
        "alpha": None,
    }

    for k in range(config.max_outer):
        elapsed_seconds = time.monotonic() - start_time
        if (
            config.max_runtime_seconds is not None
            and elapsed_seconds >= config.max_runtime_seconds
        ):
            runtime_timeout = True
            print("\nRuntime limit reached before the next generation.")
            log_step(
                f"runtime_limit before_iteration={k}, "
                f"elapsed_seconds={elapsed_seconds:.6f}"
            )
            break

        if config.alpha_mode == "fixed":
            alpha = config.alpha

        alpha_used = alpha
        rho_used = rho
        parent_states = [
            {
                "x": np.asarray(state["x"], dtype=float).copy(),
                "lambda_vec": np.asarray(state["lambda_vec"], dtype=float).reshape(-1).copy(),
            }
            for state in beam_states
        ]
        all_candidates = []
        raw_solver_sample_count = 0
        local_candidate_counts = []

        print(f"\n--- Outer Iteration {k} ---")
        print(
            f"linearization points = {len(parent_states)}, "
            f"alpha = {alpha_used:.6e}, rho = {rho_used:.6e}"
        )
        log_step(
            f"iteration_start iter={k}, parents={len(parent_states)}, "
            f"alpha={alpha_used:.12g}, rho={rho_used:.12g}"
        )

        for parent_index, parent_state in enumerate(parent_states):
            x_center = parent_state["x"]
            parent_lambda_vec = parent_state["lambda_vec"]
            set_model_lambda_vec(model, parent_lambda_vec)
            if (
                config.max_runtime_seconds is not None
                and time.monotonic() - start_time >= config.max_runtime_seconds
            ):
                runtime_timeout = True
                print(
                    f"[runtime] stopped after {parent_index}/{len(parent_states)} "
                    "linearization solves in this generation."
                )
                log_step(
                    f"runtime_limit iter={k}, completed_parents={parent_index}/"
                    f"{len(parent_states)}"
                )
                break

            print(
                f"[QHD] linearization {parent_index + 1}/{len(parent_states)}"
            )
            log_step(
                f"qhd_start iter={k}, parent={parent_index + 1}/"
                f"{len(parent_states)}, center={format_step_vector(x_center)}"
            )
            lagrangian, variable_list, var_bound_list = (
                model.build_linear_ALM_Lagrangian_syms(
                    x_center=x_center,
                    rho=rho_used,
                    ref_bus_id=None,
                    mu_prox=config.mu_prox,
                )
            )
            solve_bounds = qhd_var_bound_list if config.option == 1 else var_bound_list
            validate_bounds(variable_list, solve_bounds)

            try:
                if config.option == 1:
                    coarse_samples = solve_subproblem_qhd(
                        lagrangian,
                        variable_list,
                        solve_bounds,
                        model=model,
                        x_center=x_center,
                        rho=rho_used,
                        config=config,
                        refine_var_bound_list=model.Var_bound_list,
                        return_coarse_samples=True,
                    )
                else:
                    coarse_samples = [
                        solve_subproblem_gurobi(
                            lagrangian,
                            variable_list,
                            solve_bounds,
                        )
                    ]
            except Exception as exc:
                log_step(
                    f"qhd_error iter={k}, parent={parent_index + 1}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise

            raw_solver_sample_count += len(coarse_samples)
            energy_func = build_scalar_evaluator(lagrangian, variable_list)
            parent_candidates = []
            for sample in coarse_samples:
                sample = np.asarray(sample, dtype=float).reshape(-1)
                try:
                    energy = energy_func(sample)
                except (FloatingPointError, OverflowError, TypeError, ValueError):
                    continue
                parent_candidates.append(
                    {
                        "x": sample.copy(),
                        "lalm_energy": energy,
                        "parent_index": parent_index,
                        "parent_lambda_vec": parent_lambda_vec.copy(),
                    }
                )

            # Stage 1: retain N distinct low-energy answers from this LALM.
            # With N parents, the global stage receives the requested N*N pool.
            parent_selected = select_lowest_energy_distinct_candidates(
                parent_candidates,
                limit=config.n_linearization_points,
                ignored_indices=fixed_bound_indices,
                atol=config.candidate_distinct_atol,
            )
            local_candidate_counts.append(len(parent_selected))
            all_candidates.extend(parent_selected)
            print(
                f"[local beam {parent_index + 1}] "
                f"raw={len(parent_candidates)}, "
                f"distinct selected={len(parent_selected)}/"
                f"{config.n_linearization_points}"
            )
            log_step(
                f"local_beam_complete iter={k}, parent={parent_index + 1}, "
                f"raw={len(parent_candidates)}, selected={len(parent_selected)}/"
                f"{config.n_linearization_points}"
            )
            for local_rank, item in enumerate(parent_selected, start=1):
                log_step(
                    f"local_candidate iter={k}, parent={parent_index + 1}, "
                    f"rank={local_rank}, energy={item['lalm_energy']:.15g}, "
                    f"x={format_step_vector(item['x'])}"
                )
            if len(parent_selected) < config.n_linearization_points:
                print(
                    f"[local beam {parent_index + 1}] warning: only "
                    f"{len(parent_selected)} distinct points were available."
                )
                log_step(
                    f"local_beam_warning iter={k}, parent={parent_index + 1}, "
                    f"requested={config.n_linearization_points}, "
                    f"available={len(parent_selected)}"
                )

        if not all_candidates:
            if runtime_timeout:
                break
            log_step(f"error iter={k}: no finite local candidates")
            raise RuntimeError("No finite coarse candidates were produced in this generation.")

        selected = select_lowest_energy_distinct_candidates(
            all_candidates,
            limit=config.n_linearization_points,
            ignored_indices=fixed_bound_indices,
            atol=config.candidate_distinct_atol,
        )
        if not selected:
            raise RuntimeError("No finite, distinct coarse candidates remained after filtering.")
        if len(selected) < config.n_linearization_points:
            print(
                f"[beam] warning: requested {config.n_linearization_points} distinct "
                f"points, but only {len(selected)} were available."
            )
            log_step(
                f"global_beam_warning iter={k}, "
                f"requested={config.n_linearization_points}, available={len(selected)}"
            )

        for energy_rank, item in enumerate(selected, start=1):
            item["energy_rank"] = energy_rank
            item["coarse_x"] = item["x"].copy()
            item["coarse_lalm_energy"] = item["lalm_energy"]
            item["coarse_objective"] = objective_func(item["coarse_x"])

        if config.beam_refine_candidates:
            refine_method = canonical_refine_method(config.refine_method)
            print(
                f"[beam refine] refining {len(selected)} selected candidates "
                f"with {refine_method}"
            )
            log_step(
                f"beam_refine_start iter={k}, candidates={len(selected)}, "
                f"method={refine_method}"
            )
            for item in selected:
                parent_state = parent_states[item["parent_index"]]
                parent_center = parent_state["x"]
                parent_lambda_vec = item["parent_lambda_vec"]
                set_model_lambda_vec(model, parent_lambda_vec)
                refined_x = refine_acopf_solution(
                    model,
                    item["coarse_x"],
                    x_center=parent_center,
                    rho=rho_used,
                    config=config,
                )
                refined_x = np.asarray(refined_x, dtype=float).reshape(-1).copy()
                item["x"] = refined_x
                item["refined"] = True
                item["objective"] = objective_func(refined_x)
                item["refined_objective"] = item["objective"]
                log_step(
                    f"beam_refined_candidate iter={k}, "
                    f"energy_rank={item['energy_rank']}, "
                    f"parent={item['parent_index'] + 1}, "
                    f"coarse_objective={item['coarse_objective']:.15g}, "
                    f"refined_objective={item['objective']:.15g}, "
                    f"x={format_step_vector(refined_x)}"
                )
        else:
            for item in selected:
                item["refined"] = False
                item["objective"] = item["coarse_objective"]
                item["refined_objective"] = None

        for item in selected:
            candidate_h = np.asarray(h_func(item["x"]), dtype=float).reshape(-1)
            parent_lambda_vec = np.asarray(
                item["parent_lambda_vec"], dtype=float
            ).reshape(-1)
            if parent_lambda_vec.shape != candidate_h.shape:
                raise ValueError(
                    f"parent lambda shape {parent_lambda_vec.shape} does not match "
                    f"h(x) shape {candidate_h.shape}"
                )
            item["h_x"] = candidate_h
            item["norm_h"] = float(np.linalg.norm(candidate_h))
            item["max_abs_h"] = float(np.max(np.abs(candidate_h)))
            _, candidate_check_flag = model.check_constraints(item["x"])
            item["check_flag"] = bool(candidate_check_flag)
            parent_x = parent_states[item["parent_index"]]["x"]
            item["step_norm"] = float(np.linalg.norm(item["x"] - parent_x))
            item["lambda_vec"] = parent_lambda_vec + alpha_used * candidate_h
            item["lambda_inf"] = float(np.max(np.abs(item["lambda_vec"])))

        active_candidate_count = min(config.beam_refine_keep, len(selected))
        active_candidates = sorted(
            selected,
            key=lambda item: item["objective"],
        )[:active_candidate_count]
        active_ids = {id(item) for item in active_candidates}
        converged_active_candidates = [
            item
            for item in active_candidates
            if item["check_flag"]
            or (item["norm_h"] < config.tol and item["step_norm"] < config.tol)
        ]
        converged_active_ids = {id(item) for item in converged_active_candidates}
        objective_best = active_candidates[0]
        evaluation = (
            min(converged_active_candidates, key=lambda item: item["objective"])
            if converged_active_candidates
            else objective_best
        )
        evaluation_rank = evaluation["energy_rank"] - 1
        x_new = evaluation["x"].copy()
        h_val = evaluation["h_x"].copy()
        norm_h = evaluation["norm_h"]
        objective_value = evaluation["objective"]
        check_flag = evaluation["check_flag"]
        step_norm = evaluation["step_norm"]
        lambda_new = evaluation["lambda_vec"].copy()
        lambda_inf = evaluation["lambda_inf"]
        set_model_lambda_vec(model, lambda_new)
        residual_improved = (
            prev_norm_h is not None
            and norm_h <= prev_norm_h * (1.0 - config.improve_tol)
        )

        print(
            f"[beam] solver samples={raw_solver_sample_count}, "
            f"candidate pool={len(all_candidates)}, "
            f"global distinct selected={len(selected)}, "
            f"active objective top-M={len(active_candidates)}, "
            f"converged active={len(converged_active_candidates)}, "
            f"evaluation energy rank={evaluation_rank + 1}"
        )
        log_step(
            f"global_beam_complete iter={k}, "
            f"solver_samples={raw_solver_sample_count}, "
            f"candidate_pool={len(all_candidates)}, selected={len(selected)}, "
            f"active={len(active_candidates)}, "
            f"converged_active={len(converged_active_candidates)}, "
            f"evaluation_rank={evaluation_rank + 1}, "
            f"local_counts={local_candidate_counts}"
        )
        for rank, item in enumerate(selected, start=1):
            marker_parts = []
            if id(item) in active_ids:
                marker_parts.append("active")
            if item is objective_best:
                marker_parts.append("objective-best")
            if id(item) in converged_active_ids:
                marker_parts.append("converged")
            if item is evaluation:
                marker_parts.append("evaluation")
            marker = " <-- " + ", ".join(marker_parts) if marker_parts else ""
            if item["refined"]:
                print(
                    f"  #{rank:02d} energy={item['lalm_energy']:.12g}, "
                    f"coarse_obj={item['coarse_objective']:.9g}, "
                    f"refined_obj={item['objective']:.9g}{marker}"
                )
            else:
                print(
                    f"  #{rank:02d} energy={item['lalm_energy']:.12g}, "
                    f"obj={item['objective']:.9g}{marker}"
                )
            log_step(
                f"global_candidate iter={k}, rank={rank}, "
                f"parent={item['parent_index'] + 1}, "
                f"energy={item['lalm_energy']:.15g}, "
                f"coarse_objective={item['coarse_objective']:.15g}, "
                f"objective={item['objective']:.15g}, "
                f"refined={item['refined']}, "
                f"active_choice={id(item) in active_ids}, "
                f"converged_choice={id(item) in converged_active_ids}, "
                f"evaluation_choice={item is evaluation}, "
                f"lambda_inf={item['lambda_inf']:.15g}, "
                f"coarse_x={format_step_vector(item['coarse_x'])}, "
                f"x={format_step_vector(item['x'])}"
            )
        print(
            f"[evaluation] objective={objective_value:.9g}, "
            f"||h(x)||={norm_h:.6e}, step={step_norm:.6e}"
        )
        log_step(
            f"evaluation iter={k}, rank={evaluation_rank + 1}, "
            f"objective={objective_value:.15g}, lalm_energy="
            f"{evaluation['lalm_energy']:.15g}, l2_norm_h={norm_h:.15g}, "
            f"max_abs_h={float(np.max(np.abs(h_val))):.15g}, "
            f"step_norm={step_norm:.15g}, feasible={check_flag}, "
            f"active_candidates={len(active_candidates)}, "
            f"converged_active={len(converged_active_candidates)}, "
            f"lambda_inf={lambda_inf:.15g}, "
            f"x={format_step_vector(x_new)}"
        )

        alpha, rho, stable_count, plateau_count, worsen_count = adapt_alpha_rho(
            norm_h=norm_h,
            lambda_inf=lambda_inf,
            prev_norm_h=prev_norm_h,
            prev_lambda_inf=prev_lambda_inf,
            alpha=alpha_used,
            rho=rho_used,
            stable_count=stable_count,
            plateau_count=plateau_count,
            worsen_count=worsen_count,
            config=config,
            alpha_max=alpha_max_current,
        )
        log_step(
            f"adaptive_update iter={k}, alpha_used={alpha_used:.15g}, "
            f"rho_used={rho_used:.15g}, next_alpha={alpha:.15g}, "
            f"next_rho={rho:.15g}, lambda_inf={lambda_inf:.15g}, "
            f"stable_count={stable_count}, plateau_count={plateau_count}, "
            f"worsen_count={worsen_count}"
        )
        prev_norm_h = norm_h
        prev_lambda_inf = lambda_inf

        # The objective-best M candidates become full primal-dual beam states
        # for the next generation, each carrying its own lambda vector.
        beam_states = [
            {"x": item["x"].copy(), "lambda_vec": item["lambda_vec"].copy()}
            for item in active_candidates
        ]
        shrink_bounds_this_round = (
            k >= config.bound_shrink_start_iter
            and (
                not config.bound_shrink_require_residual_improvement
                or residual_improved
            )
        )
        if shrink_bounds_this_round:
            bounds_shrink_count += 1
            qhd_var_bound_list = shrink_bounds_around_refined_solution(
                original_var_bound_list,
                qhd_var_bound_list,
                x_new,
                shrink_factor=config.bound_shrink_factor,
                min_shrink_factor=config.bound_shrink_min_factor,
                fixed_indices=fixed_bound_indices,
            )
            validate_bounds(model.variable_list, qhd_var_bound_list)

        effective_cumulative_shrink = max(
            config.bound_shrink_min_factor,
            config.bound_shrink_factor ** bounds_shrink_count,
        )
        if shrink_bounds_this_round:
            print(
                f"[bounds] accepted residual improvement; shrink "
                f"#{bounds_shrink_count} around objective-best solution; "
                f"width factor={config.bound_shrink_factor:.6g}, "
                f"cumulative~={effective_cumulative_shrink:.6g}"
            )
            log_step(
                f"bounds_shrink iter={k}, applied=True, "
                f"shrink_count={bounds_shrink_count}, "
                f"factor={config.bound_shrink_factor:.15g}, "
                f"cumulative_factor={effective_cumulative_shrink:.15g}, "
                f"center={format_step_vector(x_new)}"
            )
        else:
            if k < config.bound_shrink_start_iter:
                hold_reason = (
                    f"warm-up ({k + 1}/{config.bound_shrink_start_iter} rounds)"
                )
            else:
                hold_reason = "nonlinear residual did not improve enough"
            print(
                f"[bounds] unchanged: {hold_reason}; "
                f"cumulative factor={effective_cumulative_shrink:.6g}"
            )
            log_step(
                f"bounds_shrink iter={k}, applied=False, reason={hold_reason}, "
                f"shrink_count={bounds_shrink_count}, "
                f"cumulative_factor={effective_cumulative_shrink:.15g}"
            )

        objective_history.append({"iter": k, "objective": objective_value})
        candidate_history.append(
            {
                "iter": k,
                "raw_solver_sample_count": raw_solver_sample_count,
                "pooled_candidate_count": len(all_candidates),
                "local_candidate_counts": local_candidate_counts.copy(),
                "selected": [
                    {
                        "x": item["x"].copy(),
                        "coarse_x": item["coarse_x"].copy(),
                        "lalm_energy": item["lalm_energy"],
                        "coarse_objective": item["coarse_objective"],
                        "objective": item["objective"],
                        "refined": item["refined"],
                        "parent_index": item["parent_index"],
                        "parent_lambda_vec": item["parent_lambda_vec"].copy(),
                        "lambda_vec": item["lambda_vec"].copy(),
                        "lambda_inf": item["lambda_inf"],
                        "l2_norm_h": item["norm_h"],
                        "max_abs_h": item["max_abs_h"],
                        "step_norm": item["step_norm"],
                        "feasible": item["check_flag"],
                        "active": id(item) in active_ids,
                        "converged_choice": id(item) in converged_active_ids,
                        "evaluation_choice": item is evaluation,
                    }
                    for item in selected
                ],
                "active_candidate_count": len(active_candidates),
                "converged_active_candidate_count": len(converged_active_candidates),
                "evaluation_rank": evaluation_rank,
            }
        )
        metric_history.append(
            {
                "iter": k,
                "max_abs_h": float(np.max(np.abs(h_val))),
                "l2_norm_h": norm_h,
                "objective": objective_value,
                "lalm_energy": evaluation["lalm_energy"],
                "evaluation_energy_rank": evaluation_rank + 1,
                "selected_candidate_count": len(selected),
                "active_candidate_count": len(active_candidates),
                "converged_active_candidate_count": len(converged_active_candidates),
                "beam_refine_candidates": config.beam_refine_candidates,
                "beam_refine_keep": config.beam_refine_keep,
                "raw_solver_sample_count": raw_solver_sample_count,
                "pooled_candidate_count": len(all_candidates),
                "local_candidate_counts": local_candidate_counts.copy(),
                "alpha": float(alpha_used),
                "rho": float(rho_used),
                "next_alpha": float(alpha),
                "next_rho": float(rho),
                "lambda_inf": lambda_inf,
                "residual_improved": residual_improved,
                "bound_shrink_applied": shrink_bounds_this_round,
                "bounds_shrink_count": bounds_shrink_count,
                "cumulative_bound_factor": effective_cumulative_shrink,
            }
        )

        objective_is_better = objective_value < best_record["objective"] - 1e-12
        objective_tied_with_better_residual = (
            abs(objective_value - best_record["objective"]) <= 1e-12
            and norm_h < best_record["metric"]
        )
        if objective_is_better or objective_tied_with_better_residual:
            best_record.update(
                {
                    "iter": k,
                    "metric": norm_h,
                    "x": x_new.copy(),
                    "objective": objective_value,
                    "h_x": h_val.copy(),
                    "lambda_vec": np.asarray(lambda_new, dtype=float).copy(),
                    "feasible": check_flag,
                    "rho": float(rho_used),
                    "alpha": float(alpha_used),
                }
            )
            print(
                f"[best objective] iter={k}, objective={objective_value:.9g}, "
                f"||h||={norm_h:.6e}"
            )
            log_step(
                f"best_objective_update iter={k}, "
                f"objective={objective_value:.15g}, l2_norm_h={norm_h:.15g}, "
                f"feasible={check_flag}, x={format_step_vector(x_new)}"
            )

        if norm_h < best_residual - 1e-12:
            best_residual = norm_h
            best_residual_iter = k

        result_note = (
            ("refined" if config.beam_refine_candidates else "coarse")
            + "_objective_best_from_top_"
            + f"{len(active_candidates)}_active_beam_candidates"
        )
        log_file = PrintQHDACOPFResults(
            model,
            x_new,
            log_file=log_file,
            iteration=k,
            folder=config.log_folder,
            print_to_console=config.print_to_console,
            rho=rho_used,
            alpha=alpha_used,
            h_x=h_val,
            lambda_vec=lambda_new,
            objective_value=objective_value,
            lalm_energy=evaluation["lalm_energy"],
            feasibility=check_flag,
            note=result_note,
        )
        log_step(f"iteration_result_written iter={k}, note={result_note}")

        xk = x_new.copy()
        previous_evaluation_x = x_new.copy()

        if converged_active_candidates:
            converged = True
            print("\nConverged!")
            log_step(
                f"converged iter={k}, feasible={check_flag}, "
                f"converged_active={len(converged_active_candidates)}, "
                f"l2_norm_h={norm_h:.15g}, step_norm={step_norm:.15g}"
            )
            break

        if (
            config.early_stop_patience > 0
            and best_residual_iter is not None
            and k - best_residual_iter >= config.early_stop_patience
        ):
            print(
                "\nEarly stop: no h(x) residual improvement for "
                f"{config.early_stop_patience} iterations."
            )
            log_step(
                f"early_stop iter={k}, patience={config.early_stop_patience}, "
                f"best_residual_iter={best_residual_iter}, "
                f"best_residual={best_residual:.15g}"
            )
            break

        if runtime_timeout:
            break

    print("\n===== End Loop =====\n")
    if runtime_timeout:
        print("Stopped because max_runtime_seconds was reached.")
    elif not converged:
        print("Stopped without satisfying the convergence test.")
    print("Final log file:", log_file)
    log_step(
        f"run_end converged={converged}, runtime_timeout={runtime_timeout}, "
        f"completed_iterations={len(metric_history)}, final_alpha={alpha:.15g}, "
        f"final_rho={rho:.15g}"
    )

    if best_record["iter"] is not None:
        print(
            "Best objective iteration:",
            best_record["iter"],
            f"objective={best_record['objective']:.9g}",
            f"l2_norm_h={best_record['metric']:.6e}",
        )
        log_file = PrintQHDACOPFResults(
            model,
            best_record["x"],
            log_file=log_file,
            iteration=best_record["iter"],
            folder=config.log_folder,
            print_to_console=False,
            rho=best_record["rho"],
            alpha=best_record["alpha"],
            h_x=best_record["h_x"],
            lambda_vec=best_record["lambda_vec"],
            objective_value=best_record["objective"],
            feasibility=best_record["feasible"],
            note="best_iteration_by_objective",
        )
        log_step(
            f"best_result_written iter={best_record['iter']}, "
            f"objective={best_record['objective']:.15g}, "
            f"l2_norm_h={best_record['metric']:.15g}, "
            f"feasible={best_record['feasible']}"
        )
        if config.return_best_solution:
            xk = best_record["x"].copy()

    save_objective_plot(
        objective_history,
        log_file,
        config.log_folder,
        show=config.show_plot,
    )
    log_step(
        f"objective_plot_written path="
        f"{Path(config.log_folder) / (Path(log_file).stem + '-Obj.png')}"
    )

    return {
        "x": xk,
        "linearization_points": [state["x"].copy() for state in beam_states],
        "beam_states": [
            {"x": state["x"].copy(), "lambda_vec": state["lambda_vec"].copy()}
            for state in beam_states
        ],
        "log_file": log_file,
        "objective_history": objective_history,
        "metric_history": metric_history,
        "candidate_history": candidate_history,
        "best_record": best_record,
        "final_alpha": alpha,
        "final_rho": rho,
        "converged": converged,
    }



def run_single_point_linear_alm(model: SympyACOPFModel, config: SolverConfig):
    validate_config(config)

    h_func = model.build_h_func()
    model.reset_lambdas(0.0)
    xk = model.build_initial_x0()

    alpha = config.alpha
    rho = config.rho
    alpha_max_current = config.alpha_max
    log_file = initialize_qhd_acopf_log(
        model,
        folder=config.log_folder,
        option=config.option,
        qhd_solver=config.qhd_solver,
    )

    print("Log file:", log_file)
    print(f"Alpha mode: {config.alpha_mode}")
    print(f"Refine method: {canonical_refine_method(config.refine_method)}")
    if config.alpha_mode == "fixed":
        print(f"Fixed alpha: {alpha}")
    else:
        print(f"Adaptive alpha/rho start: alpha={alpha}, rho={rho}")

    print("\n===== Start Linear ALM Loop =====\n")
    start_time = time.monotonic()
    runtime_timeout = False
    objective_history = []
    metric_history = []
    plateau_count = 0
    worsen_count = 0
    stable_count = 0
    prev_norm_h = None
    prev_lambda_inf = None
    original_var_bound_list = [[float(b[0]), float(b[1])] for b in model.Var_bound_list]
    qhd_var_bound_list = [[float(b[0]), float(b[1])] for b in original_var_bound_list]
    fixed_bound_indices = reference_voltage_bound_indices(model)
    previous_coarse = None
    coarse_repeat_count = 0
    bounds_shrink_count = 0
    best_record = {
        "iter": None,
        "metric": float("inf"),
        "x": None,
        "objective": None,
        "h_x": None,
        "lambda_vec": None,
        "feasible": False,
        "rho": None,
        "alpha": None,
    }

    for k in range(config.max_outer):
        elapsed_seconds = time.monotonic() - start_time
        if (
            config.max_runtime_seconds is not None
            and elapsed_seconds >= config.max_runtime_seconds
        ):
            runtime_timeout = True
            print(
                "\nRuntime limit reached before starting the next iteration: "
                f"{elapsed_seconds / 3600:.2f} hours elapsed. "
                "Stopping loop and plotting current history."
            )
            break

        if config.alpha_mode == "fixed":
            alpha = config.alpha

        alpha_used = alpha
        rho_used = rho
        print(f"\n--- Outer Iteration {k} ---")
        print(f"alpha = {alpha_used:.6e}, rho = {rho_used:.6e}")

        lagrangian, variable_list, var_bound_list = model.build_linear_ALM_Lagrangian_syms(
            x_center=xk,
            rho=rho_used,
            ref_bus_id=None,
            mu_prox=config.mu_prox,
        )
        coarse_var_bound_list = qhd_var_bound_list if config.option == 1 else var_bound_list
        validate_bounds(variable_list, coarse_var_bound_list)
        validate_bounds(variable_list, model.Var_bound_list)

        if config.option == 1:
            x_coarse, x_new = solve_subproblem_qhd(
                lagrangian,
                variable_list,
                coarse_var_bound_list,
                model=model,
                x_center=xk,
                rho=rho_used,
                config=config,
                refine_var_bound_list=model.Var_bound_list,
            )
        else:
            x_coarse = solve_subproblem_gurobi(lagrangian, variable_list, coarse_var_bound_list)
            x_new = refine_acopf_solution(
                model,
                x_coarse,
                x_center=xk,
                rho=rho_used,
                config=config,
        )

        coarse_h_val = h_func(x_coarse)
        if previous_coarse is not None and coarse_solution_unchanged(
            x_current=x_coarse,
            x_previous=previous_coarse,
            ignored_indices=fixed_bound_indices,
            atol=config.coarse_repeat_atol,
        ):
            coarse_repeat_count += 1
        else:
            coarse_repeat_count = 1
        previous_coarse = np.asarray(x_coarse, dtype=float).copy()
        should_shrink_bounds = (
            config.option == 1
            and coarse_repeat_count >= config.coarse_repeat_limit
        )

        coarse_norm_h = float(np.linalg.norm(coarse_h_val))
        coarse_objective_value = evaluate_objective(model, x_coarse)
        coarse_lalm_energy = evaluate_sympy_expression(lagrangian, variable_list, x_coarse)
        _, coarse_check_flag = model.check_constraints(x_coarse)
        print(f"[coarse:LALM] ||h(x)|| = {coarse_norm_h:.6e}")
        print(f"[coarse:LALM] objective = {coarse_objective_value:.9g}")
        print(f"[coarse:LALM] actual energy = {coarse_lalm_energy:.12g}")
        print(
            f"[coarse:LALM] repeated coarse count (excluding ref V) = "
            f"{coarse_repeat_count}/{config.coarse_repeat_limit}"
        )

        h_val = h_func(x_new)
        norm_h = float(np.linalg.norm(h_val))
        print(f"[refined:{canonical_refine_method(config.refine_method)}] ||h(x)|| = {norm_h:.6e}")

        lambda_new, h_x = model.update_lambda(x_new, alpha=alpha_used, h_func=h_func)
        h_old = h_func(xk)
        lambda_inf = float(np.max(np.abs(lambda_new)))
        print(f"[rho-check] ||h_old||={np.linalg.norm(h_old):.3e}, ||h_new||={norm_h:.3e}, rho={rho_used:.3g}")

        alpha, rho, stable_count, plateau_count, worsen_count = adapt_alpha_rho(
            norm_h=norm_h,
            lambda_inf=lambda_inf,
            prev_norm_h=prev_norm_h,
            prev_lambda_inf=prev_lambda_inf,
            alpha=alpha_used,
            rho=rho_used,
            stable_count=stable_count,
            plateau_count=plateau_count,
            worsen_count=worsen_count,
            config=config,
            alpha_max=alpha_max_current,
        )

        prev_norm_h = norm_h
        prev_lambda_inf = lambda_inf
        print(
            f"[adaptive] next alpha={alpha:.6e}, next rho={rho:.6e}, "
            f"alpha_max={alpha_max_current:.6e}, lambda_inf={lambda_inf:.3e}"
        )

        _, check_flag = model.check_constraints(x_new)
        print("Constraint check:", check_flag)

        objective_value = evaluate_objective(model, x_new)
        objective_history.append({"iter": k, "objective": objective_value})
        metric_history.append(
            {
                "iter": k,
                "max_abs_h": float(np.max(np.abs(h_val))),
                "l2_norm_h": norm_h,
                "coarse_l2_norm_h": coarse_norm_h,
                "coarse_max_abs_h": float(np.max(np.abs(coarse_h_val))),
                "coarse_objective": coarse_objective_value,
                "coarse_lalm_energy": coarse_lalm_energy,
                "refined_objective": objective_value,
                "refine_method": canonical_refine_method(config.refine_method),
                "alpha": float(alpha_used),
                "rho": float(rho_used),
                "next_alpha": float(alpha),
                "next_rho": float(rho),
                "alpha_max": float(alpha_max_current),
                "lambda_inf": lambda_inf,
                "coarse_repeat_count": coarse_repeat_count,
                "bounds_shrink_count": bounds_shrink_count,
                "rho_after_bound_shrink": None,
                "alpha_max_after_bound_shrink": None,
            }
        )

        if norm_h < best_record["metric"] - 1e-12:
            best_record.update(
                {
                    "iter": k,
                    "metric": norm_h,
                    "x": x_new.copy(),
                    "objective": objective_value,
                    "h_x": np.asarray(h_val, dtype=float).copy(),
                    "lambda_vec": np.asarray(lambda_new, dtype=float).copy(),
                    "feasible": check_flag,
                    "rho": float(rho_used),
                    "alpha": float(alpha_used),
                }
            )
            print(
                f"[best] iter={k}, l2_norm_h={norm_h:.6e}, "
                f"objective={objective_value:.9g}"
            )

        log_file = PrintQHDACOPFResults(
            model,
            x_coarse,
            log_file=log_file,
            iteration=k,
            folder=config.log_folder,
            print_to_console=config.print_to_console,
            rho=rho_used,
            alpha=alpha_used,
            h_x=coarse_h_val,
            lambda_vec=None,
            objective_value=coarse_objective_value,
            lalm_energy=coarse_lalm_energy,
            feasibility=coarse_check_flag,
            note="coarse_solution_before_refine",
        )

        log_file = PrintQHDACOPFResults(
            model,
            x_new,
            log_file=log_file,
            iteration=k,
            folder=config.log_folder,
            print_to_console=config.print_to_console,
            rho=rho_used,
            alpha=alpha_used,
            h_x=h_val,
            lambda_vec=lambda_new,
            objective_value=objective_value,
            feasibility=check_flag,
            note=f"refined_solution_{canonical_refine_method(config.refine_method)}",
        )

        if should_shrink_bounds:
            bounds_shrink_count += 1
            current_var_bound_list = [[float(b[0]), float(b[1])] for b in qhd_var_bound_list]
            qhd_var_bound_list = shrink_bounds_around_refined_solution(
                original_var_bound_list,
                current_var_bound_list,
                x_new,
                shrink_factor=config.bound_shrink_factor,
                min_shrink_factor=config.bound_shrink_min_factor,
                fixed_indices=fixed_bound_indices,
            )
            validate_bounds(model.variable_list, qhd_var_bound_list)
            rho_before_shrink = rho
            rho = max(
                float(config.rho_min),
                float(rho) * float(config.bound_shrink_factor),
            )
            alpha_max_before_shrink = alpha_max_current
            alpha_max_current = max(
                float(config.alpha_min),
                float(alpha_max_current) * float(config.bound_shrink_factor),
            )
            alpha = min(alpha, alpha_max_current)
            metric_history[-1]["rho_after_bound_shrink"] = float(rho)
            metric_history[-1]["alpha_max_after_bound_shrink"] = float(alpha_max_current)
            effective_cumulative_shrink = max(
                float(config.bound_shrink_min_factor),
                float(config.bound_shrink_factor) ** bounds_shrink_count,
            )
            status = (
                f"[bounds] coarse answer unchanged for {config.coarse_repeat_limit} iterations "
                "(excluding reference-bus V_R/V_I); "
                f"shrink #{bounds_shrink_count}: using "
                f"{config.bound_shrink_factor:.6g} of current widths around refined solution "
                f"(cumulative factor ~= {effective_cumulative_shrink:.6g}, "
                f"min {config.bound_shrink_min_factor:.6g}); "
                "(QHD bounds only; refine/model bounds unchanged); "
                f"rho {rho_before_shrink:.6g} -> {rho:.6g}; "
                f"alpha_max {alpha_max_before_shrink:.6g} -> {alpha_max_current:.6g}; "
                f"next alpha clamped to {alpha:.6g}."
            )
            print(status)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(status + "\n")
            previous_coarse = None
            coarse_repeat_count = 0

        elapsed_seconds = time.monotonic() - start_time
        if (
            config.max_runtime_seconds is not None
            and elapsed_seconds >= config.max_runtime_seconds
        ):
            runtime_timeout = True
            print(
                "\nRuntime limit reached after this iteration: "
                f"{elapsed_seconds / 3600:.2f} hours elapsed. "
                "Stopping loop and plotting current history."
            )
            xk = (
                best_record["x"].copy()
                if config.return_best_solution and best_record["x"] is not None
                else x_new.copy()
            )
            break

        if check_flag:
            print("\nConverged!")
            xk = x_new.copy()
            break

        step_norm = float(np.linalg.norm(x_new - xk))
        if norm_h < config.tol and step_norm < 1e-5:
            print("\nConverged!")
            xk = x_new.copy()
            break

        if (
            config.early_stop_patience > 0
            and best_record["iter"] is not None
            and k - best_record["iter"] >= config.early_stop_patience
        ):
            print(
                "\nEarly stop: no residual improvement for "
                f"{config.early_stop_patience} iterations."
            )
            xk = (
                best_record["x"].copy()
                if config.return_best_solution and best_record["x"] is not None
                else x_new.copy()
            )
            break

        xk = x_new.copy()

    print("\n===== End Loop =====\n")
    if runtime_timeout:
        print("Stopped because max_runtime_seconds was reached.")
    print("Final log file:", log_file)
    if best_record["iter"] is not None:
        print(
            "Best residual iteration:",
            best_record["iter"],
            f"objective={best_record['objective']:.9g}",
            f"l2_norm_h={best_record['metric']:.6e}",
            f"rho={best_record['rho']:.6g}",
            f"alpha={best_record['alpha']:.6g}",
        )
        log_file = PrintQHDACOPFResults(
            model,
            best_record["x"],
            log_file=log_file,
            iteration=best_record["iter"],
            folder=config.log_folder,
            print_to_console=False,
            rho=best_record["rho"],
            alpha=best_record["alpha"],
            h_x=best_record["h_x"],
            lambda_vec=best_record["lambda_vec"],
            objective_value=best_record["objective"],
            feasibility=best_record["feasible"],
            note="best_iteration_by_l2_norm_h",
        )
        if config.return_best_solution:
            xk = best_record["x"].copy()
    save_objective_plot(
        objective_history,
        log_file,
        config.log_folder,
        show=config.show_plot,
    )

    return {
        "x": xk,
        "log_file": log_file,
        "objective_history": objective_history,
        "metric_history": metric_history,
        "best_record": best_record,
        "final_alpha": alpha,
        "final_rho": rho,
    }



def run_linear_alm(model: SympyACOPFModel, config: SolverConfig):
    if config.coarse_beam_search:
        return run_coarse_beam_search_alm(model, config)
    return run_single_point_linear_alm(model, config)

def main():
    config = SolverConfig()
    model = build_model(config.n_bus)
    ref_bus_id = apply_reference_voltage_bounds(model)
    print(
        "Model initialized with",
        config.n_bus,
        "buses",
        model.n_lines,
        "lines and",
        model.n_gens,
        "generators.",
    )
    print(
        f"Reference bus {ref_bus_id} bounds: "
        f"V_R in [{REF_VR_BOUND[0]}, {REF_VR_BOUND[1]}], "
        f"V_I in [{REF_VI_BOUND[0]}, {REF_VI_BOUND[1]}]"
    )
    run_linear_alm(model, config)


if __name__ == "__main__":
    main()
