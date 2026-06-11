#!/usr/bin/env python
# coding: utf-8

import json
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
    n_bus: int = 14
    max_outer: int = 2000
    tol: float = 1e-4
    option: int = 1  # 1: QHD, 2: Gurobi
    qhd_solver: str = "simbi"  # simbi / openjij / gurobi
    refine_method: str = "TNC_orig"  # none / ipopt_orig / TNC_orig / GurobiALM / GurobiOrig
    rho: float = 80.0
    alpha: float = 5.0
    mu_prox: float = 1 #2e-2
    alpha_mode: str = "adaptive"  # adaptive / fixed
    alpha_min: float = 1e-2
    alpha_max: float = 8.0
    rho_min: float = 20.0
    rho_max: float = 120.0
    plateau_window: int = 4
    worsen_window: int = 2
    stable_window: int = 4
    improve_tol: float = 0.005
    worsen_tol: float = 0.03
    qhd_refine: bool = True
    simbi_resolution: int = 16
    simbi_shots: int = 128
    simbi_agents: int = 4096
    simbi_max_steps: int = 42000
    simbi_seed: int | None = 42
    simbi_best_only: bool = False
    simbi_ballistic: bool = False
    simbi_heated: bool | None = None
    early_stop_patience: int = 300
    tnc_maxfun: int | None = None
    ipopt_max_iter: int = 350
    gurobi_time_limit: float | None = 60.0
    gurobi_threads: int = 0
    gurobi_log_to_console: bool = False
    return_best_solution: bool = True
    print_to_console: bool = True
    show_plot: bool = True
    log_folder: str = "logs"


REFINE_METHODS = {"none", "ipopt_orig", "TNC_orig", "GurobiALM", "GurobiOrig"}
REFINE_METHOD_ALIASES = {method.lower(): method for method in REFINE_METHODS}


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
):
    qhd_model = QHD.SymPy(lagrangian, variable_list, var_bound_list)
    refine_method = canonical_refine_method(config.refine_method)
    qhd_post_processing_method = "TNC" if refine_method == "none" else refine_method

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
            multi_gpu=config.simbi_multi_gpu,
            num_gpus=config.simbi_num_gpus,
            gpu_ids=config.simbi_gpu_ids,
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

    should_refine = bool(config.qhd_refine and refine_method != "none")
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
        )

    response = qhd_model.optimize(refine=should_refine, verbose=0)
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
):
    if config.alpha_mode != "adaptive":
        return alpha, rho, stable_count, plateau_count, worsen_count

    if prev_norm_h is not None:
        ratio = norm_h / max(prev_norm_h, 1e-12)

        if ratio <= 1.0 - config.improve_tol:
            stable_count += 1
            plateau_count = 0
            worsen_count = 0
            if stable_count >= config.stable_window:
                alpha = min(alpha * 1.03, config.alpha_max)
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


def run_linear_alm(model: SympyACOPFModel, config: SolverConfig):
    validate_config(config)

    h_func = model.build_h_func()
    model.reset_lambdas(0.0)
    xk = model.build_initial_x0()

    alpha = config.alpha
    rho = config.rho
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
    objective_history = []
    metric_history = []
    plateau_count = 0
    worsen_count = 0
    stable_count = 0
    prev_norm_h = None
    prev_lambda_inf = None
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
        if config.alpha_mode == "fixed":
            alpha = config.alpha

        print(f"\n--- Outer Iteration {k} ---")
        print(f"alpha = {alpha:.6e}, rho = {rho:.6e}")

        lagrangian, variable_list, var_bound_list = model.build_linear_ALM_Lagrangian_syms(
            x_center=xk,
            rho=rho,
            ref_bus_id=None,
            mu_prox=config.mu_prox,
        )
        validate_bounds(variable_list, var_bound_list)

        if config.option == 1:
            x_coarse, x_new = solve_subproblem_qhd(
                lagrangian,
                variable_list,
                var_bound_list,
                model=model,
                x_center=xk,
                rho=rho,
                config=config,
            )
        else:
            x_coarse = solve_subproblem_gurobi(lagrangian, variable_list, var_bound_list)
            x_new = refine_acopf_solution(
                model,
                x_coarse,
                x_center=xk,
                rho=rho,
                config=config,
            )

        coarse_h_val = h_func(x_coarse)
        coarse_norm_h = float(np.linalg.norm(coarse_h_val))
        coarse_objective_value = evaluate_objective(model, x_coarse)
        _, coarse_check_flag = model.check_constraints(x_coarse)
        print(f"[coarse:LALM] ||h(x)|| = {coarse_norm_h:.6e}")
        print(f"[coarse:LALM] objective = {coarse_objective_value:.9g}")

        h_val = h_func(x_new)
        norm_h = float(np.linalg.norm(h_val))
        print(f"[refined:{canonical_refine_method(config.refine_method)}] ||h(x)|| = {norm_h:.6e}")

        lambda_new, h_x = model.update_lambda(x_new, alpha=alpha, h_func=h_func)
        h_old = h_func(xk)
        lambda_inf = float(np.max(np.abs(lambda_new)))
        print(f"[rho-check] ||h_old||={np.linalg.norm(h_old):.3e}, ||h_new||={norm_h:.3e}, rho={rho:.3g}")

        alpha, rho, stable_count, plateau_count, worsen_count = adapt_alpha_rho(
            norm_h=norm_h,
            lambda_inf=lambda_inf,
            prev_norm_h=prev_norm_h,
            prev_lambda_inf=prev_lambda_inf,
            alpha=alpha,
            rho=rho,
            stable_count=stable_count,
            plateau_count=plateau_count,
            worsen_count=worsen_count,
            config=config,
        )

        prev_norm_h = norm_h
        prev_lambda_inf = lambda_inf
        print(f"[adaptive] next alpha={alpha:.6e}, next rho={rho:.6e}, lambda_inf={lambda_inf:.3e}")

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
                "refined_objective": objective_value,
                "refine_method": canonical_refine_method(config.refine_method),
                "alpha": float(alpha),
                "rho": float(rho),
                "lambda_inf": lambda_inf,
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
                    "rho": float(rho),
                    "alpha": float(alpha),
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
            print_to_console=False,
            rho=rho,
            alpha=alpha,
            h_x=coarse_h_val,
            lambda_vec=None,
            objective_value=coarse_objective_value,
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
            rho=rho,
            alpha=alpha,
            h_x=h_val,
            lambda_vec=lambda_new,
            objective_value=objective_value,
            feasibility=check_flag,
            note=f"refined_solution_{canonical_refine_method(config.refine_method)}",
        )

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


def main():
    config = SolverConfig()
    model = build_model(config.n_bus)
    print(
        "Model initialized with",
        config.n_bus,
        "buses",
        model.n_lines,
        "lines and",
        model.n_gens,
        "generators.",
    )
    run_linear_alm(model, config)


if __name__ == "__main__":
    main()