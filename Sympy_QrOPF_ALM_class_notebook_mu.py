#!/usr/bin/env python
# coding: utf-8

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

try:
    from qhdopt import QHD
except ImportError:
    QHD = None

from Sympy_QrOPF_ALM_class_notebook_mu_deps import (
    PrintQHDACOPFResults,
    SympyACOPFModel,
    extract_qhd_solution_vector,
    initialize_qhd_acopf_log,
    solve_with_gurobi_from_sympy,
)


@dataclass
class SolverConfig:
    n_bus: int = 2
    max_outer: int = 900
    tol: float = 1e-4
    option: int = 2  # 1: QHD, 2: Gurobi
    qhd_solver: str = "simbi"  # simbi / openjij / gurobi
    rho: float = 80.0
    alpha: float = 5.0
    mu_prox: float = 2e-2
    alpha_mode: str = "adaptive"  # adaptive / fixed
    alpha_min: float = 1e-2
    alpha_max: float = 8.0
    rho_min: float = 20.0
    rho_max: float = 640.0
    plateau_window: int = 4
    worsen_window: int = 2
    stable_window: int = 4
    improve_tol: float = 0.02
    worsen_tol: float = 0.03
    qhd_refine: bool = True
    simbi_resolution: int = 14
    simbi_shots: int = 128
    simbi_agents: int = 4096
    simbi_max_steps: int = 1300
    simbi_seed: int | None = 42
    simbi_best_only: bool = False
    simbi_ballistic: bool = False
    simbi_heated: bool | None = None
    early_stop_patience: int = 300
    return_best_solution: bool = True
    print_to_console: bool = True
    show_plot: bool = True
    log_folder: str = "logs"
    pg_zero_warn_tol: float = 1e-8


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

    pretty_file = Path(f"case{n_bus}_custom_pretty.json")
    compact_file = Path(f"case{n_bus}_custom.json")
    json_file = pretty_file if pretty_file.exists() else compact_file
    sbase, buses, lines, gens = load_matpower_json(str(json_file))
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
    if config.option == 1 and QHD is None:
        raise ImportError("qhdopt is required when option=1.")
    if config.qhd_solver not in {"simbi", "openjij", "gurobi"}:
        raise ValueError("qhd_solver must be 'simbi', 'openjij', or 'gurobi'.")
    if config.max_outer <= 0:
        raise ValueError("max_outer must be positive.")
    if config.mu_prox < 0:
        raise ValueError("mu_prox must be nonnegative.")
    if config.simbi_resolution <= 0:
        raise ValueError("simbi_resolution must be positive.")
    if config.simbi_shots <= 0:
        raise ValueError("simbi_shots must be positive.")


def validate_bounds(variable_list, var_bound_list) -> None:
    if len(variable_list) != len(var_bound_list):
        raise ValueError(
            f"Bounds length mismatch: {len(variable_list)} variables, "
            f"{len(var_bound_list)} bounds."
        )

    bad_bounds = []
    for i, (var, bnd) in enumerate(zip(variable_list, var_bound_list)):
        lb, ub = float(bnd[0]), float(bnd[1])
        if ub < lb:
            bad_bounds.append((i, str(var), lb, ub))

    if bad_bounds:
        for item in bad_bounds:
            print("Invalid bound:", item)
        raise ValueError("var_bound_list contains invalid bounds (ub < lb).")


def solve_subproblem_qhd(lagrangian, variable_list, var_bound_list, config: SolverConfig):
    if QHD is None:
        raise ImportError("qhdopt is required for QHD subproblem solving.")

    qhd_model = QHD.SymPy(lagrangian, variable_list, var_bound_list)

    if config.qhd_solver == "simbi":
        qhd_model.simbi_setup(
            resolution=config.simbi_resolution,
            shots=config.simbi_shots,
            agents=config.simbi_agents,
            max_steps=config.simbi_max_steps,
            embedding_scheme="unary",
            post_processing_method="TNC",
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
            post_processing_method="TNC",
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
            post_processing_method="TNC",
        )
    else:
        raise ValueError(f"Unsupported qhd_solver={config.qhd_solver!r}.")

    response = qhd_model.optimize(refine=config.qhd_refine, verbose=0)
    return extract_qhd_solution_vector(response, expected_len=len(variable_list))


def solve_subproblem_gurobi(lagrangian, variable_list, var_bound_list):
    return solve_with_gurobi_from_sympy(
        L_sym=lagrangian,
        variable_list=variable_list,
        Var_bound_list=var_bound_list,
        verbose=False,
    )


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
                rho = min(rho * 2.0, config.rho_max)
                alpha = max(alpha * 0.9, config.alpha_min)
                plateau_count = 0
                print(f"[adaptive] Plateau detected, rho -> {rho:.6e}, alpha -> {alpha:.6e}")

    if prev_lambda_inf is not None and prev_lambda_inf > 1e-10 and lambda_inf > 1.4 * prev_lambda_inf:
        alpha = max(alpha * 0.85, config.alpha_min)
        stable_count = 0
        worsen_count = 0
        print(f"[adaptive] Lambda growth too fast, alpha -> {alpha:.6e}")

    rho = min(max(rho, config.rho_min), config.rho_max)
    alpha = min(max(alpha, config.alpha_min), config.alpha_max)
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


def evaluate_objective(model: SympyACOPFModel, x_vec) -> float:
    subs_dict = {var: val for var, val in zip(model.variable_list, x_vec)}
    return float(sp.N(model.objective.subs(subs_dict)))


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
    if config.alpha_mode == "fixed":
        print(f"Fixed alpha={alpha:.6e}, rho={rho:.6e}, mu_prox={config.mu_prox:.6e}")
    else:
        print(
            "Adaptive alpha/rho start: "
            f"alpha={alpha:.6e}, rho={rho:.6e}, mu_prox={config.mu_prox:.6e}"
        )

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
        "max_abs_h": float("inf"),
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
            rho = config.rho

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
        validate_bounds(variable_list, var_bound_list)

        if config.option == 1:
            x_new = solve_subproblem_qhd(lagrangian, variable_list, var_bound_list, config)
        else:
            x_new = solve_subproblem_gurobi(lagrangian, variable_list, var_bound_list)

        h_val = h_func(x_new)
        vals_new = model._unpack_x(x_new)
        pg_values = np.asarray(vals_new["P_G"], dtype=float)
        total_pg = float(np.sum(pg_values))
        total_load_p = float(np.sum(model.P_D))
        max_abs_h = float(np.max(np.abs(h_val)))
        norm_h = float(np.linalg.norm(h_val))
        print(f"total Pg = {total_pg:.6e}, total load P = {total_load_p:.6e}")
        print(f"max |h(x)| = {max_abs_h:.6e}")
        print(f"||h(x)|| = {norm_h:.6e}")
        if total_load_p > config.pg_zero_warn_tol and np.max(np.abs(pg_values)) <= config.pg_zero_warn_tol:
            print(
                "[warning] All Pg values are still at zero. "
                "Increase rho/alpha or use alpha_mode='adaptive' so the ALM penalty can overcome generation cost."
            )

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
        )

        prev_norm_h = norm_h
        prev_lambda_inf = lambda_inf
        if config.alpha_mode == "adaptive":
            print(f"[adaptive] next alpha={alpha:.6e}, next rho={rho:.6e}, lambda_inf={lambda_inf:.3e}")

        _, check_flag = model.check_constraints(x_new)
        print("Constraint check:", check_flag)

        objective_value = evaluate_objective(model, x_new)
        objective_history.append({"iter": k, "objective": objective_value})
        metric_history.append(
            {
                "iter": k,
                "max_abs_h": max_abs_h,
                "l2_norm_h": norm_h,
                "alpha": float(alpha_used),
                "rho": float(rho_used),
                "next_alpha": float(alpha),
                "next_rho": float(rho),
                "lambda_inf": lambda_inf,
                "total_pg": total_pg,
                "total_load_p": total_load_p,
            }
        )

        if norm_h < best_record["metric"] - 1e-12:
            best_record.update(
                {
                    "iter": k,
                    "metric": norm_h,
                    "max_abs_h": max_abs_h,
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
                f"[best] iter={k}, max_abs_h={max_abs_h:.6e}, "
                f"l2_norm_h={norm_h:.6e}, objective={objective_value:.9g}"
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
        )

        xk = x_new.copy()

        if max_abs_h < config.tol:
            print("\nConverged.")
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
            if config.return_best_solution and best_record["x"] is not None:
                xk = best_record["x"].copy()
            break

    print("\n===== End Loop =====\n")
    print("Final log file:", log_file)

    if best_record["iter"] is not None:
        print(
            "Best residual iteration:",
            best_record["iter"],
            f"objective={best_record['objective']:.9g}",
            f"max_abs_h={best_record['max_abs_h']:.6e}",
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
