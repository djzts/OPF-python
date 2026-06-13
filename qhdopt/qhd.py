import time
import warnings
from typing import List, Tuple, Union, Optional, Callable
import jax.numpy as jnp
import numpy as np
import sympy
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from sympy import lambdify
from sympy.core.function import Function
from sympy.core.symbol import Symbol
from qhdopt.backend.backend import Backend
from qhdopt.qhd_base import QHD_Base
from qhdopt.backend import dwave_backend
from qhdopt.backend import openjij_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import gen_new_func_with_affine_trans, \
    generate_bounds, quad_to_gen


ACOPF_REFINE_METHODS = {"ipopt_orig", "TNC_orig", "GurobiALM", "GurobiOrig"}
ACOPF_REFINE_METHOD_ALIASES = {method.lower(): method for method in ACOPF_REFINE_METHODS}


def canonical_acopf_refine_method(method: str) -> str:
    key = str(method).lower()
    if key not in ACOPF_REFINE_METHOD_ALIASES:
        allowed = ", ".join(sorted(ACOPF_REFINE_METHODS))
        raise ValueError(f"ACOPF refine method must be one of: {allowed}.")
    return ACOPF_REFINE_METHOD_ALIASES[key]


class QHD:
    """
    Provides functionality to run Quantum Hamiltonian Gradient Descent as introduced
    by https://arxiv.org/pdf/2303.01471.pdf

    A user should initialize QHD through the use of the functions: QHD.QP and QHD.Sympy
    """

    def __init__(
            self,
            func: Function,
            syms: List[Symbol],
            bounds: Union[Tuple, List, None] = None,
    ):
        self.qubits = None
        self.qs = None
        self.raw_result = None
        self.decoded_samples = None
        self.post_processed_samples = None
        self.acopf_refine_problem = None
        self.info = dict()
        self.syms = syms
        self.syms_index = {syms[i]: i for i in range(len(syms))}
        self.func = func
        self.bounds = bounds
        self.lambda_numpy = lambdify(syms, func, jnp)
        self.dimension = len(syms)
        if len(syms) != len(func.free_symbols):
            warnings.warn("The number of function free symbols does not match the number of syms.",
                          RuntimeWarning)

    def generate_affined_func(self) -> Tuple[Function, List[Symbol]]:
        """
        Internal method for generating a new Sympy function with an
        affine transformation which calculated from the bounds property
        inputted from the user
        """
        self.lb, self.scaling_factor = generate_bounds(self.bounds, self.dimension)
        func, syms = gen_new_func_with_affine_trans(self.affine_transformation, self.func,
                                                    self.syms)
        return func, syms

    @classmethod
    def SymPy(cls, func: Function, syms: List[Symbol],
              bounds: Union[Tuple, List, None] = None) -> 'QHD':
        """
        Initialize QHD with a sympy function and its symbols

        Args:
            func: Sympy function
            syms: List of symbols
            bounds: Bounds of the function

        Returns:
            QHD: An instance of QHD
        """
        return cls(func, syms, bounds)

    @classmethod
    def QP(cls, Q: List[List[float]], b: List[float],
           bounds: Union[Tuple, List, None] = None) -> 'QHD':
        """
        Initialize QHD with a quadratic programming format

        Args:
            Q: Quadratic matrix
            b: Linear vector
            bounds: Bounds of the function

        Returns:
            QHD: An instance of QHD
        """
        f, xl = quad_to_gen(Q, b)
        return cls(f, xl, bounds)

    def dwave_setup(
            self,
            resolution: int,
            shots: int = 100,
            api_key: Optional[str] = None,
            api_key_from_file: Optional[str] = None,
            embedding_scheme: str = "unary",
            anneal_schedule: Optional[List[List[int]]] = None,
            penalty_coefficient: float = 0,
            penalty_ratio: float = 0.75,
            chain_strength_ratio: float = 1.05,
            post_processing_method: str = "TNC",
    ):
        """
        Configures the settings for quantum optimization using D-Wave systems.

        Args:
            resolution: The number of bits representing each variable.
            shots: The number of times the quantum device runs to find the solution.
            api_key: Direct API key for connecting to D-Wave's cloud service.
            api_key_from_file: Path to a file containing the D-Wave API key.
            embedding_scheme: Method used for mapping logical variables to physical qubits.
            anneal_schedule: Custom annealing schedule for quantum annealing.
            penalty_coefficient: Coefficient used to enforce constraints in the quantum model.
            penalty_ratio: Ratio used to calculate penalty coefficients.
            post_processing_method: Classical optimization method used after quantum sampling.
            chain_strength_ratio: Ratio of strength of chains in embedding.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.dwave_setup(
            resolution=resolution,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            chain_strength_ratio=chain_strength_ratio,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def ionq_setup(
            self,
            resolution: int,
            shots: int = 100,
            api_key: Optional[str] = None,
            api_key_from_file: Optional[str] = None,
            embedding_scheme: str = "onehot",
            penalty_coefficient: float = 0,
            time_discretization: int = 10,
            gamma: float = 5,
            post_processing_method: str = "TNC",
            on_simulator: bool = False,
    ):
        """
        Configures the settings for running QHD using IonQ systems.

        Args:
            resolution: The resolution of the quantum representation.
            shots: Number of measurements to perform on the quantum state.
            api_key: API key for accessing IonQ's quantum computing service.
            api_key_from_file: Path to file containing the API key for IonQ.
            embedding_scheme: Strategy for encoding problem variables into quantum states.
            penalty_coefficient: Multiplier for penalty terms in the quantum formulation.
            time_discretization: Number of time steps for simulating quantum evolution.
            gamma: Scaling factor for the quantum evolution's time discretization.
            post_processing_method: Algorithm for refining quantum results classically.
            on_simulator: Flag to indicate if the quantum simulation should run on a simulator.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.ionq_setup(
            resolution=resolution,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
            on_simulator=on_simulator
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def qutip_setup(
            self,
            resolution: int,
            shots: int = 100,
            embedding_scheme: str = "onehot",
            penalty_coefficient: float = 0,
            time_discretization: int = 10,
            gamma: float = 5,
            post_processing_method: str = "TNC",
    ):
        """
        Configures the settings for quantum simulation of QHD using QuTiP.

        Args:
            resolution: The resolution for encoding variables into quantum states.
            shots: Number of repetitions for the quantum state measurement.
            embedding_scheme: Encoding strategy for representing problem variables.
            penalty_coefficient: Coefficient for penalties in the quantum problem formulation.
            time_discretization: Number of intervals for the quantum evolution simulation.
            gamma: Parameter controlling the strength of the quantum system evolution.
            post_processing_method: Classical method used for post-processing quantum results.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.qutip_setup(
            resolution=resolution,
            shots=shots,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def openjij_setup(
        self,
        resolution: int,
        shots: int = 100,
        sampler_name: str = "SQASampler",
        seed: Optional[int] = None,
        debug: bool = False,
        post_processing_method: str = "TNC",
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        embedding_scheme: str = "unary",
        sampler_init_kwargs: Optional[dict] = None,
        sample_kwargs: Optional[dict] = None,
    ):
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)

        if sampler_init_kwargs is None:
            sampler_init_kwargs = {}
        if sample_kwargs is None:
            sample_kwargs = {}

        self.qhd_base.openjij_setup(
            resolution=resolution,
            shots=shots,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            sampler_name=sampler_name,
            seed=seed,
            debug=debug,
            sampler_init_kwargs=sampler_init_kwargs,
            sample_kwargs=sample_kwargs,
        )

        self.shots = shots
        self.post_processing_method = post_processing_method

        # 如果你希望 OpenJij 也支持 penalty/embedding_scheme（像 DWaveBackend 一样）
        # 需要把 QHD_Base.openjij_setup 和 OpenJijBackend.__init__ 参数对齐（见下方第 3 点）

    def simbi_setup(
            self,
            resolution: int,
            shots: int = 100,
            embedding_scheme: str = "unary",
            anneal_schedule: Optional[List[List[int]]] = None,
            penalty_coefficient: float = 0.0,
            penalty_ratio: float = 0.75,
            chain_strength_ratio: float = 1.05,
            api_key='DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291',
            # ---- SB params ----
            domain: str = "spin",
            device: str = "cuda",                # "cpu" / "cuda" / "cuda:0"
            agents: Optional[int] = None,
            max_steps: Optional[int] = 2000,
            ballistic: Optional[bool] = False,   # "ballistic" / "discrete"
            heated: Optional[bool] = None,
            best_only: bool = False,
            seed: Optional[int] = None,
            multi_gpu: bool = False,
            num_gpus: Optional[int] = None,
            gpu_ids: Optional[List[int]] = None,
            post_processing_method: str = "TNC",
            **sb_kwargs,
    ):
        """
        Configures the settings for local optimization using Simulated Bifurcation (SB).
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)

        # NOTE: QHD_Base 里目前叫 sb_setup（不是 simbi_setup）
        self.qhd_base.sb_setup(
            resolution=resolution,
            shots=shots,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            chain_strength_ratio=chain_strength_ratio,
            api_key=api_key,
            domain=domain,
            device=device,
            agents=agents,
            max_steps=max_steps,
            ballistic=ballistic,
            heated=heated,
            best_only=best_only,
            seed=seed,
            multi_gpu=multi_gpu,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            **sb_kwargs,
        )

        self.shots = shots
        self.post_processing_method = post_processing_method

    def gurobi_setup(
            self,
            resolution: int,
            shots: int = 100,
            embedding_scheme: str = "unary",
            anneal_schedule: Optional[List[List[int]]] = None,
            penalty_coefficient: float = 0.0,
            penalty_ratio: float = 0.75,
            chain_strength: float = 1.0,
            api_key= 'DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291',
            api_key_from_file: Optional[str] = None,
            solver_mode: str = "ising",
            time_limit: Optional[float] = None,
            mip_gap: Optional[float] = None,
            threads: Optional[int] = None,
            log_to_console: bool = False,
            post_processing_method: str = "TNC",
    ):
        """
        Configures the settings for local optimization using Gurobi.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.gurobi_setup(
            resolution=resolution,
            shots=shots,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            chain_strength=chain_strength,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            solver_mode=solver_mode,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_to_console=log_to_console,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def affine_transformation(self, x: np.ndarray) -> np.ndarray:
        """
        Applies an affine transformation to the input array.

        Args:
            x: The input array to be transformed.

        Returns:
            Transformed array according to the defined scaling factor and lower bounds.
        """
        return self.scaling_factor * x + self.lb

    def fun_eval(self, x: np.ndarray):
        """
        function evaluation when x is in the original box (non-normalized)
        """
        x = x.astype(jnp.float32)
        return self.lambda_numpy(*x)


    def set_acopf_refine_problem(
            self,
            objective: Optional[Function] = None,
            constraints: Optional[List[Function]] = None,
            lambda_vec: Optional[Union[List[float], np.ndarray]] = None,
            rho: float = 0.0,
            x_center: Optional[Union[List[float], np.ndarray]] = None,
            mu_prox: float = 0.0,
            best_only: bool = True,
            tnc_options: Optional[dict] = None,
            ipopt_options: Optional[dict] = None,
            gurobi_options: Optional[dict] = None,
            refine_bounds: Optional[Union[Tuple, List, Bounds]] = None,
    ):
        """
        Configure a full ACOPF refinement problem for post-processing.

        The QHD coarse solve still optimizes self.func. The refinement methods
        below can instead use the original ACOPF objective and constraints:
        - ipopt_orig: original objective with equality constraints
        - TNC_orig: full ACOPF ALM objective with box bounds
        - GurobiALM: full ACOPF ALM using Gurobi nonconvex QCQP form
        - GurobiOrig: original objective with equality constraints using Gurobi QCQP
        """
        if constraints is None:
            constraints = []
        constraints = list(constraints)
        if objective is None:
            objective = self.func

        if lambda_vec is None:
            lambda_vec = np.zeros(len(constraints), dtype=float)
        lambda_vec = np.asarray(lambda_vec, dtype=float).reshape(-1)
        if lambda_vec.size != len(constraints):
            raise ValueError(
                f"lambda_vec length {lambda_vec.size} does not match "
                f"number of constraints {len(constraints)}."
            )

        if x_center is not None:
            x_center = np.asarray(x_center, dtype=float).reshape(-1)
            if x_center.size != self.dimension:
                raise ValueError(
                    f"x_center length {x_center.size} does not match dimension {self.dimension}."
                )

        if refine_bounds is not None:
            if isinstance(refine_bounds, Bounds):
                refine_bounds_obj = refine_bounds
            else:
                refine_lb, refine_scale = generate_bounds(refine_bounds, self.dimension)
                refine_ub = [
                    float(lb) + float(width)
                    for lb, width in zip(refine_lb, refine_scale)
                ]
                refine_bounds_obj = Bounds(
                    np.asarray(refine_lb, dtype=float),
                    np.asarray(refine_ub, dtype=float),
                )
        else:
            refine_bounds_obj = None

        self.acopf_refine_problem = {
            "objective": objective,
            "constraints": constraints,
            "lambda_vec": lambda_vec,
            "rho": float(rho),
            "x_center": x_center,
            "mu_prox": float(mu_prox),
            "best_only": bool(best_only),
            "tnc_options": dict(tnc_options or {}),
            "ipopt_options": dict(ipopt_options or {}),
            "gurobi_options": dict(gurobi_options or {}),
            "refine_bounds": refine_bounds_obj,
        }
        return self

    @staticmethod
    def _clip_to_bounds(x: np.ndarray, bounds: Bounds) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.minimum(np.maximum(x, np.asarray(bounds.lb, dtype=float)), np.asarray(bounds.ub, dtype=float))

    def _lambdify_scalar_and_grad(self, expr: Function):
        expr = sympy.expand(expr)
        grad_expr = [sympy.diff(expr, var) for var in self.syms]
        f_raw = lambdify(self.syms, expr, "numpy")
        g_raw = lambdify(self.syms, grad_expr, "numpy")

        def fun(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            return float(f_raw(*x))

        def jac(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            return np.asarray(g_raw(*x), dtype=float).reshape(-1)

        return fun, jac

    def _lambdify_constraints_and_jac(self, constraints: List[Function]):
        h_vec = sympy.Matrix(constraints)
        jac_expr = h_vec.jacobian(self.syms)
        h_raw = lambdify(self.syms, constraints, "numpy")
        jac_raw = lambdify(self.syms, jac_expr, "numpy")

        def cons(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            return np.asarray(h_raw(*x), dtype=float).reshape(-1)

        def jac(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            return np.asarray(jac_raw(*x), dtype=float)

        return cons, jac

    def _build_full_acopf_alm_expr(self):
        if self.acopf_refine_problem is None:
            raise ValueError("ACOPF refine problem is not configured. Call set_acopf_refine_problem first.")

        problem = self.acopf_refine_problem
        expr = problem["objective"]
        constraints = problem["constraints"]
        lambda_vec = problem["lambda_vec"]
        rho = problem["rho"]

        for lam_i, h_i in zip(lambda_vec, constraints):
            if lam_i != 0.0:
                expr += sympy.Float(lam_i) * h_i
            if rho != 0.0:
                expr += sympy.Float(rho) * sympy.Rational(1, 2) * h_i ** 2

        x_center = problem["x_center"]
        mu_prox = problem["mu_prox"]
        if mu_prox > 0.0 and x_center is not None:
            for var, val in zip(self.syms, x_center):
                expr += sympy.Float(mu_prox) * sympy.Rational(1, 2) * (var - sympy.Float(val)) ** 2

        return sympy.expand(expr)

    def _sympy_poly_to_gurobi_expr(self, expr: Function, gurobi_vars: List, max_degree: int = 2):
        expanded = sympy.expand(expr)
        poly = sympy.Poly(expanded, self.syms)
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

    def _setup_acopf_gurobi_model(self, name: str):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError(f"{name} requires gurobipy in the active Python environment.") from exc

        problem = self.acopf_refine_problem or {}
        options = problem.get("gurobi_options", {})
        model_g = gp.Model(name)
        model_g.Params.OutputFlag = 1 if options.get("log_to_console", False) else 0
        model_g.Params.NonConvex = 2
        if options.get("time_limit") is not None:
            model_g.Params.TimeLimit = float(options["time_limit"])
        if options.get("threads") is not None:
            model_g.Params.Threads = int(options["threads"])
        return gp, GRB, model_g

    def _add_acopf_gurobi_vars(self, model_g, bounds: Bounds, x0: Optional[np.ndarray] = None):
        if x0 is not None:
            x0 = self._clip_to_bounds(x0, bounds)

        variables = []
        for idx, sym in enumerate(self.syms):
            var = model_g.addVar(lb=float(bounds.lb[idx]), ub=float(bounds.ub[idx]), name=f"x_{idx}_{sym}")
            if x0 is not None:
                var.Start = float(x0[idx])
            variables.append(var)
        model_g.update()
        return variables

    def _refine_tnc_orig(self, x0: np.ndarray, bounds: Bounds):
        problem = self.acopf_refine_problem
        alm_expr = self._build_full_acopf_alm_expr()
        fun, jac = self._lambdify_scalar_and_grad(alm_expr)
        options = {"gtol": 1e-6, "eps": 1e-9}
        options.update(problem.get("tnc_options", {}))
        x0 = self._clip_to_bounds(x0, bounds)

        result = minimize(fun, x0, method="TNC", jac=jac, bounds=bounds, options=options)
        if not result.success:
            warnings.warn(f"TNC_orig refine did not report success: {result.message}", RuntimeWarning)
        return self._clip_to_bounds(result.x, bounds), float(result.fun)

    def _refine_ipopt_orig(self, x0: np.ndarray, bounds: Bounds):
        try:
            import cyipopt
        except ImportError as exc:
            raise RuntimeError("ipopt_orig requires cyipopt in the active Python environment.") from exc

        problem = self.acopf_refine_problem
        fun, jac = self._lambdify_scalar_and_grad(problem["objective"])
        cons_fun, cons_jac = self._lambdify_constraints_and_jac(problem["constraints"])
        options = {"tol": 1e-6, "max_iter": 100, "hessian_approximation": "limited-memory"}
        options.update(problem.get("ipopt_options", {}))
        x0 = self._clip_to_bounds(x0, bounds)

        result = cyipopt.minimize_ipopt(
            fun,
            x0,
            jac=jac,
            bounds=bounds,
            constraints={"type": "eq", "fun": cons_fun, "jac": cons_jac},
            options=options,
        )
        if not bool(getattr(result, "success", False)):
            warnings.warn(f"ipopt_orig refine did not report success: {getattr(result, 'message', '')}", RuntimeWarning)
        return self._clip_to_bounds(result.x, bounds), float(fun(result.x))

    def _refine_gurobi_orig(self, x0: np.ndarray, bounds: Bounds):
        _, _, model_g = self._setup_acopf_gurobi_model("GurobiOrig_ACOPF_QCQP")
        x_vars = self._add_acopf_gurobi_vars(model_g, bounds, x0=x0)
        problem = self.acopf_refine_problem

        model_g.setObjective(self._sympy_poly_to_gurobi_expr(problem["objective"], x_vars, max_degree=2))
        for idx, h_expr in enumerate(problem["constraints"]):
            model_g.addConstr(self._sympy_poly_to_gurobi_expr(h_expr, x_vars, max_degree=2) == 0.0,
                              name=f"acopf_h_{idx}")

        model_g.optimize()
        if model_g.SolCount < 1:
            raise RuntimeError(f"GurobiOrig did not return a solution, status={model_g.Status}.")
        return np.asarray([var.X for var in x_vars], dtype=float), float(model_g.ObjVal)

    def _refine_gurobi_alm(self, x0: np.ndarray, bounds: Bounds):
        _, GRB, model_g = self._setup_acopf_gurobi_model("GurobiALM_full_ACOPF_ALM")
        x0 = self._clip_to_bounds(x0, bounds)
        x_vars = self._add_acopf_gurobi_vars(model_g, bounds, x0=x0)
        problem = self.acopf_refine_problem

        gurobi_obj = self._sympy_poly_to_gurobi_expr(problem["objective"], x_vars, max_degree=2)
        cons_fun, _ = self._lambdify_constraints_and_jac(problem["constraints"])
        h_start = cons_fun(x0)

        for idx, (lam_i, h_expr) in enumerate(zip(problem["lambda_vec"], problem["constraints"])):
            h_var = model_g.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"h_{idx}")
            h_var.Start = float(h_start[idx])
            model_g.addConstr(
                h_var == self._sympy_poly_to_gurobi_expr(h_expr, x_vars, max_degree=2),
                name=f"define_h_{idx}",
            )
            if lam_i != 0.0:
                gurobi_obj += float(lam_i) * h_var
            gurobi_obj += 0.5 * float(problem["rho"]) * h_var * h_var

        x_center = problem["x_center"]
        mu_prox = problem["mu_prox"]
        if mu_prox > 0.0 and x_center is not None:
            for var, center_val in zip(x_vars, x_center):
                gurobi_obj += 0.5 * float(mu_prox) * (var - float(center_val)) * (var - float(center_val))

        model_g.setObjective(gurobi_obj)
        model_g.optimize()
        if model_g.SolCount < 1:
            raise RuntimeError(f"GurobiALM did not return a solution, status={model_g.Status}.")
        return np.asarray([var.X for var in x_vars], dtype=float), float(model_g.ObjVal)

    def acopf_optimizer_helper(self, samples: List[np.ndarray], bounds: Bounds, solver: str):
        method = canonical_acopf_refine_method(solver)
        if self.acopf_refine_problem is None:
            raise ValueError("ACOPF refine problem is not configured. Call set_acopf_refine_problem first.")
        problem = self.acopf_refine_problem
        refine_bounds = problem.get("refine_bounds")
        if refine_bounds is not None:
            bounds = refine_bounds

        if problem.get("best_only", True) and getattr(self, "coarse_minimizer", None) is not None:
            samples_to_refine = [self.coarse_minimizer]
        else:
            samples_to_refine = samples

        opt_samples = []
        sample_times = []
        minimizer = np.zeros(self.dimension)
        current_best = float("inf")
        start_time = time.time()

        for sample in samples_to_refine:
            if sample is None:
                opt_samples.append(None)
                continue
            sample_start_time = time.time()
            x0 = self._clip_to_bounds(sample, bounds)
            if method == "TNC_orig":
                result_x, result_val = self._refine_tnc_orig(x0, bounds)
            elif method == "ipopt_orig":
                result_x, result_val = self._refine_ipopt_orig(x0, bounds)
            elif method == "GurobiALM":
                result_x, result_val = self._refine_gurobi_alm(x0, bounds)
            elif method == "GurobiOrig":
                result_x, result_val = self._refine_gurobi_orig(x0, bounds)
            else:
                raise ValueError(f"Unsupported ACOPF refine method: {method}.")

            sample_times.append(time.time() - sample_start_time)
            opt_samples.append(result_x)
            if result_val < current_best:
                current_best = result_val
                minimizer = result_x

        post_processing_time = time.time() - start_time
        self.info["sample_times"] = sample_times
        self.info["post_processing_method"] = method
        return opt_samples, minimizer, current_best, post_processing_time, sample_times

    def generate_guess_in_box(self, shots: int = 1) -> List[np.ndarray]:
        """
        Generates initial guesses within the defined bounds.
        By default, generate a sample with a single guess (shots = 1)

        Args:
            shots: Number of guesses to generate.

        Returns:
            A list containing the generated guesses.
        """
        initial_guess = []
        for _ in range(shots):
            initial_guess.append(self.lb + self.scaling_factor * np.random.rand(self.dimension))

        return initial_guess

    def validate_guess_in_box(self, guesses: List[np.ndarray]) -> None:
        """
        Validates if the provided guesses are within the bounds.

        Args:
            guesses: List of guesses to validate.
        """
        tol=1e-4
        for guess in guesses:
            for i in range(len(self.lb)):
                lb = self.lb[i]
                ub = self.lb[i] + self.scaling_factor[i]
                assert lb - tol <= guess[i] <= ub + tol

    def classically_optimize(self, verbose=0, initial_guess=None, num_shots=100, solver="IPOPT") -> Response:
        """
        Optimizes a given function classically over a set of samples and within specified bounds.

        Args:
            samples: Initial samples for the optimization.
            bounds: Bounds within which the optimization is to be performed.
            solver: The optimization method

        Returns:
            Response object containing samples, minimum, minimizer, and other info
        """
        self.generate_affined_func()
        if initial_guess is None:
            initial_guess = self.generate_guess_in_box(num_shots)

        self.validate_guess_in_box(initial_guess)
        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        start_time = time.time()
        samples, minimizer, minimum, optimize_time, sample_times = self.classical_optimizer_helper(initial_guess,
                                                                                     bounds,
                                                                                     solver,
                                                                                     self.fun_eval)
        end_time = time.time()

        self.info["refined_minimum"] = minimum
        self.info["refining_time"] = optimize_time
        self.info["decoding_time"] = 0
        self.info["compile_time"] = 0
        self.info["backend_time"] = 0
        self.info["refine_status"] = True
        self.info["refining_time"] = end_time - start_time
        self.info["sample_times"] = sample_times

        classical_response = Response(self.info, refined_samples=samples, refined_minimum=minimum,
                                      refined_minimizer=minimizer, func=self.fun_eval)

        if verbose > 0:
            classical_response.print_time_info()
            classical_response.print_solver_info()
        self.response = classical_response

        return classical_response

    def classical_optimizer_helper(self, samples: List[np.ndarray], bounds: Bounds, solver: str,
                                   f: Callable) -> Tuple[
        List[np.ndarray], np.ndarray, float, float, List]:
        """
        Helper function to optimize a given function classically over a set of samples and within specified bounds.

        Args:
            samples: Initial samples for the optimization.
            bounds: Bounds within which the optimization is to be performed.
            solver: The optimization method
            f: The function to be optimized

        Returns:
            Tuple of samples, minimizer, minimum, and post-processing time
        """
        num_samples = len(samples)
        opt_samples = []
        sample_times = []
        minimizer = np.zeros(self.dimension)
        current_best = float("inf")
        f_eval_jit = jit(f)
        f_eval_grad = jit(grad(f_eval_jit))
        obj_hess = jit(jacrev(jacfwd(f_eval_jit)))
        start_time = time.time()
        for k in range(num_samples):
            if samples[k] is None:
                opt_samples.append(None)
                continue
            sample_start_time = time.time()
            x0 = jnp.array(samples[k])
            if solver == "TNC":
                result = minimize(
                    f_eval_jit,
                    x0,
                    method="TNC",
                    jac=f_eval_grad,
                    bounds=bounds,
                    options={"gtol": 1e-6, "eps": 1e-9},
                )
            elif solver == "IPOPT":
                import cyipopt
                result = cyipopt.minimize_ipopt(
                    f_eval_jit,
                    x0,
                    jac=f_eval_grad,
                    hess=obj_hess,
                    bounds=bounds,
                    options={"tol": 1e-6, "max_iter": 100},
                )
            else:
                raise Exception(
                    "The Specified Post Processing Method is Not Supported."
                )
            sample_times.append(time.time() - sample_start_time)
            opt_samples.append(result.x)
            val = float(f(result.x))
            if val < current_best:
                current_best = val
                minimizer = result.x
        end_time = time.time()
        post_processing_time = end_time - start_time
        self.info["sample_times"] = sample_times

        return opt_samples, minimizer, current_best, post_processing_time, sample_times

    def post_process(self) -> Tuple[np.ndarray, float, float]:
        """
        Private function to post-process the QHD samples returned from a quantum backend.

        Returns:
            Tuple of minimizer, minimum, and post-processing time
        """
        if self.decoded_samples is None:
            raise Exception("No results on record.")
        samples = self.decoded_samples
        solver = self.post_processing_method

        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        if str(solver).lower() in ACOPF_REFINE_METHOD_ALIASES:
            opt_samples, minimizer, current_best, post_processing_time, sample_times = self.acopf_optimizer_helper(
                samples, bounds, solver)
        else:
            opt_samples, minimizer, current_best, post_processing_time, sample_times = self.classical_optimizer_helper(
                samples, bounds, solver, self.fun_eval)
        self.post_processed_samples = opt_samples
        self.info["post_processing_time"] = post_processing_time

        return minimizer, current_best, post_processing_time

    def compile_only(self) -> Backend:
        return self.qhd_base.compile_only()

    def optimize(self, refine: bool = True, verbose: int = 0, override=None) -> Response:
        """
        User-facing function to run QHD on the optimization problem

        Args:
            refine: Flag to indicate if fine-tuning should be performed.
            compile_only: Flag to indicate if only the compilation should be performed.
            verbose: Verbosity level (0, 1, 2 for increasing detail).

        Returns:
            Response object containing samples, minimum, minimizer, and other info
        """
        response = self.qhd_base.optimize(verbose, override)
        print("Minimizer:", response.minimizer)
        #print("Minimum:", response.minimum)
        #print("Samples:", response.samples)
        self.coarse_minimizer, self.coarse_minimum, self.decoded_samples = self.affine_mapping(
            response.minimizer, response.minimum, response.samples)
        self.info["refine_status"] = refine
        if refine:
            start_time_finetuning = time.time()
            refined_minimizer, refined_minimum, _ = self.post_process()
            end_time_finetuning = time.time()
            self.info["refined_minimum"] = refined_minimum
            self.info["refining_time"] = end_time_finetuning - start_time_finetuning
            qhd_response = Response(self.info, self.decoded_samples, self.coarse_minimum,
                                    self.coarse_minimizer,
                                    self.post_processed_samples, refined_minimum, refined_minimizer,
                                    self.fun_eval)
        else:
            qhd_response = Response(self.info, self.decoded_samples, self.coarse_minimum,
                                    self.coarse_minimizer, self.fun_eval)

        if verbose > 0:
            qhd_response.print_time_info()
            qhd_response.print_solver_info()
        self.response = qhd_response

        return qhd_response

    def affine_mapping(self, minimizer: np.ndarray, minimum: float, samples: List[np.ndarray]) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Maps the minimizer and samples from the normalized space to the original space.

        Args:
            minimizer: The minimizer in the normalized space.
            minimum: The minimum value of the function.
            samples: The samples in the normalized space.

        Returns:
            Tuple of the minimizer, minimum, and samples in the original space.
        """
        original_minimizer = self.affine_transformation(minimizer)
        original_minimum = minimum
        original_samples = []

        for k in range(len(samples)):
            if samples[k] is None:
                original_samples.append(None)
            else:
                original_samples.append(self.affine_transformation(samples[k]))

        return original_minimizer, original_minimum, original_samples

    def get_solution(self, var=None):
        """
        var can be
        - None (return all values)
        - a Symbol (return the value of the symbol)
        - a list of Symbols (return a list of the values of the symbols)
        """

        values = self.response.minimizer

        if var is None:
            return values
        if isinstance(var, sympy.Symbol):
            return values[self.syms_index[var]]
        # Otherwise, v is a list of Symbols.
        return [values[self.syms_index[v]] for v in var]
    
    def solver_param_diagnose(self):
        if not isinstance(self.qhd_base.backend, dwave_backend.DWaveBackend):
            raise Exception(
                "This function is only used for D-Wave backends."
            )
        
        if self.response.samples is None:
            raise Exception(
                "This function must run after executing QHD on D-Wave."
            )

        h, J, _ = self.qhd_base.backend.exec(verbose=0, info=self.qhd_base.info, compile_only=True)
        hmax = np.max(np.abs(list(h)))
        Jmax = np.max(np.abs(list(J.values())))
        chain_break_fraction = self.qhd_base.backend.dwave_response.record['chain_break_fraction']

        shots_in_subspace = len(self.response.coarse_samples)

        print("***Solver Parameter Diagnosis***")
        print("---Solver Parameters---")
        print(f"hmax = {hmax}, Jmax = {Jmax}")
        print(f"penalty ratio = {self.qhd_base.backend.penalty_ratio}, penalty coefficient = {self.qhd_base.backend.penalty_coefficient}")
        print(f"chain strength ratio = {self.qhd_base.backend.chain_strength_ratio}, chain strength = {self.qhd_base.backend.chain_strength}")
        print("---Solution Stats---")
        print(f"total shots = {self.qhd_base.backend.shots}, shots in subspace = {shots_in_subspace}")
        print(f"median chain break fraction = {np.median(chain_break_fraction)}")

