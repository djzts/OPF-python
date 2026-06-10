from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider

from qhdopt.backend.backend import Backend


class GurobiBackend(Backend):
    """
    Local backend using Gurobi to solve either the compiled Ising model directly
    or an equivalent QUBO reformulation.

    The backend always returns spin samples in {-1, +1}, regardless of solver_mode.
    """

    def __init__(
        self,
        resolution,
        dimension,
        univariate_dict,
        bivariate_dict,
        shots: int = 100,
        embedding_scheme: str = "unary",
        anneal_schedule: Optional[List[List[int]]] = None,
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        chain_strength: float = 1.0,
        api_key = 'DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291', #Optional[str] = None,
        api_key_from_file: Optional[str] = None,
        solver_mode: str = "ising",
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        log_to_console: bool = False,
    ):
        super().__init__(
            resolution,
            dimension,
            shots,
            embedding_scheme,
            univariate_dict,
            bivariate_dict,
        )
        if anneal_schedule is None:
            anneal_schedule = [[0, 0], [20, 1]]

        self.anneal_schedule = anneal_schedule
        self.penalty_coefficient = float(penalty_coefficient)
        self.penalty_ratio = float(penalty_ratio)
        self.chain_strength = float(chain_strength)
        self.solver_mode = str(solver_mode).lower()
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.log_to_console = log_to_console

        self.api_key = api_key
        if api_key_from_file is not None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()

        self.h = None
        self.J = None
        self.qubo_linear = None
        self.qubo_quadratic = None
        self.qubo_constant = 0.0
        self._compiled_penalty = None

    def _validate_solver_mode(self) -> None:
        if self.solver_mode not in {"ising", "qubo"}:
            raise ValueError(
                f"Unsupported solver_mode={self.solver_mode!r}. Use 'ising' or 'qubo'."
            )

    def _compile_to_ising(self, penalty_coefficient: float) -> Tuple[List[float], Dict[tuple, float]]:
        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]

        qs.add_evolution(
            self.H_p(qubits, self.univariate_dict, self.bivariate_dict)
            + penalty_coefficient * self.H_pen(qubits),
            1,
        )

        dwp = DWaveProvider(api_key=self.api_key)
        h, J = dwp.compile(qs, self.anneal_schedule, self.chain_strength)
        return h, J

    def _calc_penalty_coefficient(self) -> float:
        if self.penalty_coefficient != 0:
            return float(self.penalty_coefficient)

        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]
        qs.add_evolution(
            self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict),
            1,
        )

        dwp = DWaveProvider(api_key=self.api_key)
        h, J = dwp.compile(qs, self.anneal_schedule, self.chain_strength)
        max_strength = np.max(np.abs(list(h) + list(J.values()))) if (len(h) + len(J)) > 0 else 0.0

        if self.embedding_scheme == "unary":
            return float(self.penalty_ratio * max_strength)
        return 0.0

    def _ising_to_qubo(
        self,
        h: List[float],
        J: Dict[tuple, float],
    ) -> Tuple[Dict[int, float], Dict[tuple, float], float]:
        h_vec = np.asarray(h, dtype=float)
        linear = {i: float(2.0 * h_vec[i]) for i in range(len(h_vec))}
        quadratic: Dict[tuple, float] = {}
        constant = float(-np.sum(h_vec))

        for (i, j), coupling in J.items():
            i = int(i)
            j = int(j)
            if i == j:
                linear[i] = linear.get(i, 0.0) + float(coupling)
                constant += float(coupling)
                continue

            if j < i:
                i, j = j, i

            coupling = float(coupling)
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 4.0 * coupling
            linear[i] = linear.get(i, 0.0) - 2.0 * coupling
            linear[j] = linear.get(j, 0.0) - 2.0 * coupling
            constant += coupling

        linear = {i: v for i, v in linear.items() if abs(v) > 0}
        quadratic = {k: v for k, v in quadratic.items() if abs(v) > 0}
        return linear, quadratic, constant

    def compile(self, info: dict, override=None):
        self._validate_solver_mode()

        penalty_coefficient = self._calc_penalty_coefficient()
        if override is not None:
            penalty_coefficient = float(override)

        self._compiled_penalty = penalty_coefficient

        start = time.time()
        self.h, self.J = self._compile_to_ising(penalty_coefficient)
        self.qubo_linear, self.qubo_quadratic, self.qubo_constant = self._ising_to_qubo(self.h, self.J)
        info["compile_time"] = time.time() - start

    def _ensure_compiled(self, info: dict, override=None):
        need_compile = self.h is None or self.J is None

        intended_penalty = self._calc_penalty_coefficient()
        if override is not None:
            intended_penalty = float(override)

        if self._compiled_penalty is None or float(intended_penalty) != float(self._compiled_penalty):
            need_compile = True

        if need_compile:
            self.compile(info, override)
        else:
            info["compile_time"] = 0.0

    def _build_model(self):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as e:
            raise ImportError(
                "GurobiBackend requires `gurobipy`. Install Gurobi and `pip install gurobipy`."
            ) from e

        model = gp.Model("qhdopt_gurobi")
        model.Params.OutputFlag = 1 if self.log_to_console else 0
        if self.time_limit is not None:
            model.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            model.Params.MIPGap = float(self.mip_gap)
        if self.threads is not None:
            model.Params.Threads = int(self.threads)

        num_vars = len(self.h)
        x = model.addVars(num_vars, vtype=GRB.BINARY, name="x")

        if self.shots and self.shots > 1:
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = int(self.shots)

        if self.solver_mode == "ising":
            obj = gp.QuadExpr()
            for i, coeff in enumerate(self.h):
                if coeff:
                    obj += float(coeff) * (2 * x[i] - 1)

            normalized_J: Dict[tuple, float] = {}
            for (i, j), coeff in self.J.items():
                i = int(i)
                j = int(j)
                coeff = float(coeff)
                if coeff == 0:
                    continue
                if j < i:
                    i, j = j, i
                normalized_J[(i, j)] = normalized_J.get((i, j), 0.0) + coeff

            for (i, j), coeff in normalized_J.items():
                if i == j:
                    obj += coeff
                else:
                    obj += coeff * (2 * x[i] - 1) * (2 * x[j] - 1)
        else:
            obj = gp.QuadExpr()
            for i, coeff in self.qubo_linear.items():
                obj += float(coeff) * x[int(i)]
            for (i, j), coeff in self.qubo_quadratic.items():
                obj += float(coeff) * x[int(i)] * x[int(j)]
            if self.qubo_constant:
                obj += float(self.qubo_constant)

        model.setObjective(obj, GRB.MINIMIZE)
        return gp, GRB, model, x

    def _extract_spin_solutions(self, model, vars_by_index) -> List[List[int]]:
        spins: List[List[int]] = []
        sol_count = int(model.SolCount)
        if sol_count <= 0:
            return spins

        num_vars = len(self.h)
        max_solutions = min(sol_count, self.shots if self.shots else sol_count)

        for sol_no in range(max_solutions):
            model.Params.SolutionNumber = sol_no
            sample = []
            for idx in range(num_vars):
                value = vars_by_index[idx].Xn if sol_count > 1 else vars_by_index[idx].X
                # Keep the extraction consistent with the Ising objective above:
                # s = 2x - 1, so x=0 -> s=-1 and x=1 -> s=+1.
                sample.append(-1 if int(round(value)) == 0 else 1)
            spins.append(sample)

        return spins

    def exec(self, verbose: int, info: dict, compile_only: bool = False, override=None) -> List[List[int]]:
        self._ensure_compiled(info, override)
        if compile_only:
            return []

        if verbose > 1:
            self.print_compilation_info()

        _, GRB, model, x = self._build_model()
        model.optimize()

        feasible_statuses = {
            GRB.OPTIMAL,
            GRB.SUBOPTIMAL,
            GRB.TIME_LIMIT,
            GRB.INTERRUPTED,
            GRB.SOLUTION_LIMIT,
        }
        if model.Status not in feasible_statuses or model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi did not return a feasible solution. status={model.Status}, sol_count={model.SolCount}"
            )

        info["backend_time"] = float(model.Runtime)
        info["average_qpu_time"] = 0.0
        info["time_on_machine"] = float(model.Runtime)
        info["overhead_time"] = 0.0

        raw_samples = self._extract_spin_solutions(model, x)

        if verbose > 0:
            print(f"Gurobi runtime: {info['backend_time']}")
            print(f"Solver mode: {self.solver_mode}")
            print(f"Solutions returned: {len(raw_samples)}")

        return raw_samples

    def print_compilation_info(self):
        print("* Compilation information (Gurobi)")
        print(f"Solver mode: {self.solver_mode}")
        print(f"Annealing schedule parameter: {self.anneal_schedule}")
        print(f"Penalty coefficient: {self._compiled_penalty}")
        print(f"Chain strength: {self.chain_strength}")
        print(f"Number of shots: {self.shots}")
