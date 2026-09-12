"""
A Gurobi backend for computing exact ground-truth solutions, used to
benchmark IBM Quantum / QAOA results against a verified-correct classical
optimum of the SAME Ising instance.

Why this is a separate file from `guorbi_backend.py`
-----------------------------------------------------
`GurobiBackend` (in `guorbi_backend.py`, used by the production ALM
beam-search solvers via `SolverConfig.qhd_solver="gurobi"`) has a confirmed
sign bug: its `_build_model()` builds the ising-mode objective using
`s = 2*x - 1` (Gurobi binary variable x=1 -> spin +1), while `Backend.H_p`,
SimuQ's D-Wave compiler, `spin_to_bitstring`, and this repo's
`qhdopt.backend.ising_compile` all use the OPPOSITE convention
`s = 1 - 2*x` (bit=1 -> spin -1). Flipping every spin's sign leaves
quadratic (J) terms unchanged but flips the sign of every linear (h) term,
so `GurobiBackend` with `solver_mode="ising"` (and `solver_mode="qubo"`,
whose `_ising_to_qubo` conversion makes the same `s=2x-1` assumption)
silently optimizes a DIFFERENT objective than the one actually compiled
from the problem.

Confirmed empirically: for a real 2-bus ALM subproblem instance, Gurobi's
own reported `ObjVal` matched `ising_energy(h, J, spins)` computed with
`spins = 2*bits-1` (its own convention) to 9+ significant figures, but
disagreed with the canonical `spins = 1-2*bits` convention by more than a
factor of 1.5x on that same optimal bitstring -- i.e. `GurobiBackend` was
not solving the problem `H_p` actually describes.

Rather than patch `guorbi_backend.py` in place (which the production
beam-search scripts depend on, and which would need a full
known-answer-file regression pass -- see `case_data/*-answer.txt` -- before
being trusted again), this backend is written fresh using the already
sign-verified `qhdopt.backend.ising_compile.compile_backend_to_ising`
helper to obtain `(h, J)`, builds the Gurobi objective with the CORRECT
`s = 1 - 2*x` substitution, and self-checks its own result against
`ising_energy(...)` before returning -- so a future reintroduction of this
sign-convention bug fails loudly instead of silently.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from qhdopt.backend.backend import Backend
from qhdopt.backend.ising_compile import compile_backend_to_ising, ising_energy


class IBMQGurobiBackend(Backend):
    """
    Local, offline Gurobi backend with a VERIFIED-correct sign convention
    (bit=1 <-> spin=-1, matching H_p / ising_compile / SimuQ's D-Wave
    compiler), intended for computing exact ground truth to compare against
    IBM Quantum (QAOA) results on the same discretized instance -- not a
    drop-in replacement for `GurobiBackend` in the production ALM loop.
    """

    def __init__(
        self,
        resolution,
        dimension,
        univariate_dict,
        bivariate_dict,
        shots: int = 1,
        embedding_scheme: str = "binary",
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        time_limit: Optional[float] = 60.0,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        log_to_console: bool = False,
    ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict, bivariate_dict)
        self.penalty_coefficient = float(penalty_coefficient)
        self.penalty_ratio = float(penalty_ratio)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.log_to_console = log_to_console

        self.h: Optional[List[float]] = None
        self.J = None
        self.status = None
        self.objective_value = None

    def _resolved_penalty_coefficient(self) -> float:
        if self.penalty_coefficient != 0:
            return self.penalty_coefficient
        if self.embedding_scheme != "unary":
            return 0.0
        from qhdopt.backend.ising_compile import estimate_unary_penalty_coefficient
        return estimate_unary_penalty_coefficient(self, self.penalty_ratio)

    def compile(self, info: dict):
        penalty_coefficient = self._resolved_penalty_coefficient()
        self.h, self.J = compile_backend_to_ising(self, penalty_coefficient=penalty_coefficient)

    def exec(self, verbose: int = 0, info: Optional[dict] = None, override=None) -> List[List[int]]:
        info = info if info is not None else {}
        if self.h is None or self.J is None:
            self.compile(info)

        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError(
                "IBMQGurobiBackend requires `gurobipy`. Install Gurobi and `pip install gurobipy`."
            ) from exc

        n = len(self.h)
        model = gp.Model("ibmq_gurobi_ground_truth")
        model.Params.OutputFlag = 1 if self.log_to_console else 0
        model.Params.NonConvex = 2
        if self.time_limit is not None:
            model.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            model.Params.MIPGap = float(self.mip_gap)
        if self.threads is not None:
            model.Params.Threads = int(self.threads)

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        if self.shots and self.shots > 1:
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = int(self.shots)

        # CORRECT convention: s_i = 1 - 2*x_i  (x=0 -> s=+1, x=1 -> s=-1),
        # matching H_p's n_j = 0.5*(I - Z) / spin_to_bitstring exactly.
        obj = gp.QuadExpr()
        for i, coeff in enumerate(self.h):
            if coeff:
                obj += float(coeff) * (1 - 2 * x[i])
        for (i, j), coeff in self.J.items():
            i, j = int(i), int(j)
            coeff = float(coeff)
            if coeff == 0:
                continue
            obj += coeff * (1 - 2 * x[i]) * (1 - 2 * x[j])

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        feasible_statuses = {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SOLUTION_LIMIT}
        if model.Status not in feasible_statuses or model.SolCount == 0:
            raise RuntimeError(
                f"IBMQGurobiBackend: Gurobi returned no feasible solution. "
                f"status={model.Status}, sol_count={model.SolCount}"
            )
        if model.Status != GRB.OPTIMAL and verbose >= 0:
            print(
                f"[IBMQGurobiBackend] WARNING: Gurobi status={model.Status} "
                f"(not GRB.OPTIMAL={GRB.OPTIMAL}); MIPGap={model.MIPGap:.3g}. "
                "This may not be a certified global optimum -- treat as a strong "
                "but not guaranteed lower bound on the true ground truth."
            )

        self.status = model.Status
        self.objective_value = float(model.ObjVal)
        info["backend_time"] = float(model.Runtime)
        info["mip_gap"] = float(model.MIPGap)

        num_solutions = min(model.SolCount, self.shots if self.shots else model.SolCount)
        raw_samples: List[List[int]] = []
        for sol_no in range(num_solutions):
            model.Params.SolutionNumber = sol_no
            bits = [int(round(x[i].Xn if model.SolCount > 1 else x[i].X)) for i in range(n)]
            # Return SPINS in the s=1-2x convention so that
            # Backend._sample_to_bitstring -> spin_to_bitstring round-trips
            # back to the exact same `bits` (spin=+1 -> bit=0, spin=-1 -> bit=1).
            raw_samples.append([1 - 2 * b for b in bits])

        # Self-check: the objective Gurobi actually optimized must agree
        # with `ising_energy` computed independently from the same (h, J) --
        # this is exactly the check that caught guorbi_backend.py's bug.
        best_bits = [0 if s == 1 else 1 for s in raw_samples[0]]
        recomputed = ising_energy(self.h, self.J, raw_samples[0])
        if not np.isclose(recomputed, self.objective_value, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                f"IBMQGurobiBackend sign-convention self-check FAILED: "
                f"Gurobi ObjVal={self.objective_value!r} but ising_energy(h, J, spins) "
                f"={recomputed!r} for the same solution. Do not trust this result."
            )

        if verbose > 0:
            print(f"[IBMQGurobiBackend] status={self.status}, objective={self.objective_value:.9g}, "
                  f"runtime={info['backend_time']:.3f}s, solutions={len(raw_samples)}, "
                  "sign-convention self-check: OK")

        return raw_samples
