"""
Gate-model QAOA backend targeting IBM Quantum, via Qiskit + either Qiskit
Aer (local simulator) or Qiskit Runtime (real IBM hardware).

Read this before using `qhd_solver="ibmq"` / `QHD.ibmq_setup()`
-----------------------------------------------------------------
1. **Use `embedding_scheme="binary"` (the default here), not "unary".**
   Unary uses `resolution` qubits per decision variable to encode
   `resolution + 1` discretization levels; binary uses `resolution` BITS per
   variable to encode `2**resolution` levels -- far more qubit-efficient.
   `resolution` here means "bits per variable", not "grid points", which is
   a different meaning than `simbi_resolution`/`SolverConfig.resolution`
   elsewhere in this repo -- do not share one value across both. Binary
   encoding can only exactly represent univariate/bivariate factors that
   are degree<=2 polynomials of their variable; see
   `Backend._fit_quadratic_from_callable` for the (loud) failure mode if a
   higher-degree function sneaks in. The linearized ALM Lagrangian this
   repo builds (objective + rho/2*h_linearized**2, with h_linearized affine)
   satisfies this by construction.

2. **Qubit budget is the binding constraint, not "quality".** A dense ALM
   subproblem (most decision variables couple to most others) needs a
   fully-connected QAOA cost Hamiltonian. On real IBM hardware (heavy-hex
   connectivity) this forces heavy SWAP routing; on Aer, a dense graph
   defeats the `matrix_product_state` method's compression entirely
   (verified empirically: a 28-qubit dense instance needed ~4.3 GB, the
   same as a plain `statevector` simulation -- i.e. MPS bought nothing
   here). `max_qubits` below defaults to a conservative 30 for exactly this
   reason; raise it only once you know what you're asking for.

3. **This backend must be reached through `QHD.SymPy(...).ibmq_setup(...)`,
   not constructed directly from a raw physical-units Lagrangian.** Like
   every other backend, it consumes `univariate_dict`/`bivariate_dict`
   produced by `decompose_function`, which assumes the function has
   already been mapped to the unit box `[0, 1]` per dimension -- this only
   happens if `QHD.generate_affined_func()` ran first (which
   `ibmq_setup`/`simbi_setup`/`gurobi_setup` etc. all do internally before
   building `QHD_Base`). Calling `decompose_function` on a physical-units
   expression directly silently builds a Hamiltonian that does not
   correspond to the physical decode `x = lb + scale * x_norm` -- this is
   exactly the bug that produced nonsensical QAOA-vs-Gurobi comparisons
   during initial development of this backend; the fix was simply routing
   through the normal `QHD.*_setup()` path.

4. **`use_simulator=True` (default) is strongly recommended before ever
   setting it to False.** IBM Quantum queue time and (for paid plans)
   billed QPU-seconds are real costs; validate the whole pipeline on Aer
   first (small `shots`/`maxiter` for a smoke test, then scale up) and only
   point at real hardware once the simulator run looks sane.

5. **A large apparent memory blowup during development turned out to be a
   Windows `tasklist`/Task Manager reporting artifact, not a real leak --
   verified with in-process `psutil` RSS readings.** Across several runs
   of a 28-qubit dense-graph circuit, `tasklist`'s "Mem Usage" column
   climbed from ~4.5 GB to ~33 GB right around the 4th COBYLA evaluation,
   reproducibly, regardless of whether `EstimatorV2` instances were reused
   or recreated per call, which primitive (Estimator/Sampler) or API
   (primitives vs `backend.run()`) was used, or whether Aer's internal
   parallelism was capped to 1 thread. The common factor turned out to be
   the measurement tool: `psutil.Process().memory_info().rss`, read from
   *inside* the same process at the same moment `tasklist` showed ~33 GB,
   consistently read ~300 MB -- true resident memory never left a normal
   range. Likely explanation: Aer's allocator/OpenMP runtime reserves a
   large virtual address range on Windows that `tasklist` counts but that
   is never actually committed to physical pages. **Trust in-process RSS
   (e.g. via `psutil`) over `tasklist`/Task Manager "Mem Usage" when
   judging whether an Aer simulator run is actually using memory.**

6. **Self-consistency check.** After sampling, this backend compares the
   shot-weighted mean Ising energy against the Estimator's reported
   expectation value at the same parameters; a `p=1`, few-iteration QAOA
   run on a wide-energy-range problem can legitimately have a large
   sampling-noise gap here (this is NOT necessarily a bug -- it was, once,
   during this backend's own development, caused by an un-inverted
   transpiler qubit layout permutation between the Estimator path and the
   Sampler-decode path; forcing an identity `initial_layout` on the
   simulator path, as done here, is what fixed it). By default this only
   warns; pass `consistency_tol` to make it raise instead.
"""
from __future__ import annotations

import os
import time
import warnings
from typing import List, Optional

import numpy as np

from qhdopt.backend.backend import Backend
from qhdopt.backend.ising_compile import compile_backend_to_ising


class IBMQBackend(Backend):
    """
    QAOA backend for IBM Quantum (Aer simulator by default, or real
    hardware via Qiskit Runtime when `use_simulator=False`).
    """

    def __init__(
        self,
        resolution,
        dimension,
        univariate_dict,
        bivariate_dict,
        shots: int = 4096,
        embedding_scheme: str = "binary",
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        reps: int = 1,
        optimizer: str = "COBYLA",
        maxiter: int = 15,
        initial_params: Optional[np.ndarray] = None,
        seed: Optional[int] = 42,
        use_simulator: bool = True,
        simulator_method: str = "statevector",
        channel: Optional[str] = None,
        instance: Optional[str] = None,
        token_env_var: str = "IBM_QUANTUM_TOKEN",
        backend_name: Optional[str] = None,
        resilience_level: int = 1,
        max_qubits: int = 30,
        consistency_tol: Optional[float] = None,
        verbose_progress: bool = False,
    ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict, bivariate_dict)

        if embedding_scheme not in ("binary", "hamming"):
            warnings.warn(
                f"IBMQBackend with embedding_scheme={embedding_scheme!r}: unary/one-hot "
                "encodings need `resolution` qubits PER VARIABLE and typically make the "
                "resulting QAOA circuit's qubit count infeasible on gate-model hardware "
                "or simulators. Prefer embedding_scheme='binary' (the default).",
                stacklevel=2,
            )

        n_qubits = dimension * resolution
        if n_qubits > max_qubits:
            raise ValueError(
                f"IBMQBackend: this problem needs {n_qubits} qubits (dimension={dimension} "
                f"x resolution={resolution} bits/variable), which exceeds "
                f"max_qubits={max_qubits}. Reduce resolution (bits/variable), reduce the "
                "problem's dimension, or explicitly raise max_qubits if you understand the "
                "cost of a circuit this large (see module docstring)."
            )

        self.penalty_coefficient = float(penalty_coefficient)
        self.penalty_ratio = float(penalty_ratio)
        self.reps = int(reps)
        self.optimizer = optimizer
        self.maxiter = int(maxiter)
        self.initial_params = initial_params
        self.seed = seed
        self.use_simulator = bool(use_simulator)
        self.simulator_method = simulator_method
        self.channel = channel
        self.instance = instance
        self.token_env_var = token_env_var
        self.backend_name = backend_name
        self.resilience_level = int(resilience_level)
        self.max_qubits = int(max_qubits)
        self.consistency_tol = consistency_tol
        self.verbose_progress = bool(verbose_progress)

        self.h: Optional[List[float]] = None
        self.J = None
        self.last_run_info: dict = {}

    # ------------------------------------------------------------------
    # Ising compilation (shared with IBMQGurobiBackend / other backends)
    # ------------------------------------------------------------------
    def _resolved_penalty_coefficient(self) -> float:
        if self.penalty_coefficient != 0:
            return self.penalty_coefficient
        if self.embedding_scheme != "unary":
            return 0.0
        from qhdopt.backend.ising_compile import estimate_unary_penalty_coefficient
        return estimate_unary_penalty_coefficient(self, self.penalty_ratio)

    def compile(self, info: dict):
        t0 = time.time()
        penalty_coefficient = self._resolved_penalty_coefficient()
        self.h, self.J = compile_backend_to_ising(self, penalty_coefficient=penalty_coefficient)
        info["compile_time"] = time.time() - t0

    def _build_cost_operator(self):
        from qiskit.quantum_info import SparsePauliOp

        n = len(self.h)
        terms = [("Z", [i], float(hi)) for i, hi in enumerate(self.h) if abs(hi) > 0]
        terms += [("ZZ", [int(i), int(j)], float(v)) for (i, j), v in self.J.items() if abs(v) > 0]
        if not terms:
            terms = [("I" * n, [], 0.0)]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=n)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def _get_runtime_backend(self):
        if self.use_simulator:
            from qiskit_aer import AerSimulator
            return AerSimulator(method=self.simulator_method), None

        from qiskit_ibm_runtime import QiskitRuntimeService

        service_kwargs = {}
        if self.channel:
            service_kwargs["channel"] = self.channel
        if self.instance:
            service_kwargs["instance"] = self.instance
        token = os.environ.get(self.token_env_var)
        if token:
            service_kwargs["token"] = token
        service = QiskitRuntimeService(**service_kwargs)

        if self.backend_name:
            backend = service.backend(self.backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)
            if self.verbose_progress:
                print(f"[IBMQBackend] no backend_name given; using least_busy: {backend.name}")

        if backend.num_qubits < len(self.h):
            raise ValueError(
                f"Selected IBM backend {backend.name!r} has {backend.num_qubits} qubits, "
                f"fewer than the {len(self.h)} this problem needs."
            )
        return backend, service

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def exec(self, verbose: int = 0, info: Optional[dict] = None, override=None) -> List[List[int]]:
        info = info if info is not None else {}
        if self.h is None or self.J is None:
            self.compile(info)

        from scipy.optimize import minimize
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        n_qubits = len(self.h)
        cost_op = self._build_cost_operator()
        ansatz = QAOAAnsatz(cost_operator=cost_op, reps=self.reps)

        backend, service = self._get_runtime_backend()

        if self.use_simulator:
            # Force an IDENTITY layout: AerSimulator here has no restricted
            # coupling_map, so this is always achievable, and it removes an
            # entire class of bug where the Estimator path (which correctly
            # applies the transpiler's layout to the cost operator) and the
            # final Sampler-decode path (reading raw classical bits back)
            # silently disagree about which physical wire is which decision
            # variable bit. See module docstring point 5.
            pm = generate_preset_pass_manager(
                optimization_level=0, backend=backend, initial_layout=list(range(n_qubits))
            )
        else:
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

        isa_ansatz = pm.run(ansatz)
        isa_cost_op = cost_op.apply_layout(isa_ansatz.layout)

        if self.use_simulator:
            final_layout = list(isa_ansatz.layout.final_index_layout())
            if final_layout != list(range(n_qubits)):
                raise AssertionError(
                    f"Expected an identity qubit layout on the simulator path, got "
                    f"{final_layout}. Refusing to continue rather than silently decode "
                    "bitstrings against the wrong qubit mapping."
                )

        if self.use_simulator:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
            estimator = AerEstimatorV2()
            session_cm = None
        else:
            from qiskit_ibm_runtime import EstimatorV2, SamplerV2, Session
            session_cm = Session(backend=backend)
            session_cm.__enter__()
            estimator = EstimatorV2(mode=session_cm)
            estimator.options.resilience_level = self.resilience_level
            sampler = SamplerV2(mode=session_cm)

        try:
            eval_count = 0
            t_opt_start = time.time()

            def cost_fn(params):
                nonlocal eval_count
                eval_count += 1
                result = estimator.run([(isa_ansatz, isa_cost_op, params)]).result()
                energy = float(result[0].data.evs)
                if self.verbose_progress or verbose > 1:
                    try:
                        import psutil
                        rss_mb = psutil.Process().memory_info().rss / 1e6
                    except Exception:
                        rss_mb = float("nan")
                    print(f"[IBMQBackend] eval {eval_count}: energy={energy:.6f} rss={rss_mb:.1f}MB", flush=True)
                return energy

            rng = np.random.default_rng(self.seed)
            x0 = (
                self.initial_params
                if self.initial_params is not None
                else rng.uniform(-0.3, 0.3, size=ansatz.num_parameters)
            )

            opt_result = minimize(cost_fn, x0, method=self.optimizer, options={"maxiter": self.maxiter})
            optimizer_time = time.time() - t_opt_start

            isa_meas = isa_ansatz.copy()
            isa_meas.measure_all()
            bound_meas = isa_meas.assign_parameters(opt_result.x)
            if self.use_simulator:
                # Use the traditional backend.run()/get_counts() execution
                # path rather than qiskit_aer.primitives.SamplerV2 for the
                # final measurement -- simpler and avoids depending on
                # primitives-specific PUB formatting for a plain shots-count
                # readout.
                job = backend.run(bound_meas, shots=self.shots)
                counts = job.result().get_counts()
            else:
                sampler_result = sampler.run([(bound_meas,)], shots=self.shots).result()
                counts = sampler_result[0].data.meas.get_counts()
        finally:
            if session_cm is not None:
                session_cm.__exit__(None, None, None)

        # ---- decode: bits[i] == qubit i directly (identity layout verified
        # above on the simulator path; real hardware relies on Qiskit's own
        # `apply_layout`/final-layout bookkeeping being correct, which this
        # backend does not independently re-verify there) ----
        def bitstring_to_bits(bitstring: str) -> List[int]:
            # Qiskit's classical bitstrings are MSB-first with qubit 0 as
            # the rightmost character.
            return [int(c) for c in reversed(bitstring)]

        raw_samples: List[List[int]] = []
        weighted_energy_sum = 0.0
        total_shots = 0
        for bitstring, count in counts.items():
            bits = bitstring_to_bits(bitstring)
            raw_samples.append(bits)
            spins = [1 - 2 * b for b in bits]
            energy = sum(self.h[i] * spins[i] for i in range(n_qubits))
            for (i, j), coupling in self.J.items():
                energy += coupling * spins[int(i)] * spins[int(j)]
            weighted_energy_sum += energy * count
            total_shots += count

        sample_mean_energy = weighted_energy_sum / total_shots if total_shots else float("nan")
        consistency_gap = abs(sample_mean_energy - opt_result.fun)

        info["backend_time"] = optimizer_time
        info["average_qpu_time"] = 0.0
        info["time_on_machine"] = optimizer_time
        info["overhead_time"] = 0.0
        info["ibmq_optimizer_evals"] = eval_count
        info["ibmq_optimizer_time"] = optimizer_time
        info["ibmq_estimator_energy_at_optimum"] = float(opt_result.fun)
        info["ibmq_sample_mean_energy"] = float(sample_mean_energy)
        info["ibmq_consistency_gap"] = float(consistency_gap)
        info["ibmq_n_distinct_bitstrings"] = len(counts)
        info["ibmq_use_simulator"] = self.use_simulator
        info["ibmq_backend_name"] = getattr(backend, "name", str(backend))
        self.last_run_info = dict(info)

        if verbose > 0:
            print(
                f"[IBMQBackend] backend={info['ibmq_backend_name']}, evals={eval_count}, "
                f"optimizer_time={optimizer_time:.1f}s, estimator_energy={opt_result.fun:.6f}, "
                f"sample_mean_energy={sample_mean_energy:.6f}, "
                f"consistency_gap={consistency_gap:.6f}, "
                f"distinct_bitstrings={len(counts)}/{self.shots}"
            )

        if self.consistency_tol is not None and consistency_gap > self.consistency_tol:
            raise RuntimeError(
                f"IBMQBackend self-consistency check failed: |sample_mean_energy - "
                f"estimator_energy| = {consistency_gap:.6f} exceeds consistency_tol="
                f"{self.consistency_tol}. Do not trust the returned samples."
            )
        elif self.consistency_tol is None and consistency_gap > 0.1 * max(1.0, abs(opt_result.fun)):
            warnings.warn(
                f"IBMQBackend: sample-mean energy ({sample_mean_energy:.6g}) differs from the "
                f"Estimator's expectation value ({opt_result.fun:.6g}) by {consistency_gap:.6g}. "
                "This can be ordinary sampling variance for a shallow (small reps/maxiter) QAOA "
                "run on a wide-energy-range problem, but consider more shots/iterations or pass "
                "consistency_tol to make this fatal instead of a warning.",
                stacklevel=2,
            )

        return raw_samples
