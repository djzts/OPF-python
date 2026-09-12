"""
Shared, backend-agnostic Hamiltonian -> Ising/QUBO compilation utilities.

`DWaveBackend`, `SimulatedBifurcationBackend`, and `GurobiBackend` each embed
a copy of the same trick: build the abstract problem Hamiltonian
`H_p(...) + penalty_coefficient * H_pen(...)` as a SimuQ `TIHamiltonian`, then
hand it to `simuq.dwave.DWaveProvider(...).compile(...)` to get back an
explicit Ising model `(h, J)`. That compile step is purely local/offline --
it algebraically translates the abstract operator into numeric spin
coefficients and never talks to D-Wave's cloud service. Only an actual
`DWaveProvider.run()` call (used exclusively by `DWaveBackend.exec`) submits
anything to real hardware.

This module factors that shared technique out of the three existing backends
so new *local/offline* backends -- in particular a future gate-model backend
that submits QAOA circuits to IBM Quantum -- can reuse it without depending
on any cloud account. `dwave_backend.py`, `simbi_backend.py`, and
`guorbi_backend.py` are intentionally left untouched for now to avoid
regressing already-working solver paths; they can be pointed at these helpers
later as a follow-up dedup.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider

# `DWaveProvider.compile(...)` only needs *some* string to construct the
# provider object; it does not validate the key or contact D-Wave's cloud
# service during `.compile()` (only `.run()` does, and nothing here calls
# `.run()`). This literal placeholder matches the one already hardcoded as a
# default in `simbi_backend.py`/`guorbi_backend.py` for the same purpose --
# it is not a secret and grants no cloud access on its own.
LOCAL_COMPILE_PLACEHOLDER_KEY = "DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291"

DEFAULT_ANNEAL_SCHEDULE = [[0, 0], [20, 1]]


def compile_backend_to_ising(
    backend,
    penalty_coefficient: float,
    anneal_schedule: Optional[List[List[int]]] = None,
    chain_strength: float = 1.0,
) -> Tuple[List[float], Dict[Tuple[int, int], float]]:
    """
    Compile `backend.H_p(...) + penalty_coefficient * backend.H_pen(...)`
    into an explicit Ising model `(h, J)` with spins in {-1, +1}.

    `backend` must expose the `Backend` interface (`.qubits`, `.H_p`,
    `.H_pen`, `.univariate_dict`, `.bivariate_dict`); it is never asked to
    `exec()` or submit anything anywhere. This is a local, offline
    computation only.

    Returns:
        h: list of length len(backend.qubits) of linear (field) coefficients.
        J: dict mapping (i, j) qubit index pairs (i < j) to coupling
           coefficients.
    """
    schedule = anneal_schedule if anneal_schedule is not None else DEFAULT_ANNEAL_SCHEDULE

    qs = QSystem()
    qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(backend.qubits))]
    qs.add_evolution(
        backend.H_p(qubits, backend.univariate_dict, backend.bivariate_dict)
        + penalty_coefficient * backend.H_pen(qubits),
        1,
    )
    dwp = DWaveProvider(api_key=LOCAL_COMPILE_PLACEHOLDER_KEY)
    h, J = dwp.compile(qs, schedule, chain_strength)
    return h, J


def estimate_unary_penalty_coefficient(
    backend,
    penalty_ratio: float,
    anneal_schedule: Optional[List[List[int]]] = None,
) -> float:
    """
    Mirror the `penalty_ratio * max(|h|, |J|)` heuristic already used by
    `DWaveBackend`/`SimulatedBifurcationBackend`/`GurobiBackend` for unary
    embeddings. Returns 0.0 for any embedding scheme that needs no penalty
    term (binary, hamming), matching `Backend.H_pen`.
    """
    if backend.embedding_scheme != "unary":
        return 0.0

    h, J = compile_backend_to_ising(backend, penalty_coefficient=0.0, anneal_schedule=anneal_schedule)
    max_strength = np.max(np.abs(list(h) + list(J.values()))) if (len(h) + len(J)) > 0 else 0.0
    return float(penalty_ratio * max_strength)


def ising_to_qubo(
    h: List[float], J: Dict[Tuple[int, int], float]
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Convert a spin-domain Ising model (s in {-1, +1}) into an equivalent
    binary-domain QUBO (x in {0, 1}) via the standard substitution
    `s = 2*x - 1`. Identical algebra to `GurobiBackend._ising_to_qubo`,
    factored out here for reuse by other local/offline backends.

    Returns:
        linear: dict {i: coefficient} (zero-valued entries dropped).
        quadratic: dict {(i, j): coefficient}, i < j (zero-valued dropped).
        constant: scalar energy offset.
    """
    h_vec = np.asarray(h, dtype=float)
    linear: Dict[int, float] = {i: float(2.0 * h_vec[i]) for i in range(len(h_vec))}
    quadratic: Dict[Tuple[int, int], float] = {}
    constant = float(-np.sum(h_vec))

    for (i, j), coupling in J.items():
        i, j = int(i), int(j)
        coupling = float(coupling)
        if i == j:
            linear[i] = linear.get(i, 0.0) + coupling
            constant += coupling
            continue
        if j < i:
            i, j = j, i
        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 4.0 * coupling
        linear[i] = linear.get(i, 0.0) - 2.0 * coupling
        linear[j] = linear.get(j, 0.0) - 2.0 * coupling
        constant += coupling

    linear = {i: v for i, v in linear.items() if abs(v) > 0}
    quadratic = {k: v for k, v in quadratic.items() if abs(v) > 0}
    return linear, quadratic, constant


def ising_energy(h: List[float], J: Dict[Tuple[int, int], float], spins: List[int]) -> float:
    """Evaluate E(s) = sum_i h_i*s_i + sum_{i<j} J_ij*s_i*s_j for a spin
    assignment (s_i in {-1, +1}). Useful for tests/debugging."""
    spins = list(spins)
    energy = sum(float(hi) * spins[i] for i, hi in enumerate(h))
    for (i, j), coupling in J.items():
        energy += float(coupling) * spins[int(i)] * spins[int(j)]
    return energy
