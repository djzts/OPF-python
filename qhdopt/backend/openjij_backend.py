from __future__ import annotations

from typing import Tuple, List, Dict
import time
import numpy as np

from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider
import openjij as oj

from qhdopt.backend.backend import Backend
from qhdopt.utils.decoding_utils import spin_to_bitstring


class OpenJijBackend(Backend):
    """
    Local CPU backend using OpenJij samplers.

    Contract:
      exec() returns raw_samples as List[List[int]] (0/1 bitstrings),
      so it can be decoded by Backend.decoder().

    Design choice:
      - self.sampler_kwargs is passed directly into the OpenJij sampler
        constructor with NO filtering / validation.
      - sample_ising() is called with num_reads=self.shots only.
      - If users pass unsupported kwargs, OpenJij should raise the error
        directly. We do not intercept or sanitize it here.
    """

    def __init__(
    self,
        resolution,
        dimension,
        univariate_dict,
        bivariate_dict,
        shots=100,
        embedding_scheme="unary",
        penalty_coefficient=0.0,
        penalty_ratio=0.75,
        api_key='DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291',
        sampler_name="SQASampler",
        seed: int | None = None,
        debug: bool = False,
        sampler_init_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
    ):
        super().__init__(
            resolution,
            dimension,
            shots,
            embedding_scheme,
            univariate_dict,
            bivariate_dict,
        )

        self.penalty_coefficient = float(penalty_coefficient)
        self.penalty_ratio = float(penalty_ratio)
        self.sampler_name = sampler_name
        self.seed = seed
        self.api_key = api_key
        self.debug = debug

        self.sampler_init_kwargs = dict(sampler_init_kwargs or {})
        self.sample_kwargs = dict(sample_kwargs or {})

        self.h = None
        self.J = None
        self._compiled_penalty = None

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _log(self, *args, force: bool = False):
        if force or self.debug:
            print("[OpenJijBackend]", *args)

    # ------------------------------------------------------------------
    # compilation helpers
    # ------------------------------------------------------------------
    def _compile_to_ising(self, penalty_coefficient: float) -> Tuple[List[float], Dict[tuple, float]]:
        """
        Use SimuQ's DWaveProvider compiler to convert QSystem to Ising (h, J).
        This is a local compilation step only.
        """
        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]

        qs.add_evolution(
            self.H_p(qubits, self.univariate_dict, self.bivariate_dict)
            + penalty_coefficient * self.H_pen(qubits),
            1,
        )

        dwp = DWaveProvider(api_key=self.api_key)

        # local compile only; use safe defaults
        anneal_schedule = [[0, 0], [20, 1]]
        chain_strength = 1.0

        h, J = dwp.compile(qs, anneal_schedule, chain_strength)
        return h, J

    def _calc_penalty_coefficient(self) -> float:
        """
        Mirror DWave/SB style:
        - if user explicitly sets penalty_coefficient, respect it
        - otherwise estimate scale from compiled (S_x + H_p)
        - unary -> penalty_ratio * max_strength
        - non-unary -> 0
        """
        if self.penalty_coefficient != 0:
            return float(self.penalty_coefficient)

        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]
        qs.add_evolution(
            self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict),
            1,
        )

        dwp = DWaveProvider(api_key=self.api_key)
        anneal_schedule = [[0, 0], [20, 1]]
        chain_strength = 1.0
        h, J = dwp.compile(qs, anneal_schedule, chain_strength)

        max_strength = np.max(np.abs(list(h) + list(J.values()))) if (len(h) + len(J)) > 0 else 0.0

        if self.embedding_scheme == "unary":
            return float(self.penalty_ratio * max_strength)
        return 0.0

    def compile(self, info: dict, override=None):
        """
        Compile into Ising (h, J) for OpenJij sampler.
        override: if not None, directly use it as penalty_coefficient
        """
        penalty_coefficient = self._calc_penalty_coefficient()
        if override is not None:
            penalty_coefficient = float(override)

        self._compiled_penalty = penalty_coefficient

        self._log("compile() called")
        self._log("penalty_coefficient =", penalty_coefficient)
        self._log("embedding_scheme   =", self.embedding_scheme)

        start = time.time()
        self.h, self.J = self._compile_to_ising(penalty_coefficient)
        info["compile_time"] = time.time() - start

        self._log("compile_time =", info["compile_time"])
        self._log("num_h =", 0 if self.h is None else len(self.h))
        self._log("num_J =", 0 if self.J is None else len(self.J))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ising_dict_to_dense(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert (h list, J dict) into:
          J_dense: NxN symmetric, zero diagonal
          h_vec:   length N
        """
        h = np.asarray(self.h, dtype=float)
        n = len(h)

        Jdense = np.zeros((n, n), dtype=float)
        for (i, j), val in self.J.items():
            i = int(i)
            j = int(j)
            if i == j:
                continue
            Jdense[i, j] += float(val)
            Jdense[j, i] += float(val)

        np.fill_diagonal(Jdense, 0.0)
        return Jdense, h

    def _build_sampler(self):
        """
        Build OpenJij sampler with raw passthrough kwargs.
        No filtering. No checking.
        """
        if self.sampler_name == "CSQASampler":
            SamplerCls = oj.CSQASampler
        elif self.sampler_name == "SQASampler":
            SamplerCls = oj.SQASampler
        else:
            raise ValueError(
                f"Unsupported sampler_name={self.sampler_name!r}. "
                f"Use 'CSQASampler' or 'SQASampler'."
            )

        self._log("sampler_name =", self.sampler_name)
        self._log("sampler_init_kwargs =", self.sampler_init_kwargs)

        sampler = SamplerCls(**self.sampler_init_kwargs)
        return sampler

    def _format_h_for_openjij(self):
        if isinstance(self.h, (list, tuple, np.ndarray)):
            return {i: float(v) for i, v in enumerate(self.h) if abs(float(v)) > 0}
        return {int(k): float(v) for k, v in self.h.items() if abs(float(v)) > 0}

    def _format_J_for_openjij(self):
        return {
            (int(i), int(j)): float(v)
            for (i, j), v in self.J.items()
            if abs(float(v)) > 0
        }

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def exec(self, verbose: int, info: dict, compile_only: bool = False, override=None) -> List[List[int]]:
        """
        Run OpenJij SA/SQA on compiled Ising problem.
        Return raw_samples as List[List[int]] (bitstrings) for Backend.decoder().
        """
        # compile cache
        need_compile = (self.h is None) or (self.J is None)

        intended_penalty = self._calc_penalty_coefficient()
        if override is not None:
            intended_penalty = float(override)

        if self._compiled_penalty is None or float(intended_penalty) != float(self._compiled_penalty):
            need_compile = True

        if need_compile:
            self.compile(info, override)
        else:
            info["compile_time"] = 0.0
            self._log("reuse cached compiled Ising")

        if compile_only:
            return []

        if verbose > 1:
            self.print_compilation_info()

        if verbose > 1:
            print("Submit Task to OpenJij:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        sampler = self._build_sampler()

        h_for_oj = self._format_h_for_openjij()
        J_for_oj = self._format_J_for_openjij()

        sample_kwargs = dict(self.sample_kwargs)
        sample_kwargs.setdefault("num_reads", self.shots)

        if self.seed is not None and "seed" not in sample_kwargs:
            sample_kwargs["seed"] = self.seed

        self._log("sample_ising h size =", len(h_for_oj))
        self._log("sample_ising J size =", len(J_for_oj))
        self._log("sample_kwargs =", sample_kwargs)

        start_run_time = time.time()
        resp = sampler.sample_ising(h_for_oj, J_for_oj, **sample_kwargs)
        backend_time = time.time() - start_run_time

        info["backend_time"] = backend_time
        info["average_qpu_time"] = 0.000
        info["time_on_machine"] = backend_time
        info["overhead_time"] = info["backend_time"] - info["time_on_machine"]

        if verbose > 1:
            print("Received Task from OpenJij:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        if verbose > 0:
            print(f"Backend runtime: {info['backend_time']}")
            print(f"Overhead Time: {info['overhead_time']}\n")

        self._log("raw response =", resp)

        raw_samples: List[List[int]] = []
        energies: List[float] = []

        for sample, energy in resp.data(["sample", "energy"]):
            spin = np.array([sample[i] for i in range(len(self.h))], dtype=int)  # -1/+1
            bit = spin_to_bitstring(spin)
            raw_samples.append(bit)
            energies.append(float(energy))

            self._log(
                "sample spin =", spin,
                "| bit =", bit,
                "| energy =", float(energy)
            )

        if len(energies) > 0:
            self._log(
                "num_samples =", len(raw_samples),
                "| min_energy =", float(np.min(energies)),
                "| max_energy =", float(np.max(energies)),
            )
        else:
            self._log("no samples returned", force=True)

        return raw_samples

    # ------------------------------------------------------------------
    # info printing
    # ------------------------------------------------------------------
    def print_compilation_info(self):
        print("* Compilation information (OpenJij)")
        print(f"Embedding scheme: {self.embedding_scheme}")
        print(f"Penalty coefficient: {self._compiled_penalty}")
        print(f"Sampler: {self.sampler_name}")
        print(f"Sampler kwargs: {self.sampler_kwargs}")
        print(f"Seed: {self.seed}")
        print(f"Number of shots: {self.shots}")