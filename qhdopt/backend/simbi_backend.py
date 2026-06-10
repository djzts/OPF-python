# qhdopt/backend/sb_backend.py
from __future__ import annotations

from typing import Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time
import numpy as np

from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider

from qhdopt.backend.backend import Backend
from qhdopt.utils.decoding_utils import spin_to_bitstring  # 你 openjij_backend.py 里同款用法


def _sb_minimize_worker(payload: dict) -> dict:
    """
    Run one independent SB sampling job in a child process.

    The payload only contains plain Python/numpy values so it can be pickled by
    multiprocessing. The returned samples are already converted to raw 0/1
    bitstrings expected by Backend.decoder().
    """
    import torch
    import simulated_bifurcation as sb

    gpu_id = int(payload["gpu_id"])
    device = f"cuda:{gpu_id}"
    agents = int(payload["agents"])
    seed = payload.get("seed")
    verbose = int(payload.get("verbose", 0))

    kwargs = dict(payload["kwargs"])
    kwargs["device"] = device
    kwargs["agents"] = agents

    if seed is not None:
        worker_seed = int(seed) + gpu_id
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)

    start = time.time()
    with torch.cuda.device(gpu_id):
        M = torch.tensor(payload["M"], dtype=torch.float32, device=device)
        v = torch.tensor(payload["v"], dtype=torch.float32, device=device)
        all_vectors, _ = sb.minimize(M, v, 0.0, **kwargs)
        torch.cuda.synchronize(gpu_id)
        spins = all_vectors.detach().cpu().numpy().astype(int)
    runtime = time.time() - start

    if spins.ndim == 1:
        spins = spins.reshape(1, -1)

    raw_samples = [spin_to_bitstring(row.tolist()) for row in spins]
    if verbose > 0:
        print(
            f"SimulatedBifurcation worker cuda:{gpu_id}: "
            f"agents={agents}, runtime={runtime:.3f}s, samples={len(raw_samples)}"
        )

    return {
        "gpu_id": gpu_id,
        "agents": agents,
        "runtime": runtime,
        "raw_samples": raw_samples,
        "spins": spins,
    }


class SimulatedBifurcationBackend(Backend):
    """
    Local backend using Simulated Bifurcation (SB), with optional GPU via torch device='cuda'.

    Contract:
      exec() returns raw_samples as List[List[int]] (0/1 bitstrings),
      so it can be decoded by Backend.decoder().
    """

    def __init__(
        self,
        resolution,
        dimension,
        univariate_dict,
        bivariate_dict,
        shots=100,
        embedding_scheme="unary",
        anneal_schedule=None,
        penalty_coefficient=0.0,
        penalty_ratio=0.75,
        chain_strength_ratio=1.05,
        api_key='DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291',
        # ---- SB minimize keyword-only params ----
        domain="spin",
        device: str = "cuda",          # "cpu" or "cuda"
        agents: int | None = None,    # SB 的并行 agent 数
        max_steps: int | None = None,
        ballistic: bool = False,      # "ballistic" or "discrete"
        heated: bool | None = None,
        best_only: bool = False,      # False => 拿到每个 agent 的解，更像“shots”
        seed: int | None = None,
        multi_gpu: bool = False,
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        **sb_kwargs,
    ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict, bivariate_dict)

        self.anneal_schedule = anneal_schedule
        if self.anneal_schedule is None:
            self.anneal_schedule = [[0, 0], [20, 1]]
        self.chain_strength_ratio = chain_strength_ratio
        self.penalty_coefficient = float(penalty_coefficient)
        self.penalty_ratio = float(penalty_ratio)

        self.domain = domain
        self.device = device
        self.agents = agents
        self.max_steps = max_steps
        self.ballistic = ballistic
        self.heated = heated
        self.best_only = best_only
        self.seed = seed
        self.multi_gpu = bool(multi_gpu)
        self.num_gpus = num_gpus
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.sb_kwargs = sb_kwargs

        # compiled Ising params
        self.h = None  # list/np array length N
        self.J = None  # dict {(i,j): val}
        self._compiled_penalty = None
        self.apikey = api_key

    # ---------- compilation: same idea as OpenJijBackend ----------
    def _compile_to_ising(self, penalty_coefficient: float) -> Tuple[List[float], dict]:
        """
        Use SimuQ's DWaveProvider compiler to convert QSystem to Ising (h, J).
        Local compilation only (no device submission).
        """
        """
        Calculates the penalty coefficient and chain strength using self.penalty_ratio.
        """
        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]

        qs.add_evolution(
            self.H_p(qubits, self.univariate_dict, self.bivariate_dict)
            + penalty_coefficient * self.H_pen(qubits),
            1
        )

        dwp = DWaveProvider(api_key = self.apikey)
        chain_strength = 1.0
        h, J = dwp.compile(qs, self.anneal_schedule, chain_strength)
        return h, J
    """

    
    """
    def _calc_penalty_coefficient_modified(self) -> float:
        # SB 本地解不需要 chain_strength
        if self.penalty_coefficient != 0:
            return float(self.penalty_coefficient)

        # 如果你确实需要 unary penalty，就用系数量级估计一下
        # （不再通过 DWave compile）
        max_strength = 0.0
        if self.univariate_dict:
            max_strength = max(max_strength, max(abs(v) for v in self.univariate_dict.values()))
        if self.bivariate_dict:
            max_strength = max(max_strength, max(abs(v) for v in self.bivariate_dict.values()))

        if self.embedding_scheme == "unary":
            return float(self.penalty_ratio * max_strength)
        return 0.0
    """
    
    #""" #Old: rely on dwave API
    def _calc_penalty_coefficient(self) -> float:
        """
        Mirror DwaveBackend logic: if user sets penalty_coefficient use it;
        else penalty_ratio * max_strength for unary, else 0.
        """
        if self.penalty_coefficient != 0:
            return float(self.penalty_coefficient)

        # build without penalty first to estimate scale
        qs = QSystem()
        qubits = [Qubit(qs, name=f"Q{i}") for i in range(len(self.qubits))]
        qs.add_evolution(self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict), 1)

        dwp = DWaveProvider(api_key = self.apikey)
        chain_strength = 1.0
        h, J = dwp.compile(qs, self.anneal_schedule, chain_strength)

        max_strength = np.max(np.abs(list(h) + list(J.values()))) if (len(h) + len(J)) > 0 else 0.0
        if self.embedding_scheme == "unary":
            return float(self.penalty_ratio * max_strength)
        return 0.0
    #"""

    def compile(self, info: dict, override=None):
        penalty_coefficient = self._calc_penalty_coefficient()
        if override is not None:
            penalty_coefficient = float(override)
        self._compiled_penalty = penalty_coefficient

        start = time.time()
        self.h, self.J = self._compile_to_ising(penalty_coefficient)
        info["compile_time"] = time.time() - start

    # ---------- helpers ----------
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
            i = int(i); j = int(j)
            if i == j:
                continue
            Jdense[i, j] += float(val)
            Jdense[j, i] += float(val)  # enforce symmetry

        np.fill_diagonal(Jdense, 0.0)
        return Jdense, h

    def _base_sb_kwargs(self) -> dict:
        kwargs = dict(self.sb_kwargs)
        kwargs["input_type"] = "spin"
        kwargs["best_only"] = self.best_only

        if self.max_steps is not None:
            kwargs["max_steps"] = int(self.max_steps)
        if self.ballistic is not None:
            kwargs["ballistic"] = bool(self.ballistic)
        if self.heated is not None:
            kwargs["heated"] = bool(self.heated)

        return kwargs

    def _split_agents(self, agents: int, num_workers: int) -> List[int]:
        base = agents // num_workers
        remainder = agents % num_workers
        return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]

    def _selected_gpu_ids(self, torch, verbose: int) -> List[int]:
        if not self.multi_gpu:
            return []

        device_name = str(self.device).lower()
        if not device_name.startswith("cuda"):
            if verbose > 0:
                print(
                    "Warning: SimulatedBifurcationBackend multi_gpu=True was requested "
                    f"but device={self.device!r} is not CUDA; falling back to single-process execution."
                )
            return []

        if not torch.cuda.is_available():
            if verbose > 0:
                print(
                    "Warning: SimulatedBifurcationBackend multi_gpu=True was requested "
                    "but no CUDA device is available; falling back to single-process execution."
                )
            return []

        visible_count = int(torch.cuda.device_count())
        if visible_count <= 0:
            if verbose > 0:
                print(
                    "Warning: SimulatedBifurcationBackend multi_gpu=True was requested "
                    "but torch.cuda.device_count() returned 0; falling back to single-process execution."
                )
            return []

        if self.gpu_ids is not None:
            selected = [int(gpu_id) for gpu_id in self.gpu_ids]
            invalid = [gpu_id for gpu_id in selected if gpu_id < 0 or gpu_id >= visible_count]
            if invalid:
                raise ValueError(
                    f"gpu_ids contains ids outside the visible CUDA range 0..{visible_count - 1}: {invalid}"
                )
        else:
            selected = list(range(visible_count))
            if self.num_gpus is not None:
                requested = int(self.num_gpus)
                if requested <= 0:
                    if verbose > 0:
                        print(
                            "Warning: num_gpus <= 0 with multi_gpu=True; "
                            "falling back to single-process execution."
                        )
                    return []
                selected = selected[:min(requested, visible_count)]

        if len(selected) <= 1:
            if verbose > 0:
                print(
                    "Warning: SimulatedBifurcationBackend multi_gpu=True needs at least "
                    f"2 selected GPUs, got {len(selected)}; falling back to single-process execution."
                )
            return []

        if verbose > 0:
            print(
                "SimulatedBifurcationBackend multi-GPU: "
                f"detected {visible_count} visible CUDA GPU(s); using {len(selected)} GPU(s): "
                + ", ".join(f"cuda:{gpu_id}" for gpu_id in selected)
            )

        return selected

    def _run_single_sb(self, torch, sb, Jdense: np.ndarray, hvec: np.ndarray, agents: int):
        kwargs = self._base_sb_kwargs()

        if str(self.device).lower().startswith("cuda") and not torch.cuda.is_available():
            dev = torch.device("cpu")
            kwargs["device"] = "cpu"
        else:
            dev = torch.device(self.device)
            kwargs["device"] = self.device

        M = torch.tensor(0.5 * Jdense, dtype=torch.float32, device=dev)
        v = torch.tensor(hvec, dtype=torch.float32, device=dev)

        kwargs["agents"] = agents

        if self.seed is not None:
            try:
                torch.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))
            except Exception:
                pass

        all_vectors, all_values = sb.minimize(M, v, 0.0, **kwargs)
        return all_vectors, all_values

    def _run_multi_gpu_sb(
        self,
        Jdense: np.ndarray,
        hvec: np.ndarray,
        agents: int,
        gpu_ids: List[int],
        verbose: int,
        info: dict,
    ) -> List[List[int]]:
        agent_splits = self._split_agents(agents, len(gpu_ids))
        tasks = [
            (gpu_id, agent_count)
            for gpu_id, agent_count in zip(gpu_ids, agent_splits)
            if agent_count > 0
        ]

        if len(tasks) <= 1:
            return []

        kwargs = self._base_sb_kwargs()
        worker_payloads = []
        for gpu_id, agent_count in tasks:
            if verbose > 0:
                print(f"SimulatedBifurcationBackend multi-GPU assignment: cuda:{gpu_id} agents={agent_count}")
            worker_payloads.append(
                {
                    "gpu_id": gpu_id,
                    "agents": agent_count,
                    "M": 0.5 * Jdense,
                    "v": hvec,
                    "kwargs": kwargs,
                    "seed": self.seed,
                    "verbose": verbose,
                }
            )

        start_run_time = time.time()
        raw_samples_by_gpu = {}
        spins_by_gpu = {}
        runtimes = {}
        agents_by_gpu = {}

        start_method = os.environ.get("QHDOPT_SB_MP_START_METHOD")
        if start_method is None:
            start_method = "spawn" if os.name == "nt" else "fork"
        context = mp.get_context(start_method)

        with ProcessPoolExecutor(max_workers=len(worker_payloads), mp_context=context) as executor:
            futures = [executor.submit(_sb_minimize_worker, payload) for payload in worker_payloads]
            for future in as_completed(futures):
                result = future.result()
                gpu_id = int(result["gpu_id"])
                raw_samples_by_gpu[gpu_id] = result["raw_samples"]
                spins_by_gpu[gpu_id] = result["spins"]
                runtimes[gpu_id] = float(result["runtime"])
                agents_by_gpu[gpu_id] = int(result["agents"])

        backend_time = time.time() - start_run_time

        raw_samples: List[List[int]] = []
        spin_arrays = []
        for gpu_id, _ in tasks:
            raw_samples.extend(raw_samples_by_gpu.get(gpu_id, []))
            if gpu_id in spins_by_gpu:
                spin_arrays.append(spins_by_gpu[gpu_id])
            if verbose > 0:
                print(
                    f"SimulatedBifurcationBackend multi-GPU runtime: "
                    f"cuda:{gpu_id} agents={agents_by_gpu.get(gpu_id, 0)} "
                    f"runtime={runtimes.get(gpu_id, 0.0):.3f}s "
                    f"samples={len(raw_samples_by_gpu.get(gpu_id, []))}"
                )

        self.simbi_solution_vector = np.vstack(spin_arrays) if spin_arrays else np.empty((0, len(hvec)), dtype=int)

        info["backend_time"] = backend_time
        info["average_qpu_time"] = 0.000
        info["time_on_machine"] = backend_time
        info["overhead_time"] = info["backend_time"] - info["time_on_machine"]
        info["simbi_multi_gpu"] = True
        info["simbi_gpu_ids"] = list(gpu_ids)
        info["simbi_agents_per_gpu"] = {f"cuda:{gpu_id}": agents_by_gpu[gpu_id] for gpu_id, _ in tasks}
        info["simbi_gpu_runtime"] = {f"cuda:{gpu_id}": runtimes[gpu_id] for gpu_id, _ in tasks}

        if verbose > 0:
            print("backend:", backend_time, "s; num_samples:", len(raw_samples))

        return raw_samples

    # ---------- exec ----------
    def exec(self, verbose: int, info: dict, compile_only: bool = False, override=None) -> List[List[int]]:
        """
        Run Simulated Bifurcation on compiled Ising problem.
        Return raw_samples as List[List[int]] (bitstrings) for Backend.decoder().
        """
        # --- compile cache: only compile when needed (safer) ---
        need_compile = (self.h is None) or (self.J is None)

        # compute intended penalty for this run
        intended_penalty = self._calc_penalty_coefficient()
        if override is not None:
            intended_penalty = float(override)

        if self._compiled_penalty is None or float(intended_penalty) != float(self._compiled_penalty):
            need_compile = True

        if need_compile:
            self.compile(info, override)
        else:
            info["compile_time"] = 0.0
        if compile_only:
            return []
        

        if verbose > 1:
            self.print_compilation_info()

        if verbose > 1:
            print("Submit Task to Simulated Bifurcation:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # optional dependency: simulated-bifurcation + torch
        try:
            import torch
            import simulated_bifurcation as sb
        except Exception as e:
            raise ImportError(
                "SimulatedBifurcationBackend requires `simulated-bifurcation` and `torch`.\n"
                "Install: pip install simulated-bifurcation  (and torch with CUDA if you want GPU)"
            ) from e

        Jdense, hvec = self._ising_dict_to_dense()
        n = len(hvec)

        # pick agents: if user didn't set, use shots
        agents = int(self.agents) if self.agents is not None else int(self.shots)

        # domain guard (decoder assumes spin)
        if self.domain != "spin":
            raise ValueError("SimulatedBifurcationBackend currently supports domain='spin' only.")

        gpu_ids = self._selected_gpu_ids(torch, verbose)
        if gpu_ids:
            raw_samples = self._run_multi_gpu_sb(Jdense, hvec, agents, gpu_ids, verbose, info)
            if raw_samples:
                return raw_samples

        kwargs = dict(self.sb_kwargs)

        if str(self.device).lower().startswith("cuda") and not torch.cuda.is_available():
            dev = torch.device("cpu")
            kwargs["device"] = "cpu"
        else:
            dev = torch.device(self.device)
            kwargs["device"] = self.device

        # Map Ising energy:
        #   E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i
        # sb.minimize expects polynomial:
        #   s^T M s + v^T s + c
        # With Jdense symmetric and zero diagonal,
        #   s^T (0.5 * Jdense) s = sum_{i<j} J_ij s_i s_j
        # => choose M = 0.5 * Jdense, v = h, c = 0

        M = torch.tensor(0.5 * Jdense, dtype=torch.float32, device=dev)
        v = torch.tensor(hvec, dtype=torch.float32, device=dev)

        kwargs["input_type"] = "spin"
        kwargs["agents"] = agents
        kwargs["best_only"] = self.best_only

        if self.max_steps is not None:
            kwargs["max_steps"] = int(self.max_steps)
        if self.ballistic is not None:
            kwargs["ballistic"] = bool(self.ballistic)
        if self.heated is not None:
            kwargs["heated"] = bool(self.heated)

        # (seed) 这个包是否接受 seed 参数取决于版本；不保证一定支持
        # 如果你需要严格可复现，我们可以再针对你装的版本做适配。
        if self.seed is not None:
            try:
                torch.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))
            except Exception:
                pass

        # 注意：sb.minimize 返回顺序看文档： value, vector  (best_only=True)
        # best_only=False 时会返回 vectors, values

        """
        (matrix: Tensor | ndarray, vector: Tensor | ndarray | None = None, 
        constant: int | float | None = None, input_type: str = "spin", dtype: dtype = torch.float32, 
        device: str = "cpu", agents: int = 128, max_steps: int = 10000, 
        best_only: bool = True, ballistic: bool = False, heated: bool = False, 
        verbose: bool = True, *, use_window: bool = True, sampling_period: int = 50, 
        convergence_threshold: int = 50) -> Tuple[Tensor, float | Tensor])
        """
        
        
        #poly_model = sb.build_model(M, vector=v, constant=0, **kwargs)
        
        start_run_time = time.time()
        #all_vectors, all_values = poly_model.minimize()
        all_vectors, all_values = sb.minimize(M, v, 0.0, **kwargs)
        backend_time = time.time() - start_run_time

        # Generate fake QPU/machine timing values
        info["backend_time"] = backend_time
        info["average_qpu_time"] = 0.000
        info["time_on_machine"] = backend_time
        info["overhead_time"] = info["backend_time"] - info["time_on_machine"]
        info["simbi_multi_gpu"] = False
        

        spins = all_vectors.detach().cpu().numpy().astype(int)   # ✅ numpy array 
        #print(all_vectors)
        self.simbi_solution_vector = all_vectors

        if spins.ndim == 1:
            # best_only=True 时只返回一个解，手动变成一条样本
            spins = spins.reshape(1, -1)

        # 逐条从 -1/+1 spin → 0/1 bitstring
        raw_samples: List[List[int]] = []
        for row in spins:
            spin_list = row.tolist()           # e.g. [-1, 1, -1, 1, ...]
            bit_list = spin_to_bitstring(spin_list)  # e.g. [0, 1, 0, 1, ...]
            raw_samples.append(bit_list)

        if verbose > 0:
            print("backend:", backend_time, "s; num_samples:", len(raw_samples))


        """
        if self.best_only:
            value, s = out
            s = np.asarray(s, dtype=int).reshape(-1)
            if s.size != n:
                raise ValueError(f"SB returned vector of size {s.size}, expected {n}")
            raw_samples.append(spin_to_bitstring(s.tolist()))
            
            raw_samples = all_vectors.cpu().numpy().astype(int).tolist()
        else:
            vectors, values = out
            # vectors: (agents, n)
            vectors = np.asarray(vectors, dtype=int)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for k in range(vectors.shape[0]):
                s = vectors[k].reshape(-1)
                raw_samples.append(spin_to_bitstring(s.tolist()))
            raw_samples = all_vectors.cpu().numpy().astype(int).tolist()
        """
        #print(raw_samples)
        return raw_samples

    def print_compilation_info(self):
        print("* Compilation information")
        print("Final Hamiltonian:")
        print("(Feature under development; only the Hamiltonian is meaningful here)")
        print(self.qs)
        print(f"Annealing schedule parameter: {self.anneal_schedule}")
        print(f"Penalty coefficient: {self.penalty_coefficient}")
        #print(f"Chain strength: {self.chain_strength}")
        print(f"Number of shots: {self.shots}")
