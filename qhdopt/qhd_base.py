import time
from typing import List, Dict, Union, Optional

from sympy import lambdify, Symbol, Function
import jax.numpy as jnp

from qhdopt.backend import dwave_backend, ionq_backend, qutip_backend
from qhdopt.backend import openjij_backend, simbi_backend, guorbi_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import decompose_function


class QHD_Base:
    """
    Provides functionality to run Quantum Hamiltonian Gradient Descent as introduced
    by https://arxiv.org/pdf/2303.01471.pdf

    A user should initialize QHD through the use of the functions: QHD_Base.QP and QHD_Base_Sympy
    """
    def __init__(
        self,
        func: Function,
        syms: List[Symbol],
        info: Dict[str, Union[int, float, str]]
    ):
        """
        Initializes the QHD_Base class.

        Args:
            func: The function to be optimized.
            syms: The list of sympy Symbols representing the variables of the function.
            info: Dictionary to store miscellaneous information about the optimization process.
        """
        self.func = func
        self.syms = syms
        self.dimension = len(syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(self.func, self.syms)
        lambda_numpy = lambdify(syms, func, jnp)
        self.f_eval = lambda x: lambda_numpy(*x)
        self.info = info

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
    ) -> None:
        """
        Sets up the D-Wave backend for quantum optimization.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of sampling shots for the D-Wave device.
            api_key: API key for accessing D-Wave services.
            api_key_from_file: Path to a file containing the API key.
            embedding_scheme: Embedding scheme for problem mapping.
            anneal_schedule: Custom annealing schedule.
            penalty_coefficient: Coefficient for penalty terms.
            penalty_ratio: Ratio of penalty terms in the objective function.
            chain_strength_ratio: Ratio of strength of chains in embedding.
        """
        self.backend = dwave_backend.DWaveBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            chain_strength_ratio=chain_strength_ratio
        )

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
        on_simulator: bool = False,
    ) -> None:
        """
        Sets up the IonQ backend for quantum optimization.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of sampling shots for the IonQ device.
            api_key: API key for accessing IonQ services.
            api_key_from_file: Path to a file containing the API key.
            embedding_scheme: Embedding scheme for problem mapping.
            penalty_coefficient: Coefficient for penalty terms.
            time_discretization: Number of time steps for discretization.
            gamma: Coefficient for transverse field in quantum annealing.
            on_simulator: Flag to run on simulator instead of actual device.
        """
        self.backend = ionq_backend.IonQBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            on_simulator=on_simulator,
            gamma=gamma,
        )

    def qutip_setup(
        self,
        resolution: int,
        shots: int = 100,
        embedding_scheme: str = "onehot",
        penalty_coefficient: float = 0,
        time_discretization: int = 10,
        gamma: float = 5
    ) -> None:
        """
        Sets up the QuTiP backend for quantum simulation.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of simulation shots.
            embedding_scheme: Embedding scheme for problem mapping.
            penalty_coefficient: Coefficient for penalty terms.
            time_discretization: Number of time steps for discretization.
            gamma: Coefficient for transverse field in quantum annealing.
        """
        self.backend = qutip_backend.QuTiPBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
        )
    
    def openjij_setup(
        self,
        resolution: int,
        shots: int = 100,
        embedding_scheme: str = "unary",
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        sampler_name: str = "SQASampler",
        seed: Optional[int] = None,
        debug: bool = False,
        sampler_init_kwargs: Optional[dict] = None,
        sample_kwargs: Optional[dict] = None,
    ) -> None:
        self.backend = openjij_backend.OpenJijBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
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

    def sb_setup(
        self,
        resolution: int,
        shots: int = 100,
        embedding_scheme: str = "unary",
        anneal_schedule: Optional[List[List[int]]] = None,
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        chain_strength_ratio: float = 1.05,
        api_key='DEV-a3f87cd2fb51d10601c4e8bd16114d92614fc291',
        # ---- SB minimize params ----
        domain: str = "spin",
        device: str = "cuda",
        agents: Optional[int] = None,
        max_steps: Optional[int] = 2000,
        ballistic: bool = False,   # 改这里，SB 默认不开 ballistic 模式
        heated: Optional[bool] = None,
        best_only: bool = False,
        seed: Optional[int] = None,
        multi_gpu: bool = False,
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        # ---- advanced SB kwargs passthrough ----
        **sb_kwargs,
    ) -> None:
        """
        Sets up the Simulated Bifurcation (SB) backend for local optimization.
        """
        self.backend = simbi_backend.SimulatedBifurcationBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
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

    def gurobi_setup(
        self,
        resolution: int,
        shots: int = 100,
        embedding_scheme: str = "unary",
        anneal_schedule: Optional[List[List[int]]] = None,
        penalty_coefficient: float = 0.0,
        penalty_ratio: float = 0.75,
        chain_strength: float = 1.0,
        api_key: Optional[str] = None,
        api_key_from_file: Optional[str] = None,
        solver_mode: str = "ising",
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        log_to_console: bool = False,
    ) -> None:
        """
        Sets up the Gurobi backend for local exact optimization.
        """
        self.backend = guorbi_backend.GurobiBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
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

    
    def compile_only(self):
        self.backend.compile(self.info)
        return self.backend

    def optimize(
            self,
            verbose: int = 0,
            override=None,
    ) -> Optional[Response]:
        """
        Executes the optimization process.

        Args:
            verbose: Verbosity level (0, 1, 2 for increasing detail).

        Returns:
            An instance of Response containing optimization results, None if compile_only is True.
        """
        raw_samples = self.backend.exec(verbose=verbose, info=self.info, override=override)


        start_time_decoding = time.time()
        coarse_minimizer, coarse_minimum, self.decoded_samples = self.backend.decoder(raw_samples,
                                                                                      self.f_eval)

        end_time_decoding = time.time()
        self.info["decoding_time"] = end_time_decoding - start_time_decoding
        qhd_response = Response(self.info, self.decoded_samples, coarse_minimum, coarse_minimizer)

        return qhd_response
