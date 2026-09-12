from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

from numpy import ndarray
from simuq import QSystem, hlist_sum, Qubit, TIHamiltonian
import numpy as np

from qhdopt.utils.decoding_utils import bitstring_to_vec, spin_to_bitstring


class Backend(ABC):
    """
    Abstract backend class which defines common functions for all backends and
    an abstract function: exec which each backend needs to implement.
    """
    def __init__(self, resolution, dimension, shots, embedding_scheme, univariate_dict, bivariate_dict):
        self.resolution = resolution
        self.dimension = dimension
        self.qs = QSystem()
        self.qubits = [Qubit(self.qs, name=f'Q{i}') for i in range(self.dimension * self.resolution)]
        if shots == None :
            shots = 100
        self.shots = shots
        self.embedding_scheme = embedding_scheme
        self.univariate_dict = univariate_dict
        self.bivariate_dict = bivariate_dict

    def S_x(self, qubits: List[Qubit]) -> TIHamiltonian:
        """
        Generates the hamiltonian S_x hamiltonian as defined
        in https://arxiv.org/pdf/2303.01471.pdf equation (F.38)

        Args:
            qubits: List of qubits

        Returns:
            TIHamiltonian: S_x Hamiltonian
        """
        return hlist_sum([qubit.X for qubit in qubits])

    def unary_penalty(self, k: int, qubits: List[Qubit]):
        """
        Generates the unary penalty hamiltonian on the kth sub-system
        as defined in https://arxiv.org/pdf/2401.08550.pdf
        """
        unary_penalty_sum = lambda k: sum(
            [
                qubits[j].Z * qubits[j + 1].Z
                for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
            ]
        )
        return (
                (-1) * qubits[k * self.resolution].Z
                + qubits[(k + 1) * self.resolution - 1].Z
                - unary_penalty_sum(k)
        )

    def H_pen(self, qubits: List[Qubit]) -> TIHamiltonian:
        """
        Generates the penalty hamiltonian across all dimensions of the problem
        uses unary_penalty as a key subroutine.

        Args:
            qubits: List of qubits

        Returns:
            TIHamiltonian: Penalty Hamiltonian
        """
        if self.embedding_scheme in ("hamming", "binary"):
            # Binary (standard base-2) encoding needs no penalty: every
            # computational basis state is a valid codeword, unlike
            # unary/one-hot which reserve most of the Hilbert space for
            # invalid patterns.
            return 0
        elif self.embedding_scheme == "unary":
            return hlist_sum(
                [self.unary_penalty(p, qubits) for p in range(self.dimension)]
            )

    def _fit_quadratic_from_callable(self, lmda, tol: float = 1e-6):
        """
        Recover (a, b, c) such that lmda(x) ~= a*x**2 + b*x + c on [0, 1] from
        three samples (x=0, 0.5, 1), then verify against a fourth point
        (x=0.25).

        The binary embedding scheme builds the problem Hamiltonian directly
        from these three coefficients (see `_binary_variable_hamiltonian`)
        rather than from the resolution-point interpolation trick used for
        unary/one-hot. That only works when the univariate/bivariate factor
        supplied by `decompose_function` truly is a degree<=2 polynomial of
        its variable -- which is what the linearized ALM Lagrangian produces
        (objective + rho/2 * h_linearized**2 is quadratic overall). Anything
        of higher degree raises a clear error instead of silently producing a
        wrong Hamiltonian; use embedding_scheme="unary" for such functions.
        """
        y0, y1, y2 = float(lmda(0.0)), float(lmda(0.5)), float(lmda(1.0))
        c = y0
        a = 2.0 * y0 - 4.0 * y1 + 2.0 * y2
        b = -3.0 * y0 + 4.0 * y1 - y2

        check_x = 0.25
        predicted = a * check_x ** 2 + b * check_x + c
        actual = float(lmda(check_x))
        if abs(predicted - actual) > tol * max(1.0, abs(actual)):
            raise ValueError(
                "embedding_scheme='binary' requires every univariate/bivariate "
                "factor of the problem to be a degree<=2 polynomial of its "
                "variable (as produced by the linearized ALM Lagrangian). "
                f"Got a function with f(0)={y0:.6g}, f(0.5)={y1:.6g}, "
                f"f(1)={y2:.6g} that does not fit a quadratic "
                f"(predicted f(0.25)={predicted:.6g}, actual={actual:.6g}). "
                "Use embedding_scheme='unary' instead for higher-degree or "
                "non-polynomial functions."
            )
        return a, b, c

    def _binary_variable_hamiltonian(
        self, qubits: List[Qubit], d: int, a: float, b: float, c: float
    ) -> TIHamiltonian:
        """
        Builds the operator a*x_d**2 + b*x_d + c for a single dimension d
        under the binary embedding, where

            x_d = sum_j w_j * n_j,   w_j = 2**j / (2**resolution - 1)

        and n_j = 0.5*(I - Z_j) is the 0/1 occupation operator for bit j of
        dimension d (bit j == 1 means qubit j is in the "one" state).

        x_d**2 is expanded analytically using the boolean identity n_j**2 ==
        n_j (n_j is a 0/1-valued diagonal operator) *before* building the
        operator, rather than by literally squaring the SimuQ operator for
        x_d. This avoids ever multiplying two operators that act on the same
        qubit -- every product below is between different qubits j != k,
        which is the same pattern the bivariate branch of `get_ham` already
        relies on (products are only ever taken across disjoint qubit sets).
        """
        r = self.resolution
        denom = float((1 << r) - 1) if r > 1 else 1.0
        weights = [((1 << j) / denom) for j in range(r)]

        def n(j):
            qubit = qubits[(d - 1) * r + j]
            return 0.5 * (qubit.I - qubit.Z)

        H = c * qubits[(d - 1) * r].I

        for j in range(r):
            coeff = b * weights[j] + a * weights[j] ** 2
            if coeff != 0.0:
                H += coeff * n(j)

        if a != 0.0:
            for j in range(r):
                for k in range(j + 1, r):
                    H += (2.0 * a * weights[j] * weights[k]) * (n(j) * n(k))

        return H

    def H_p(self, qubits: List[Qubit], univariate_dict: dict, bivariate_dict: dict) -> TIHamiltonian:
        """
        Generates the problem hamiltonian, as defined in https://arxiv.org/pdf/2303.01471.pdf (F.24)
        for the hamming embedding, and modified for the unary and one-hot embedding in ways that can
        be found in https://arxiv.org/pdf/2401.08550.pdf.

        Args:
            qubits: List of qubits
            univariate_dict: Dictionary of univariate terms
            bivariate_dict: Dictionary of bivariate terms

        Returns:
            TIHamiltonian: Problem Hamiltonian
        """
        # Encoding of the X operator as defined in https://browse.arxiv.org/pdf/2303.01471.pdf (F.16)
        def Enc_X(k):
            S_z = lambda k: sum(
                [qubits[j].Z for j in range(k * self.resolution, (k + 1) * self.resolution)]
            )
            return (1 / 2) + (-1 / (2 * self.resolution)) * S_z(k)

        def get_ham(d, lmda):
            def n_j(d, j):
                return 0.5 * (
                        qubits[(d - 1) * self.resolution + j].I - qubits[
                    (d - 1) * self.resolution + j].Z
                )

            if self.embedding_scheme == "unary":

                def eval_lmda_unary():
                    eval_points = [i / self.resolution for i in range(self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_unary()
                H = eval_lmda[0] * qubits[(d - 1) * self.resolution].I
                for i in range(len(eval_lmda) - 1):
                    H += (eval_lmda[i + 1] - eval_lmda[i]) * n_j(d, self.resolution - i - 1)

                return H

            elif self.embedding_scheme == "onehot":

                def eval_lmda_onehot():
                    eval_points = [i / self.resolution for i in range(1, self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_onehot()
                H = 0
                for i in range(len(eval_lmda)):
                    H += eval_lmda[i] * n_j(d, self.resolution - i - 1)

                return H

            elif self.embedding_scheme == "binary":
                a, b, c = self._fit_quadratic_from_callable(lmda)
                return self._binary_variable_hamiltonian(qubits, d, a, b, c)

        H: TIHamiltonian = 0
        for key, value in univariate_dict.items():
            coefficient, lmda = value
            if self.embedding_scheme == "hamming":
                H += coefficient*lmda(Enc_X(key - 1))
            else:
                ham = get_ham(key, lmda)
                H += coefficient * ham

        for key, value in bivariate_dict.items():
            d1, d2 = key
            for term in value:
                coefficient, lmda1, lmda2 = term
                if self.embedding_scheme == "hamming":
                    H += coefficient * lmda1(Enc_X(d1 - 1)) * lmda2(Enc_X(d2 - 1))
                else:
                    H += coefficient * (get_ham(d1, lmda1) * get_ham(d2, lmda2))

        return H

    def _sample_to_bitstring(self, sample: List[int]) -> List[int]:
        """
        Normalize backend output into a 0/1 bitstring for decoding.

        Backends may return either:
        - binary samples in {0, 1}
        - spin samples in {-1, +1}
        """
        if sample is None:
            return None

        normalized = [int(v) for v in sample]
        sample_set = set(normalized)

        if sample_set.issubset({0, 1}):
            return normalized
        if sample_set.issubset({-1, 1}):
            return spin_to_bitstring(normalized)

        raise ValueError(f"Unsupported sample values for decoding: {sample_set}")

    def decoder(self, raw_samples: List[int], f_eval: Callable) -> Tuple[ndarray, int, List[ndarray]]:
        """
        decodes the raw samples returned from the backend into samples
        which are the form (a_1, a_2,...,a_d) where d is the dimension
        of the problem and a_j is a number between 0 and 1.

        Args:
            raw_samples: List of raw samples
            f_eval: Function to evaluate the samples

        Returns:
            Tuple: minimizer, minimum, qhd_samples
        """
        qhd_samples = []
        minimizer = np.zeros(self.dimension)
        minimum = float("inf")

        for i in range(len(raw_samples)):
            bitstring = self._sample_to_bitstring(raw_samples[i])
            qhd_samples.append(bitstring_to_vec(self.embedding_scheme, bitstring, self.dimension, self.resolution))
            if qhd_samples[i] is None:
                continue
            new_f = float(f_eval(qhd_samples[i]))
            if new_f < minimum:
                minimum = new_f
                minimizer = qhd_samples[i]

        return minimizer, minimum, qhd_samples

    def H_k(self, qubits: List[Qubit] = None) -> TIHamiltonian:
        if qubits is None:
            qubits = self.qubits
        if self.embedding_scheme == "onehot":

            def onehot_driving_sum(k):
                return sum(
                    [
                        0.5
                        * (
                                qubits[j].X * qubits[j + 1].X
                                + qubits[j].Y * qubits[j + 1].Y
                        )
                        for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
                    ]
                )

            return (-0.5 * self.resolution ** 2) * hlist_sum(
                [onehot_driving_sum(p) for p in range(self.dimension)]
            )
        else:
            return (-0.5 * self.resolution ** 2) * hlist_sum([qubit.X for qubit in qubits])

    def compile(self, info):
        """
        Compiles the problem description into a format that the backend can run.

        Args:
            info: Dictionary to store information about the compilation
        """
        pass

    @abstractmethod
    def exec(self, verbose: int, info: dict, override=None):
        """
        Executes the quantum backend to run QHD

        Args:
            verbose: Verbosity level
            info: Dictionary to store information about the execution
        """
        pass
