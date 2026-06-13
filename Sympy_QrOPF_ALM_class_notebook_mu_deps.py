#!/usr/bin/env python
# coding: utf-8

"""Small compatibility helpers for Sympy_QrOPF_ALM_class_notebook_mu.py."""

import numpy as np

from Sympy_QrOPF_ALM_class import (
    PrintQHDACOPFResults,
    SympyACOPFModel,
    append_qhd_acopf_results,
    format_qhd_acopf_results,
    initialize_qhd_acopf_log,
    solve_with_gurobi_from_sympy,
)


def ensure_1d_decision_vector(x, expected_len=None, name="x"):
    """Convert a solver output into a flat float vector and validate its size."""
    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        raise ValueError(
            f"{name} must be a 1D decision vector, but got a scalar-like value: {arr!r}. "
            "This usually means the solver result field was None or a single number."
        )

    arr = arr.reshape(-1)

    if expected_len is not None and arr.size != expected_len:
        raise ValueError(
            f"{name} must have length {expected_len}, but got shape {arr.shape} "
            f"with {arr.size} entries."
        )

    return arr


def extract_qhd_solution_vector(response, prefer_refined=True, expected_len=None):
    """
    Extract a usable 1D minimizer vector from a QHD response object.

    If refined minimization is unavailable, fall back to the coarse minimizer.
    """
    if prefer_refined:
        candidate_names = ["refined_minimizer", "minimizer", "coarse_minimizer"]
    else:
        candidate_names = ["coarse_minimizer", "minimizer", "refined_minimizer"]

    for attr in candidate_names:
        if hasattr(response, attr):
            candidate = getattr(response, attr)
            if candidate is not None:
                return ensure_1d_decision_vector(
                    candidate,
                    expected_len=expected_len,
                    name=f"response.{attr}",
                )

    available = [attr for attr in candidate_names if hasattr(response, attr)]
    raise ValueError(
        "Could not extract a solution vector from the QHD response. "
        f"Tried fields {candidate_names}, available among them: {available}."
    )


__all__ = [
    "PrintQHDACOPFResults",
    "SympyACOPFModel",
    "append_qhd_acopf_results",
    "ensure_1d_decision_vector",
    "extract_qhd_solution_vector",
    "format_qhd_acopf_results",
    "initialize_qhd_acopf_log",
    "solve_with_gurobi_from_sympy",
]
