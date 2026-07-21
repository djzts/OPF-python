# Response to Reviewer Comment 3: Benchmarking Against Classical Solvers

## Reviewer Comment

**3. Benchmarking against Classical Solvers:** Consider adding a comparison against established industrial-standard solvers in terms of metrics such as solution quality, constraint violation levels, computation time, convergence behavior, and sensitivity to initial conditions.

## Response

We thank the reviewer for this important and constructive suggestion. We agree that benchmarking against established classical nonlinear optimization solvers is necessary to properly contextualize the proposed QHD-LALM-SB/QCE approach.

To address this comment, we added a new benchmarking study for the 3-bus ACOPF case. We compared the proposed QCE/QHD coarse-only approach against four classical continuous optimization methods:

- Newton-Raphson active-set KKT method
- Truncated Newton method
- Interior-point method
- Sequential quadratic programming (SQP)

The comparison evaluates the requested metrics, including solution quality, objective gap, equality constraint violation, computation time, convergence behavior, and sensitivity to initial conditions.

## Added Benchmark Summary

| Method | Objective | Objective Gap | L2 Constraint Residual | Max Constraint Violation | Runtime | Notes |
|---|---:|---:|---:|---:|---:|---|
| SQP | 0.531710655973 | 0.000000% | 2.41e-15 | 1.49e-15 | 0.035 s | Best classical reference solution |
| Interior-point | 0.531710666213 | 1.93e-06% | 5.72e-14 | 3.01e-14 | 0.85 s | Near machine-precision feasibility |
| Newton-Raphson active-set KKT | 0.531710656171 | 3.74e-08% | 1.05e-10 | 1.00e-10 | 0.51 s | Requires correct active-set assumption |
| Truncated Newton | 0.531723322219 | 0.002382% | 5.23e-08 | 1.74e-08 | 16.0 s | Uses staged equality-penalty formulation |
| QCE/QHD coarse-only, best residual | 0.535002759703 | 0.619153% | 6.79e-04 | 3.26e-04 | 29897 s | Best residual at iteration 228 |
| QCE/QHD coarse-only, final | 0.535495747987 | 0.711871% | 8.68e-04 | 5.57e-04 | 29897 s | Final iteration remains infeasible |

## Interpretation Added to the Manuscript

The added benchmark shows that, for this small smooth continuous ACOPF instance, mature classical nonlinear programming solvers such as SQP and interior-point methods converge to near machine-precision feasibility and are substantially faster than the current coarse-only QCE/QHD implementation.

This result is expected and helps clarify the role of the proposed QHD method. We do not position QHD as a direct replacement for SQP or interior-point methods on small smooth continuous OPF problems. Instead, QHD is intended to provide a complementary global search and candidate-generation mechanism, especially for cases where the OPF landscape is highly nonconvex, initialization-sensitive, or contains discrete or hybrid decision variables.

The QCE/QHD run still demonstrates useful global search behavior: it steadily reduces the nonlinear residual and identifies candidate operating points, but the coarse-only implementation reaches a quantization/bound-shrinkage plateau before satisfying the final feasibility tolerance. Therefore, we revised the discussion to emphasize that QHD-generated candidates should be combined with local continuous refinement when high-accuracy feasibility is required.

## Manuscript Revision

We revised the manuscript to include:

- A benchmark table comparing QHD/QCE with SQP, interior-point, truncated Newton, and Newton-Raphson methods.
- Objective-gap comparison between QHD/QCE and the classical reference solution.
- Constraint-violation comparison using both L2 residual and maximum equality residual.
- Runtime and convergence-behavior comparison.
- Sensitivity-to-initial-condition analysis.
- A clarification that infeasible low-objective records should not be interpreted as valid OPF solutions.
- A discussion of QHD as a global candidate-generation and hybrid optimization component rather than a direct replacement for mature local NLP solvers on small smooth OPF cases.

## Revised Claim

Based on the new benchmark, we have revised the claim as follows:

> The proposed QHD-LALM-SB/QCE method is not intended to replace mature local NLP solvers for small smooth continuous ACOPF instances, where SQP and interior-point methods are highly effective. Instead, QHD provides a complementary global search mechanism that can generate candidate solutions for subsequent local refinement and may be particularly valuable for larger, more nonconvex, initialization-sensitive, or discrete/hybrid OPF formulations.

## Final Response Text

We thank the reviewer for this important suggestion. In response, we added a new benchmark comparing the proposed QHD-LALM-SB/QCE approach with four classical nonlinear optimization methods: Newton-Raphson active-set KKT, truncated Newton, interior-point, and SQP. The comparison includes solution quality, objective gap, equality constraint violation, computation time, convergence behavior, and sensitivity to initial conditions.

The benchmark shows that, for the small smooth 3-bus continuous ACOPF case, SQP and interior-point methods achieve near machine-precision feasibility. The SQP reference solution has objective 0.531710655973 with maximum equality residual 1.49e-15, while the interior-point method reaches maximum residual 3.01e-14. The truncated Newton method also reaches a near-feasible solution, with maximum residual 1.74e-08.

The QCE/QHD coarse-only run reaches its best residual at iteration 228, with objective 0.535002759703, objective gap +0.619153%, L2 constraint residual 6.79e-04, and maximum equality residual 3.26e-04. The final QCE/QHD iteration remains infeasible, with objective gap +0.711871% and maximum equality residual 5.57e-04.

These results clarify the intended role of the proposed method. For small smooth continuous OPF instances, mature local NLP solvers remain superior in final feasibility and runtime. The value of QHD lies in its complementary role as a global candidate-generation or hybrid optimization component, especially for nonconvex, initialization-sensitive, or discrete/mixed-variable OPF settings. We revised the manuscript accordingly and added the benchmark table, convergence figures, and discussion of this distinction.
