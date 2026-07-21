# 3-bus classical solver comparison against QCE coarse-only

## Technical summary

- The continuous constrained optimizers solve the same 3-bus ACOPF to near machine precision: SQP reaches max |h| = 1.491862e-15, interior-point reaches 3.010439e-14, and the staged truncated-Newton penalty reaches 1.741195e-08.
- The requested QCE run is coarse-only (`refine_method='none'`, `qhd_refine=False`) and stops at max |h| = 3.258883e-04 for its best residual record. That is still about 32.588828x above the configured 1e-5 feasibility tolerance.
- Newton-Raphson active-set KKT succeeds for 5 of 5 tested starts when the correct active generator bound is supplied. This is useful for local KKT solving, but it is less general than SQP or interior-point because the active set is assumed rather than discovered.
- The QCE log ran from 2026-06-22 05:08:44 to 2026-06-22 13:27:01 (2.989700e+04 s), while the local SciPy continuous solves completed in seconds or less on this machine. Runtime is therefore directionally useful but not a hardware-normalized benchmark.

## Method results

| Method | Best start | Success | Objective | Obj gap % | L2 h | Max abs h | Runtime s | Iters | State/flow dist |
|---|---|---|---|---|---|---|---|---|---|
| SQP (SLSQP) | qce_final_iter | True | 0.531711 | 0 | 2.414189e-15 | 1.491862e-15 | 0.034619 | 11 | 0 |
| Interior-point (trust-constr) | qce_final_iter | True | 0.531711 | 1.925857e-06 | 5.722520e-14 | 3.010439e-14 | 0.854616 | 125 | 5.933041e-08 |
| Truncated Newton (TNC penalty) | qce_best_l2_iter | True | 0.531723 | 0.002382 | 5.227591e-08 | 1.741195e-08 | 16.039825 | 569 | 0.026573 |
| Newton-Raphson active-set KKT | flat_start | True | 0.531711 | 3.739782e-08 | 1.048274e-10 | 1.000000e-10 | 0.506410 | 4 | 1.422083e-09 |
| QCE coarse-only | best_max_iter_228 | False | 0.535003 | 0.619153 | 6.794968e-04 | 3.258883e-04 | 2.989700e+04 | 329 | 0.501449 |

## Objective comparison

| Selection | Objective | Obj gap % | L2 h | Max abs h | Feasible/near-feasible |
|---|---|---|---|---|---|
| SQP (SLSQP) | 0.531711 | 0 | 2.414189e-15 | 1.491862e-15 | True |
| Interior-point (trust-constr) | 0.531711 | 1.925857e-06 | 5.722520e-14 | 3.010439e-14 | True |
| Truncated Newton (TNC penalty) | 0.531723 | 0.002382 | 5.227591e-08 | 1.741195e-08 | True |
| Newton-Raphson active-set KKT | 0.531711 | 3.739782e-08 | 1.048274e-10 | 1.000000e-10 | True |
| QCE best residual iter 228 | 0.535003 | 0.619153 | 6.794968e-04 | 3.258883e-04 | False |
| QCE final iter 328 | 0.535496 | 0.711871 | 8.684086e-04 | 5.570550e-04 | False |

Using the SQP solution as the reference objective (0.531710655973), the continuous constrained methods are essentially objective-equivalent. TNC is slightly higher at +0.002382%, while the QCE best-residual record is +0.619153% and the QCE final record is higher still. The QCE `best_iteration_by_objective` summary in the raw log has objective 0, but it is infeasible and is therefore excluded from this objective-quality comparison.

## QCE baseline behavior

| QCE selection | Iteration | Objective | L2 h | Max abs h | Load % |
|---|---|---|---|---|---|
| Best L2 residual | 228 | 0.535003 | 6.794968e-04 | 3.258883e-04 | 99.975523 |
| Best max residual | 228 | 0.535003 | 6.794968e-04 | 3.258883e-04 | 99.975523 |
| Final main-loop record | 328 | 0.535496 | 8.684086e-04 | 5.570550e-04 | 99.908857 |

The 05-08-44 QCE trajectory has 329 main-loop records. It first reaches L2 h <= 1e-2 at iteration 73 and L2 h <= 1e-3 at iteration 219. The final 99 records have zero logged step norm, which matches the plateau behavior described in the existing two-log analysis file.

The two-log analysis establishes the reference ACOPF objective as 0.531710655973 and notes that the earlier 05-08-43 QCE run also stayed infeasible, with best L2 h around 7.25e-4 and a 100-iteration plateau. The 05-08-44 run improves the minimum residual slightly, but the parsed final point has a large reactive-power split: Q_G is approximately [-0.409, 0.395] rather than the reference dispatch near [-0.0136, -0.00639]. This explains why a small equality residual can still correspond to a noticeably different reactive operating branch.

## Sensitivity to initial conditions

| Method | Start | Success | Obj gap % | Max abs h | Runtime s | Iters |
|---|---|---|---|---|---|---|
| SQP (SLSQP) | flat_start | True | -2.375877e-12 | 3.436418e-13 | 0.033056 | 10 |
| SQP (SLSQP) | qce_best_l2_iter | True | -6.661338e-14 | 1.568190e-15 | 0.034063 | 11 |
| SQP (SLSQP) | qce_final_iter | True | 0 | 1.491862e-15 | 0.034619 | 11 |
| SQP (SLSQP) | random_perturbed_flat | False | -17.621553 | 1.748846 | 0.054161 | 21 |
| Interior-point (trust-constr) | flat_start | True | 9.629244e-06 | 7.032534e-13 | 0.194413 | 33 |
| Interior-point (trust-constr) | qce_best_l2_iter | True | 9.629241e-06 | 7.663592e-13 | 0.791165 | 115 |
| Interior-point (trust-constr) | qce_final_iter | True | 1.925857e-06 | 3.010439e-14 | 0.854616 | 125 |
| Interior-point (trust-constr) | random_perturbed_flat | True | 1.925859e-06 | 5.517808e-14 | 0.322169 | 48 |
| Truncated Newton (TNC penalty) | flat_start | True | -5.512426e-05 | 3.091338e-08 | 13.828697 | 504 |
| Truncated Newton (TNC penalty) | qce_best_l2_iter | True | 0.002382 | 1.741195e-08 | 16.039825 | 569 |
| Truncated Newton (TNC penalty) | qce_final_iter | True | 0.011431 | 1.971733e-08 | 10.769914 | 402 |
| Truncated Newton (TNC penalty) | random_perturbed_flat | True | 0.001374 | 1.786953e-08 | 15.649399 | 575 |
| Newton-Raphson active-set KKT | flat_start | True | 3.739782e-08 | 1.000000e-10 | 0.506410 | 4 |
| Newton-Raphson active-set KKT | qce_best_l2_iter | True | 3.761418e-08 | 1.000000e-10 | 2.248677 | 14 |
| Newton-Raphson active-set KKT | qce_final_iter | True | 3.761362e-08 | 1.000000e-10 | 2.264950 | 14 |
| Newton-Raphson active-set KKT | random_perturbed_flat | True | 3.761436e-08 | 1.000000e-10 | 0.719586 | 5 |
| Newton-Raphson active-set KKT | sqp_solution_warm_start | True | 3.761444e-08 | 1.000000e-10 | 0.002770 | 1 |

SQP is robust from the flat and QCE starts but can fail from a larger arbitrary perturbation. Interior-point is the most robust across the tested starts. The truncated-Newton penalty method is also fairly robust, but it trades constraint accuracy for a longer staged penalty solve and depends on large penalty weights. The active-set Newton-Raphson implementation converges on these starts after the correct active bound is supplied, but it remains sensitive to that modeling assumption and should not be treated as a generic inequality-constrained solver without active-set detection.

## Convergence interpretation

QCE makes broad global search progress early but becomes quantized by the coarse grid, bound shrinkage, alpha floor, and rho cap. The continuous methods do not have that quantization barrier: SQP and interior-point use local constraint models to drive the equality residual to numerical precision, and the penalty TNC method reduces violation as rho increases. The price is that these continuous methods rely on smooth local derivatives and do not provide the same beam-search exploration that QCE provides.

## Implementation notes from the QCE server

- Source file: `Sympy_OPF_LALM_mu_final_3bus_QCE_server.py`.
- The server is configured as coarse-only QCE: `refine_method='none'` and `qhd_refine=False`.
- The beam width is `n_linearization_points=10`, with adaptive `alpha` and `rho` capped at `rho_max=256`.
- `simbi_agents` and `simbi_max_steps` are annotated as integers but assigned with `/4`, producing floats. Python accepts this, but casting to `int` would make the solver configuration less ambiguous.

## Recommended next steps

1. Use SQP or interior-point as the deterministic continuous reference for 3-bus result validation.
2. Use QCE best-residual points as candidate warm starts, then add a continuous local refinement stage if the goal is feasibility below 1e-5.
3. Treat Newton-Raphson KKT as an active-set method: it works here after assuming generator 1 is at Pmin, but a general implementation should detect or update the active set.
4. For future QCE runs, log both coarse and post-refine metrics when refinement is enabled, so the comparison is made after each full coarse+refine round.

## Output files

- `3bus_method_start_metrics.csv`: method-by-start metrics.
- `3bus_method_best_summary.csv`: best result by method plus QCE baseline.
- `3bus_objective_comparison.csv`: objective and objective-gap comparison source data.
- `3bus_qce_iteration_metrics.csv`: parsed QCE iteration metrics from the 05-08-44 log.
- `3bus_solver_convergence_history.csv`: source data for convergence plots.
- `3bus_objective_gap_comparison.png`, `3bus_best_residual_comparison.png`, `3bus_runtime_comparison.png`, `3bus_convergence_residual.png`, `3bus_initial_sensitivity.png`: generated figures.

## Sources

- `logs\QCE_result\Buses-3_06-22-2026_05-08-44.txt`
- `logs\QCE_result\Buses-3_06-22-2026_two_log_analysis.md`
- `Sympy_OPF_LALM_mu_final_3bus_QCE_server.py`
- `plot_qhd_convergence_diagnostics.py` for the shared 3-bus case equations and metric helpers.
