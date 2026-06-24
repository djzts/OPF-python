# 3-bus QCE log analysis

## Method

- Parsed the QHD iteration tables using the same bus-state and branch-flow comparison method as `analyze_qhd_vs_standard.py`.
- The repository has no `3bus-answer.txt`, so the reference was generated from the repository's 3-bus ACOPF equations with multi-start SciPy/SLSQP.
- Reference solve: success `True`, objective `0.531710655973`, maximum equality residual `2.479e-13`.
- The trailing `best_iteration_by_objective` block is a summary record and was excluded from the main iteration trajectory.

## `Buses-3_06-22-2026_01-53-26.txt`

The file is 289 bytes and contains only the log header. It has no `Iteration:` records, bus tables, or branch tables, so no numerical QHD-versus-standard comparison is possible.

## `Buses-3_06-22-2026_05-08-43.txt`

- Parsed main-loop iterations: `348` (`0` through `347`).
- No iteration is marked feasible.
- First nonzero objective: iteration `17`.
- First `||h||2 <= 1`: iteration `7`.
- First `||h||2 <= 0.1`: iteration `34`.
- First `||h||2 <= 0.01`: iteration `112`.
- First `||h||2 <= 0.001`: iteration `239`.

| Selection criterion | Iteration | Objective | Objective gap | `||h||2` | max `|h|` | Load supplied | Bus L2 | Branch L2 | Combined L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Closest combined state/flow | 98 | 0.535330139374 | +0.680724% | 1.282751e-2 | 8.161350e-3 | 100.300870% | 0.00246521 | 0.00291720 | 0.00381933 |
| Closest objective | 143 | 0.531621363722 | -0.016793% | 5.291811e-3 | 3.339440e-3 | 99.871222% | 0.02204226 | 0.01942057 | 0.02937720 |
| Smallest `||h||2` | 247 | 0.531185908484 | -0.098690% | 7.248938e-4 | 4.786251e-4 | 99.934110% | 0.01560457 | 0.01386530 | 0.02087461 |
| Final/main iteration | 347 | 0.530833790851 | -0.164914% | 8.471825e-4 | 4.216061e-4 | 99.947443% | 0.01458988 | 0.01266564 | 0.01932054 |

### Main observations

1. The run made real progress: the nonlinear residual fell from `4.12` at iteration 0 to about `8.47e-4` at the end, and load supplied approached 100%.
2. It did not satisfy the configured convergence tolerance. The final residual remains far above `1e-5`, and every main-loop record is marked infeasible.
3. Iteration 247 has the smallest L2 residual, while iteration 248 has the smallest maximum component residual. These are different scalar criteria.
4. Iteration 98 is closest to the standard bus/branch solution, but its constraint residual is still `1.28e-2`. The closest state is not the most feasible state.
5. Iterations 248 through 347 are numerically identical in the logged variables and metrics: a 100-iteration plateau.
6. `rho` reaches its cap of `256` at iteration 245. `alpha` reaches its floor of `0.01` at iteration 148. Further iterations do not escape the discretization/bound plateau.
7. The final `best_iteration_by_objective` block points back to iteration 16 with objective 0 and `||h||2 = 0.241`. It is infeasible and should not be treated as the final or best physical solution.

### Reference dispatch

- `P_G = [0, 0.302917073]`
- `Q_G = [-0.013592136, -0.006387496]`
- `V_R = [1, 1.002711020, 0.987218281]`
- `V_I = [0, 0.007160961, -0.021646815]`

The final main-loop dispatch is close in active generation (`P_G = [0, 0.302419]`) but still has nonzero physical residual and a noticeably different reactive-power split.
