# Continuous-only SciPy/TNC baseline (2026-09-19)

Run `python output/run_tnc_continuous_baseline.py` from this repository root. The script writes `summary.csv`, `history.csv`, the exact logged starting vectors, and the selected TNC vectors to this folder.

## Design

- Cases: 2, 3, 5, 9, and 14 buses, with the same case data and rectangular residual definitions used by the active SC solver. Parsed JSON case data for 5, 9, and 14 buses is identical between the code and data repositories.
- Start: the `qhd_start iter=0` center from each specified `logs/single_beam` run. The reference-bus real and imaginary voltage bounds are set to `[0.999999, 1.000001]` and `[-0.000001, 0.000001]`, as in the active code.
- Solver: `scipy.optimize.minimize(method="TNC")` on the original nonlinear objective plus a staged quadratic equality penalty, with penalty values 128, 512, 2048, 8192, 32768, 131072, and 524288. Each stage allows up to 1800 function evaluations. Stop once the maximum absolute original residual is at most `1e-5`.
- Selection: smallest `(max_abs_h, l2_h)` among all TNC callbacks and stage endpoints. The archived SB/QHD+TNC comparator is selected from its logged `evaluation` records by the same ordering.
- The analytic residual Jacobian was compared against central differences at the initial point of every case. Maximum absolute entrywise discrepancy was `7.53e-9`.

## Results

| Buses | TNC objective | TNC max residual | SB/QHD+TNC objective | SB/QHD+TNC max residual | Iterations to 1e-4 (TNC / hybrid) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.609608 | 3.901e-6 | 0.609604 | 4.957e-6 | 77 / 67 |
| 3 | 0.531657 | 3.423e-6 | 0.531683 | 2.563e-6 | 131 / 103 |
| 5 | 9.194915 | 3.860e-6 | 9.194992 | 5.854e-6 | 319 / 90 |
| 9 | 4.098038 | 2.682e-6 | 4.100612 | 4.582e-5 | 559 / 66 |
| 14 | 4.874922 | 3.693e-6 | 4.879711 | 3.302e-5 | 463 / 67 |

All TNC selected points meet the study's `1e-5` maximum-residual criterion. At `1e-4`, the archived hybrid has fewer recorded outer rounds than the continuous baseline has TNC optimizer steps on every case. These are **method-specific iteration units**: one hybrid outer round includes an SB/QHD solve and TNC refinement, while one TNC step is a continuous optimizer update. The comparison is descriptive rather than a matched-work ablation, and no multiple-seed statistics are implied by these five deterministic runs. See `iteration_threshold_comparison.csv` for first-crossing counts at four thresholds.
