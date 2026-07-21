# TNC vs QHD-SB 3-bus comparison over plot_iteration 0..230

## Scope

- Source data: `3bus_tnc_vs_qhdsb_convergence.csv`.
- Window: `plot_iteration = 0..230`. The first row is the shared QHD-SB initial center for both methods.
- TNC is the rerun from that same initial center. QHD-SB is the coarse-only QCE beam trajectory.
- Objective gaps use the SQP reference objective `0.531710655973` from `3bus_objective_comparison.csv`.
- Objective values from infeasible points are reported, but should not be interpreted as final OPF objective quality without the residual columns.

## Start metrics

| Method | Objective | Obj gap vs SQP % | L2 h | Max abs h | Load % | Same start |
|---|---|---|---|---|---|---|
| Truncated Newton | 0.562978 | 5.880542 | 0.462225 | 0.300000 | 100.000000 | True |
| QHD-SB coarse-only | 0.562978 | 5.880542 | 0.462225 | 0.300000 | 100.000000 | True |

## Objective comparison

| Selection | TNC plot iter | TNC obj | TNC obj gap % | TNC L2 h | TNC max abs h | QHD-SB plot iter | QHD-SB obj | QHD-SB obj gap % | QHD-SB L2 h | QHD-SB max abs h | Lower obj | Lower L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| initial_common_center | 0 | 0.562978 | 5.880542 | 0.462225 | 0.300000 | 0 | 0.562978 | 5.880542 | 0.462225 | 0.300000 | tie | tie |
| first_l2_le_1e-3 | 197 | 0.546754 | 2.829294 | 6.505748e-04 | 2.386962e-04 | 220 | 0.535003 | 0.619153 | 9.823599e-04 | 6.354092e-04 | QHD-SB coarse-only | Truncated Newton |
| best_l2_residual | 228 | 0.528880 | -0.532305 | 5.315941e-04 | 1.789716e-04 | 229 | 0.535003 | 0.619153 | 6.794968e-04 | 3.258883e-04 | Truncated Newton | Truncated Newton |
| final_plot_iteration_230 | 230 | 0.528879 | -0.532614 | 5.318992e-04 | 1.792424e-04 | 230 | 0.535496 | 0.711871 | 8.684086e-04 | 5.570550e-04 | Truncated Newton | Truncated Newton |
| min_objective_with_l2_le_1e-3 | 207 | 0.528019 | -0.694356 | 9.847719e-04 | 6.723381e-04 | 220 | 0.535003 | 0.619153 | 9.823599e-04 | 6.354092e-04 | Truncated Newton | QHD-SB coarse-only |
| min_objective_overall_unfiltered | 6 | 0.218309 | -58.942231 | 0.063840 | 0.022915 | 1 | 0 | -100.000000 | 0.843194 | 0.329299 | QHD-SB coarse-only | Truncated Newton |

## Iteration-level comparison highlights

- Same-iteration residual winner counts: TNC lower L2 in 78 rows, QHD-SB lower L2 in 152 rows, tie in 1 row.
- TNC lower-L2 intervals: 1-43, 196-230.
- QHD-SB lower-L2 interval: 44-195.
- TNC first reaches L2 <= 1e-3 at plot iteration 197; QHD-SB first reaches it at plot iteration 220.
- At plot iteration 230, TNC has L2 h = 5.318992e-04 and objective = 0.528879; QHD-SB has L2 h = 8.684086e-04 and objective = 0.535496.
- Neither method reaches the strict max-abs residual tolerance 1e-5 within this 0..230 window, so objective comparisons should be read together with residual violation levels.

## Key iteration metrics

| Plot iter | TNC obj | TNC L2 h | QHD-SB obj | QHD-SB L2 h | TNC/QHD L2 | Lower L2 |
|---|---|---|---|---|---|---|
| 0 | 0.562978 | 0.462225 | 0.562978 | 0.462225 | 1.000000 | tie |
| 1 | 0.532226 | 0.142532 | 0 | 0.843194 | 0.169038 | Truncated Newton |
| 5 | 0.229104 | 0.064574 | 0 | 1.293472 | 0.049923 | Truncated Newton |
| 10 | 0.254545 | 0.055401 | 0 | 0.597513 | 0.092720 | Truncated Newton |
| 25 | 0.250537 | 0.052952 | 0.622285 | 0.148915 | 0.355588 | Truncated Newton |
| 50 | 0.250529 | 0.052953 | 0.515786 | 0.041711 | 1.269535 | QHD-SB coarse-only |
| 100 | 0.250506 | 0.052958 | 0.549688 | 0.006007 | 8.816024 | QHD-SB coarse-only |
| 150 | 0.250539 | 0.052951 | 0.538595 | 0.002601 | 20.355509 | QHD-SB coarse-only |
| 195 | 0.545141 | 0.003756 | 0.537890 | 0.001649 | 2.277013 | QHD-SB coarse-only |
| 197 | 0.546754 | 6.505748e-04 | 0.537820 | 0.001689 | 0.385278 | Truncated Newton |
| 200 | 0.546666 | 6.032893e-04 | 0.537468 | 0.001617 | 0.373136 | Truncated Newton |
| 220 | 0.528817 | 5.434876e-04 | 0.535003 | 9.823599e-04 | 0.553247 | Truncated Newton |
| 228 | 0.528880 | 5.315941e-04 | 0.535355 | 8.735795e-04 | 0.608524 | Truncated Newton |
| 229 | 0.528877 | 5.321429e-04 | 0.535003 | 6.794968e-04 | 0.783143 | Truncated Newton |
| 230 | 0.528879 | 5.318992e-04 | 0.535496 | 8.684086e-04 | 0.612499 | Truncated Newton |

## Output tables

- `3bus_tnc_qhdsb_230_start_metrics.csv`: two-row start comparison confirming the common initial point.
- `3bus_tnc_qhdsb_230_objective_comparison.csv`: objective-focused selections, with residuals attached for feasibility context.
- `3bus_tnc_qhdsb_230_iteration_metrics.csv`: complete side-by-side data for plot iterations 0..230.
- `3bus_tnc_qhdsb_230_key_iteration_metrics.csv`: compact checkpoint table for manuscript or slides.
