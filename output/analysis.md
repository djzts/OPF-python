# QHD-LALM-SB Log Analysis for 2/3/5/9-Bus Cases

This report combines the requested log folders and excludes 14-bus results. All detailed source data for the plots is saved as CSV in this same `output` folder.

## Data Scope
| Experiment | Bus | Logs | Latest selected log | Main records |
| --- | --- | --- | --- | --- |
| before-multi-beam | 2 | 5 | Buses-2_07-01-2026_22-09-32.txt | 97 |
| before-multi-beam | 3 | 10 | Buses-3_07-02-2026_15-33-59.txt | 341 |
| before-multi-beam | 5 | 2 | Buses-5_07-01-2026_22-11-17.txt | 123 |
| before-multi-beam | 9 | 3 | Buses-9_07-01-2026_22-11-18.txt | 60 |
| coarse only | 2 | 1 | Buses-2_07-06-2026_11-02-21.txt | 97 |
| coarse only | 3 | 1 | Buses-3_07-06-2026_11-02-23.txt | 341 |
| coarse only | 5 | 1 | Buses-5_07-06-2026_11-02-26.txt | 123 |
| coarse only | 9 | 1 | Buses-9_07-06-2026_11-02-29.txt | 60 |
| multi_beam | 2 | 1 | Buses-2_07-03-2026_17-29-20.txt | 106 |
| multi_beam | 3 | 1 | Buses-3_07-03-2026_17-31-49.txt | 232 |
| multi_beam | 5 | 1 | Buses-5_07-03-2026_17-29-25.txt | 123 |
| multi_beam | 9 | 1 | Buses-9_07-03-2026_17-29-29.txt | 60 |
| single_beam | 2 | 1 | Buses-2_07-07-2026_21-37-55.txt | 102 |
| single_beam | 3 | 1 | Buses-3_07-07-2026_21-37-55.txt | 326 |
| single_beam | 5 | 1 | Buses-5_07-07-2026_21-37-59.txt | 359 |
| single_beam | 9 | 1 | Buses-9_07-07-2026_21-38-05.txt | 182 |

## Reference Objectives
| Bus | Reference objective | Source |
| --- | --- | --- |
| 2-bus | 0.610 | SLSQP reference from plot_qhd_convergence_diagnostics.py case model |
| 3-bus | 0.532 | logs/QCE_result/Buses-3_06-22-2026_05-08-43_vs_44_QCE_analysis.md |
| 5-bus | 9.195 | 5bus-answer.txt / SLSQP cross-check |
| 9-bus | 4.098 | 9bus-answer.txt |

## Latest-Log Post-Refine/Final Comparison
This table uses the per-round result after the coarse QHD/SB solve plus refinement when refinement is present. If no refinement vector is present for a round, the final coarse vector is used.

| Bus | Experiment | N | Best iter | Best L2 residual | Max |h| at best L2 | Obj gap % | Final L2 | First <=1e-4 | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-bus | before-multi-beam | 10 | 96 | 9.882e-06 | 5.398e-06 | -0.005 | 9.882e-06 | 72 | Buses-2_07-01-2026_22-09-32.txt |
| 2-bus | coarse only | 10 | 96 | 9.882e-06 | 5.398e-06 | -0.005 | 9.882e-06 | 72 | Buses-2_07-06-2026_11-02-21.txt |
| 2-bus | multi_beam | 10 | 105 | 9.224e-06 | 5.202e-06 | -0.001 | 9.224e-06 | 74 | Buses-2_07-03-2026_17-29-20.txt |
| 2-bus | single_beam | 1 | 101 | 9.366e-06 | 4.956e-06 | -0.006 | 9.366e-06 | 72 | Buses-2_07-07-2026_21-37-55.txt |
| 3-bus | before-multi-beam | 10 | 307 | 1.347e-05 | 4.594e-06 | -0.013 | 1.752e-05 | 127 | Buses-3_07-02-2026_15-33-59.txt |
| 3-bus | coarse only | 10 | 307 | 1.347e-05 | 4.594e-06 | -0.013 | 1.800e-05 | 127 | Buses-3_07-06-2026_11-02-23.txt |
| 3-bus | multi_beam | 10 | 231 | 2.385e-06 | 8.217e-07 | 0.001 | 2.385e-06 | 119 | Buses-3_07-03-2026_17-31-49.txt |
| 3-bus | single_beam | 1 | 225 | 5.539e-06 | 2.872e-06 | -0.005 | 2.116e-05 | 118 | Buses-3_07-07-2026_21-37-55.txt |
| 5-bus | before-multi-beam | 10 | 117 | 3.235e-05 | 1.307e-05 | -6.684e-04 | 6.266e-05 | 89 | Buses-5_07-01-2026_22-11-17.txt |
| 5-bus | coarse only | 10 | 117 | 3.235e-05 | 1.307e-05 | -6.684e-04 | 7.782e-05 | 89 | Buses-5_07-06-2026_11-02-26.txt |
| 5-bus | multi_beam | 10 | 122 | 1.496e-04 | 5.624e-05 | -0.012 | 1.496e-04 |  | Buses-5_07-03-2026_17-29-25.txt |
| 5-bus | single_beam | 1 | 258 | 1.531e-05 | 6.831e-06 | -8.153e-04 | 3.272e-05 | 116 | Buses-5_07-07-2026_21-37-59.txt |
| 9-bus | before-multi-beam | 10 | 59 | 4.229e-04 | 1.681e-04 | 0.025 | 4.229e-04 |  | Buses-9_07-01-2026_22-11-18.txt |
| 9-bus | coarse only | 10 | 59 | 4.229e-04 | 1.681e-04 | 0.025 | 4.229e-04 |  | Buses-9_07-06-2026_11-02-29.txt |
| 9-bus | multi_beam | 10 | 59 | 4.373e-04 | 1.675e-04 | 0.022 | 4.373e-04 |  | Buses-9_07-03-2026_17-29-29.txt |
| 9-bus | single_beam | 1 | 81 | 1.401e-04 | 5.798e-05 | 0.059 | 2.934e-04 |  | Buses-9_07-07-2026_21-38-05.txt |

## Coarse vs Post-Refine Paired Comparison
| Bus | Experiment | N | Coarse best iter | Coarse best L2 | Post best iter | Post/refined best L2 | Coarse/Post ratio | Residual reduction % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-bus | before-multi-beam | 10 | 93 | 3.218e-04 | 96 | 9.882e-06 | 32.560 | 96.929 |
| 2-bus | coarse only | 10 | 93 | 3.218e-04 | 96 | 9.882e-06 | 32.560 | 96.929 |
| 2-bus | multi_beam | 10 | 87 | 3.175e-04 | 105 | 9.224e-06 | 34.425 | 97.095 |
| 2-bus | single_beam | 1 | 101 | 3.181e-04 | 101 | 9.366e-06 | 33.965 | 97.056 |
| 3-bus | before-multi-beam | 10 | 210 | 4.588e-04 | 307 | 1.347e-05 | 34.054 | 97.064 |
| 3-bus | coarse only | 10 | 210 | 4.588e-04 | 307 | 1.347e-05 | 34.054 | 97.064 |
| 3-bus | multi_beam | 10 | 228 | 4.758e-04 | 231 | 2.385e-06 | 199.521 | 99.499 |
| 3-bus | single_beam | 1 | 210 | 4.597e-04 | 225 | 5.539e-06 | 82.995 | 98.795 |
| 5-bus | before-multi-beam | 10 | 105 | 0.007 | 117 | 3.235e-05 | 231.736 | 99.568 |
| 5-bus | coarse only | 10 | 105 | 0.007 | 117 | 3.235e-05 | 231.736 | 99.568 |
| 5-bus | multi_beam | 10 | 103 | 0.007 | 122 | 1.496e-04 | 49.498 | 97.980 |
| 5-bus | single_beam | 1 | 188 | 0.007 | 258 | 1.531e-05 | 481.057 | 99.792 |
| 9-bus | before-multi-beam | 10 | 56 | 0.002 | 59 | 4.229e-04 | 5.467 | 81.710 |
| 9-bus | coarse only | 10 | 56 | 0.002 | 59 | 4.229e-04 | 5.467 | 81.710 |
| 9-bus | multi_beam | 10 | 52 | 0.002 | 59 | 4.373e-04 | 5.198 | 80.764 |
| 9-bus | single_beam | 1 | 108 | 0.001 | 81 | 1.401e-04 | 10.422 | 90.405 |

## Key Observations
- 2-bus: best latest-log residual is 9.224e-06 from `multi_beam` at iteration 105 (objective gap -0.001%).
- 3-bus: best latest-log residual is 2.385e-06 from `multi_beam` at iteration 231 (objective gap 0.001%).
- 5-bus: best latest-log residual is 1.531e-05 from `single_beam` at iteration 258 (objective gap -8.153e-04%).
- 9-bus: best latest-log residual is 1.401e-04 from `single_beam` at iteration 81 (objective gap 0.059%).

- 2-bus: `multi_beam` has the lower best residual; the gap is about 1.02x (9.224e-06 for multi_beam vs 9.366e-06 for single_beam).
- 3-bus: `multi_beam` has the lower best residual; the gap is about 2.32x (2.385e-06 for multi_beam vs 5.539e-06 for single_beam).
- 5-bus: `single_beam` has the lower best residual; the gap is about 9.77x (1.496e-04 for multi_beam vs 1.531e-05 for single_beam).
- 9-bus: `single_beam` has the lower best residual; the gap is about 3.12x (4.373e-04 for multi_beam vs 1.401e-04 for single_beam).

- The post-refine/final stage is now compared explicitly against the coarse-pre-refine stage for each round.
- The best-residual iteration is often better than the final iteration; this is especially visible where later iterations drift after reaching a low residual.
- The residual columns use recomputed ACOPF residuals when the bus/branch tables are complete; otherwise they fall back to the values printed in the log.

## Generated Files
- `iteration_metrics.csv`: all parsed per-iteration records.
- `round_stage_metrics.csv`: paired coarse and post-refine/final source data parsed from `coarse_x` and final `x` vectors.
- `round_stage_summary.csv`: one row per log/stage.
- `selected_latest_round_stage_summary.csv`: latest valid log for each experiment/bus/stage.
- `paired_coarse_post_latest.csv`: coarse-vs-post-refine comparison for the latest selected logs.
- `plot_source_selected_convergence.csv`: post-refine/final per-iteration source data used by the convergence plots.
- `all_runs_summary.csv`: one row per parsed log.
- `selected_latest_summary.csv`: latest valid log for each experiment/bus pair.
- `best_across_logs_summary.csv`: best historical log for each experiment/bus pair by L2 residual.
- `best_l2_residual_by_experiment.png`, `objective_gap_pct_by_experiment.png`, `iterations_to_1e4_by_experiment.png`: post-refine/final cross-case comparison plots.
- `coarse_vs_post_refine_improvement.png`: paired residual improvement from refine.
- `convergence_l2_*bus.png` and `convergence_objective_gap_*bus.png`: bus-specific convergence plots.
