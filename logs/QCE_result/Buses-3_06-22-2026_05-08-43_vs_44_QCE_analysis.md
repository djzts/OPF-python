# QCE coarse-only comparison: 05-08-43 vs 05-08-44

Reference ACOPF objective: `0.531710655973`.

| Metric | 05-08-43 | 05-08-44 |
|---|---:|---:|
| Main-loop records | 348 | 329 |
| Best `||h||2` | `7.277876e-4` at iter 247 | `6.793279e-4` at iter 228 |
| Best max `|h|` | `4.214531e-4` at iter 248 | `3.235054e-4` at iter 228 |
| First `||h||2 <= 0.01` | iter 112 | iter 73 |
| First `||h||2 <= 0.001` | iter 239 | iter 219 |
| Final objective gap | `-0.164921%` | `+0.711900%` |
| Final load supplied | `99.947667%` | `99.909000%` |
| Final combined reference distance | `0.0126797` | `0.498191` |
| Final plateau | iter 248–347 (100 records) | iter 229–328 (100 records) |
| First iteration with `rho=256` | 245 | 226 |

## Interpretation

- `05-08-44` reduces the nonlinear residual faster and reaches a slightly smaller minimum residual.
- `05-08-43` is much closer to the reference ACOPF state and has a smaller final objective gap.
- The large reference distance in `05-08-44` is dominated by reactive-power circulation, not active generation:
  - Reference `Q_G = [-0.013592, -0.006388]`.
  - 05-08-43 final `Q_G = [-0.003240, -0.016639]`.
  - 05-08-44 final `Q_G = [-0.409251, 0.394609]`.
- The 05-08-44 final branch-Q distance is about `0.497643`, while its branch-P distance is only about `0.002937`.
- Because the OPF objective penalizes active generation but not reactive circulation directly, a low-residual coarse solution can remain far from the selected standard solution in Q variables.
- Both runs end in a 100-iteration quantization/bound plateau. Additional iterations after the plateau add no value without changing resolution, bounds, or local refinement.

## Recommended selection

- Use iteration 247 from `05-08-43` when prioritizing the best balance of objective, residual, and proximity to the reference solution.
- Use iteration 228 from `05-08-44` only when the smallest coarse-only equality residual is the primary criterion; it belongs to a different high-Q operating branch.
