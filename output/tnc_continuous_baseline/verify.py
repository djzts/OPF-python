import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plot_qhd_convergence_diagnostics import load_case  # noqa: E402

folder = Path(__file__).resolve().parent
with (folder / "summary.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 5
for row in rows:
    bus = int(row["bus"])
    case = load_case(bus, ROOT)
    x = np.load(folder / f"case{bus}_tnc_best.npy")
    assert x.size == case.n_variables
    assert np.isclose(case.objective(x), float(row["tnc_best_objective"]), atol=1e-11)
    assert np.isclose(np.linalg.norm(case.h(x)), float(row["tnc_best_l2"]), atol=1e-11)
    assert np.isclose(np.max(np.abs(case.h(x))), float(row["tnc_best_max_abs"]), atol=1e-11)
    assert float(row["tnc_best_max_abs"]) <= 1e-5
    print(f"{bus}-bus: objective and residual verified")
