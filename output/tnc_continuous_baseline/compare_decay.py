import csv
import re
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[2]
folder = Path(__file__).resolve().parent
with (folder / "summary.csv").open(newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))
with (folder / "history.csv").open(newline="", encoding="utf-8") as f:
    history = list(csv.DictReader(f))

thresholds = (1e-2, 1e-3, 1e-4, 1e-5)
print("bus,threshold,tnc_optimizer_steps,hybrid_outer_rounds")
for row in summary:
    bus = row["bus"]
    log = Path(row["sb_log"])
    content = log.read_text(encoding="utf-8", errors="replace")
    start_match = re.search(r"\[step ([^\]]+)\] run_start", content)
    assert start_match, log
    start = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
    sb = []
    for line in content.splitlines():
        if "] evaluation " not in line:
            continue
        match = re.search(r"^\[step ([^\]]+)\] evaluation iter=(\d+).*?max_abs_h=([-+\deE.]+)", line)
        if match:
            stamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            sb.append(((stamp - start).total_seconds(), int(match.group(2)), float(match.group(3))))
    tnc = [(float(h["seconds"]), int(h["iteration"]), float(h["max_abs"])) for h in history if h["bus"] == bus]
    for threshold in thresholds:
        thit = next(((sec, it) for sec, it, residual in tnc if residual <= threshold), None)
        shit = next(((sec, it) for sec, it, residual in sb if residual <= threshold), None)
        print(f"{bus},{threshold:g},{thit[1] if thit else ''},{shit[1] + 1 if shit else ''}")
