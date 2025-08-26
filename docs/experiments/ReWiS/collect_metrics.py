#!/usr/bin/env python3
import json, re, sys, pathlib
p = pathlib.Path(results)
metrics = {}
for f in p.rglob(*.log):
    text = f.read_text(errors=ignore)
    m = re.findall(r"F1[:=]\s*([0-9]+\.?[0-9]*)", text)
    if m:
        metrics[str(f)] = {"macro_f1": float(m[-1])}
(pathlib.Path(results)/ metrics.json).write_text(json.dumps(metrics, indent=2))
print("Wrote results/metrics.json with", len(metrics), "entries")
