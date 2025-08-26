#!/usr/bin/env python3
"""
Plot and summarize PSTA/ESTA experiment results.

Inputs (defaults):
  - PSTA JSONs: results_gpu/d5/*.json
  - ESTA JSONs: results_gpu/d6/*.json

Outputs (defaults):
  - Figure (PNG/PDF): paper/figures/d5_d6_results.{png,pdf}
  - LaTeX table:      paper/tables/d5_d6_summary.tex
  - CSV summaries:    paper/figures/d5_summary.csv, paper/figures/d6_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
except Exception:
	plt = None  # type: ignore

try:
	import numpy as np
except Exception:
	np = None  # type: ignore

import statistics as stats


def extract_model_from_name(name: str) -> str:
	m = re.search(r"paperA_([a-zA-Z0-9_]+)_hard", name)
	return m.group(1) if m else "unknown"


def parse_metrics(text: str) -> Tuple[float | None, float | None]:
	try:
		data = json.loads(text)
	except Exception:
		data = {}
	mf = data.get("macro_f1") if isinstance(data, dict) else None
	br = data.get("brier") if isinstance(data, dict) else None
	if mf is None:
		m = re.search(r"macro_f1\"?\s*[:=]\s*([0-9]\.[0-9]+)", text)
		mf = float(m.group(1)) if m else None
	if br is None:
		m = re.search(r"brier\"?\s*[:=]\s*([0-9]\.[0-9]+)", text)
		br = float(m.group(1)) if m else None
	return mf, br


def summarize_dir(dir_path: Path) -> Dict[str, Dict[str, List[float]]]:
	summary: Dict[str, Dict[str, List[float]]] = {}
	for fp in sorted(dir_path.glob("*.json")):
		text = fp.read_text(encoding="utf-8", errors="ignore")
		model = extract_model_from_name(fp.name)
		mf, br = parse_metrics(text)
		if mf is None or br is None:
			continue
		summary.setdefault(model, {"macro_f1": [], "brier": []})
		summary[model]["macro_f1"].append(mf)
		summary[model]["brier"].append(br)
	return summary


def mean_std(values: List[float]) -> Tuple[float, float]:
	if not values:
		return float("nan"), float("nan")
	if np is not None:
		arr = np.array(values, dtype=float)
		return float(arr.mean()), float(arr.std(ddof=0))
	return float(stats.mean(values)), float(stats.pstdev(values) if len(values) > 1 else 0.0)


def write_csv(summary: Dict[str, Dict[str, List[float]]], out_csv: Path, label: str) -> None:
	lines = ["protocol,model,n,macro_f1_mean,macro_f1_std,brier_mean,brier_std"]
	for model in sorted(summary):
		mf_mean, mf_std = mean_std(summary[model]["macro_f1"])
		br_mean, br_std = mean_std(summary[model]["brier"])
		lines.append(f"{label},{model},{len(summary[model]['macro_f1'])},{mf_mean:.6f},{mf_std:.6f},{br_mean:.6f},{br_std:.6f}")
	out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_composite(d5: Dict[str, Dict[str, List[float]]], d6: Dict[str, Dict[str, List[float]]], out_path: Path) -> None:
	if plt is None:
		print('[warn] matplotlib not available; skipping composite figure')
		return
	models = sorted(set(d5.keys()) | set(d6.keys()))

	def stat(summary: Dict[str, Dict[str, List[float]]], model: str, key: str) -> Tuple[float, float]:
		return mean_std(summary.get(model, {}).get(key, []))

	d5_f1 = [stat(d5, m, 'macro_f1')[0] for m in models]
	d5_f1_std = [stat(d5, m, 'macro_f1')[1] for m in models]
	d6_f1 = [stat(d6, m, 'macro_f1')[0] for m in models]
	d6_f1_std = [stat(d6, m, 'macro_f1')[1] for m in models]

	d5_br = [stat(d5, m, 'brier')[0] for m in models]
	d5_br_std = [stat(d5, m, 'brier')[1] for m in models]
	d6_br = [stat(d6, m, 'brier')[0] for m in models]
	d6_br_std = [stat(d6, m, 'brier')[1] for m in models]

	fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={'width_ratios': [1.6, 1.4]})

	# Panel A: Grouped bars Macro F1 with legend outside
	x = (np.arange(len(models)) if np is not None else list(range(len(models))))
	width = 0.35
	axes[0].bar([xi - width/2 for xi in x], d5_f1, width, yerr=d5_f1_std, label='PSTA', color='#4C78A8', capsize=3)
	axes[0].bar([xi + width/2 for xi in x], d6_f1, width, yerr=d6_f1_std, label='ESTA', color='#F58518', capsize=3)
	axes[0].set_xticks(x)
	axes[0].set_xticklabels([m.replace('_','-') for m in models], rotation=20, ha='right')
	axes[0].set_ylabel('Macro F1')
	axes[0].set_ylim(0.6, 1.01)
	axes[0].set_title('A) Macro F1 (mean±std)')
	axes[0].grid(axis='y', linestyle=':', alpha=0.3)
	leg0 = axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

	# Panel B: Brier with legend outside
	axes[1].bar([xi - width/2 for xi in x], d5_br, width, yerr=d5_br_std, label='PSTA', color='#9ecae1', capsize=3)
	axes[1].bar([xi + width/2 for xi in x], d6_br, width, yerr=d6_br_std, label='ESTA', color='#fdd0a2', capsize=3)
	axes[1].set_xticks(x)
	axes[1].set_xticklabels([m.replace('_','-') for m in models], rotation=20, ha='right')
	axes[1].set_ylabel('Brier')
	axes[1].set_title('B) Brier (mean±std)')
	axes[1].grid(axis='y', linestyle=':', alpha=0.3)
	leg1 = axes[1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

	fig.tight_layout(w_pad=1.0)
	fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
	fig.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--d5-dir", type=Path, default=Path("results_gpu/d5"))
	parser.add_argument("--d6-dir", type=Path, default=Path("results_gpu/d6"))
	parser.add_argument("--out-fig-prefix", type=Path, default=Path("paper/figures/d5_d6_results"))
	parser.add_argument("--out-tex", type=Path, default=Path("paper/tables/d5_d6_summary.tex"))
	parser.add_argument("--out-d5-csv", type=Path, default=Path("paper/figures/d5_summary.csv"))
	parser.add_argument("--out-d6-csv", type=Path, default=Path("paper/figures/d6_summary.csv"))
	args = parser.parse_args()

	d5 = summarize_dir(args.d5_dir)
	d6 = summarize_dir(args.d6_dir)

	# CSV summaries
	args.out_d5_csv.parent.mkdir(parents=True, exist_ok=True)
	write_csv(d5, args.out_d5_csv, "PSTA")
	write_csv(d6, args.out_d6_csv, "ESTA")

	# Composite figure (two panels)
	comp = args.out_fig_prefix.with_name('d5_d6_composite')
	render_composite(d5, d6, comp)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(130)

