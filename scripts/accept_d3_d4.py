#!/usr/bin/env python3
"""
Acceptance checker for D3/D4 experiments.

D3: Cross-domain generalization (LOSO/LORO) validation
D4: Sim2Real label efficiency validation

Reads result JSON files and verifies:
- D3: Cross-domain performance consistency across models and seeds
- D4: Label efficiency achieving ≥90% performance with ≤20% labels

Produces:
 - results/metrics/d3_acceptance_report.txt
 - results/metrics/d4_acceptance_report.txt  
 - results/metrics/summary_d3.csv
 - results/metrics/summary_d4.csv

Exit code:
 - 0 if all checks pass
 - 2 if any acceptance check fails

Usage:
  python scripts/accept_d3_d4.py --d3_root results/d3 --d4_root results/d4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class D3ResultRow:
    file: str
    protocol: str  # LOSO or LORO
    model: str
    seed: int
    macro_f1: Optional[float]
    ece: Optional[float]
    falling_f1: Optional[float]
    epileptic_fall_f1: Optional[float]
    elderly_fall_f1: Optional[float]


@dataclass  
class D4ResultRow:
    file: str
    model: str
    seed: int
    label_ratio: float
    transfer_method: str
    zero_shot_f1: Optional[float]
    target_f1: Optional[float]
    target_ece: Optional[float]
    improvement: Optional[float]


def safe_get(dct: Dict[str, Any], *keys: str, default=None):
    cur: Any = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_d3_result(filepath: Path) -> Optional[D3ResultRow]:
    """Parse a D3 result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        protocol = safe_get(data, 'protocol', default='')
        model = safe_get(data, 'model', default='')
        seed = safe_get(data, 'seed', default=0)
        
        aggregate = safe_get(data, 'aggregate_stats', default={})
        macro_f1 = safe_get(aggregate, 'macro_f1', 'mean')
        ece = safe_get(aggregate, 'ece', 'mean')
        falling_f1 = safe_get(aggregate, 'falling_f1', 'mean')
        epileptic_fall_f1 = safe_get(aggregate, 'epileptic_fall_f1', 'mean')
        elderly_fall_f1 = safe_get(aggregate, 'elderly_fall_f1', 'mean')
        
        return D3ResultRow(
            file=str(filepath),
            protocol=protocol,
            model=model,
            seed=seed,
            macro_f1=macro_f1,
            ece=ece,
            falling_f1=falling_f1,
            epileptic_fall_f1=epileptic_fall_f1,
            elderly_fall_f1=elderly_fall_f1
        )
    except Exception as e:
        print(f"Failed to parse D3 result {filepath}: {e}", file=sys.stderr)
        return None


def parse_d4_result(filepath: Path) -> Optional[D4ResultRow]:
    """Parse a D4 result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model = safe_get(data, 'model', default='')
        seed = safe_get(data, 'seed', default=0)
        label_ratio = safe_get(data, 'label_ratio', default=0.0)
        transfer_method = safe_get(data, 'transfer_method', default='')
        
        zero_shot_f1 = safe_get(data, 'zero_shot_metrics', 'macro_f1')
        target_f1 = safe_get(data, 'target_metrics', 'macro_f1')
        target_ece = safe_get(data, 'target_metrics', 'ece')
        
        improvement = None
        if zero_shot_f1 is not None and target_f1 is not None:
            improvement = target_f1 - zero_shot_f1
        
        return D4ResultRow(
            file=str(filepath),
            model=model,
            seed=seed,
            label_ratio=label_ratio,
            transfer_method=transfer_method,
            zero_shot_f1=zero_shot_f1,
            target_f1=target_f1,
            target_ece=target_ece,
            improvement=improvement
        )
    except Exception as e:
        print(f"Failed to parse D4 result {filepath}: {e}", file=sys.stderr)
        return None


def validate_d3_acceptance(rows: List[D3ResultRow]) -> Tuple[bool, List[str]]:
    """Validate D3 cross-domain generalization acceptance criteria."""
    issues = []
    
    # Group by protocol and model
    protocol_model_seeds = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row.macro_f1 is not None:
            protocol_model_seeds[row.protocol][row.model].add(row.seed)
    
    # Check minimum seeds per protocol-model combination
    min_seeds = 3
    for protocol, models in protocol_model_seeds.items():
        for model, seeds in models.items():
            if len(seeds) < min_seeds:
                issues.append(f"{protocol} {model}: only {len(seeds)} seeds, need ≥{min_seeds}")
    
    # Check performance consistency
    performance_stats = defaultdict(list)
    for row in rows:
        if row.macro_f1 is not None:
            performance_stats[f"{row.protocol}_{row.model}"].append(row.macro_f1)
    
    for key, f1_scores in performance_stats.items():
        if len(f1_scores) >= 3:
            mean_f1 = sum(f1_scores) / len(f1_scores)
            std_f1 = (sum((x - mean_f1)**2 for x in f1_scores) / len(f1_scores))**0.5
            cv = std_f1 / mean_f1 if mean_f1 > 0 else float('inf')
            
            # Check for reasonable performance (>50% macro F1)
            if mean_f1 < 0.5:
                issues.append(f"{key}: low mean macro F1 = {mean_f1:.3f}")
            
            # Check for reasonable consistency (CV < 0.1)  
            if cv > 0.1:
                issues.append(f"{key}: high variability CV = {cv:.3f}")
    
    return len(issues) == 0, issues


def validate_d4_acceptance(rows: List[D4ResultRow]) -> Tuple[bool, List[str]]:
    """Validate D4 Sim2Real label efficiency acceptance criteria."""
    issues = []
    
    # Group by model and transfer method
    model_method_data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.target_f1 is not None and row.label_ratio is not None:
            model_method_data[row.model][row.transfer_method].append(row)
    
    # Check label efficiency: target is ≥90% performance with ≤20% labels
    target_performance = 0.90
    max_label_ratio = 0.20
    
    for model, methods in model_method_data.items():
        for method, results in methods.items():
            # Find best performance within target label ratio
            efficient_results = [r for r in results if r.label_ratio <= max_label_ratio]
            
            if not efficient_results:
                issues.append(f"{model} {method}: no results within {max_label_ratio*100}% label budget")
                continue
            
            best_f1 = max(r.target_f1 for r in efficient_results if r.target_f1 is not None)
            best_result = next(r for r in efficient_results if r.target_f1 == best_f1)
            
            if best_f1 < target_performance:
                issues.append(f"{model} {method}: best F1 {best_f1:.3f} < {target_performance:.3f} within {max_label_ratio*100}% labels")
            else:
                print(f"✓ {model} {method}: achieved {best_f1:.3f} F1 with {best_result.label_ratio*100}% labels")
    
    # Check minimum seeds coverage
    min_seeds = 3
    model_seeds = defaultdict(set)
    for row in rows:
        model_seeds[row.model].add(row.seed)
    
    for model, seeds in model_seeds.items():
        if len(seeds) < min_seeds:
            issues.append(f"{model}: only {len(seeds)} seeds, need ≥{min_seeds}")
    
    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Accept D3/D4 experiments")
    parser.add_argument("--d3_root", default="results/d3", help="D3 results root directory")
    parser.add_argument("--d4_root", default="results/d4", help="D4 results root directory")
    parser.add_argument("--output_dir", default="results/metrics", help="Output directory for reports")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    success = True
    all_issues = []
    
    # Process D3 results
    print("=== D3 Cross-Domain Validation ===")
    d3_rows = []
    d3_root = Path(args.d3_root)
    
    if d3_root.exists():
        for json_file in d3_root.rglob("*.json"):
            if "summary" not in json_file.name:
                row = parse_d3_result(json_file)
                if row:
                    d3_rows.append(row)
        
        d3_valid, d3_issues = validate_d3_acceptance(d3_rows)
        if not d3_valid:
            success = False
            all_issues.extend([f"D3: {issue}" for issue in d3_issues])
        
        # Export D3 summary
        d3_csv_path = output_dir / "summary_d3.csv"
        with open(d3_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'protocol', 'model', 'seed', 'macro_f1', 'ece', 'falling_f1', 'epileptic_fall_f1', 'elderly_fall_f1'])
            for row in d3_rows:
                writer.writerow([row.file, row.protocol, row.model, row.seed, row.macro_f1, row.ece, row.falling_f1, row.epileptic_fall_f1, row.elderly_fall_f1])
        
        print(f"D3 summary exported to {d3_csv_path}")
    else:
        print(f"D3 root directory {d3_root} not found")
        success = False
        all_issues.append("D3: Results directory not found")
    
    # Process D4 results  
    print("\n=== D4 Sim2Real Label Efficiency ===")
    d4_rows = []
    d4_root = Path(args.d4_root)
    
    if d4_root.exists():
        for json_file in d4_root.rglob("*.json"):
            if "summary" not in json_file.name:
                row = parse_d4_result(json_file)
                if row:
                    d4_rows.append(row)
        
        d4_valid, d4_issues = validate_d4_acceptance(d4_rows)
        if not d4_valid:
            success = False
            all_issues.extend([f"D4: {issue}" for issue in d4_issues])
        
        # Export D4 summary
        d4_csv_path = output_dir / "summary_d4.csv"
        with open(d4_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'model', 'seed', 'label_ratio', 'transfer_method', 'zero_shot_f1', 'target_f1', 'target_ece', 'improvement'])
            for row in d4_rows:
                writer.writerow([row.file, row.model, row.seed, row.label_ratio, row.transfer_method, row.zero_shot_f1, row.target_f1, row.target_ece, row.improvement])
        
        print(f"D4 summary exported to {d4_csv_path}")
    else:
        print(f"D4 root directory {d4_root} not found") 
        success = False
        all_issues.append("D4: Results directory not found")
    
    # Generate combined acceptance report
    report_path = output_dir / "d3_d4_acceptance_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("D3/D4 EXPERIMENTAL ACCEPTANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"D3 Results: {len(d3_rows)} configurations processed\n")
        f.write(f"D4 Results: {len(d4_rows)} configurations processed\n\n")
        
        if success:
            f.write("OVERALL STATUS: ✓ ACCEPTED\n\n")
        else:
            f.write("OVERALL STATUS: ✗ REJECTED\n\n")
            f.write("ISSUES FOUND:\n")
            for issue in all_issues:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        # D3 Analysis
        f.write("D3 CROSS-DOMAIN ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        d3_by_protocol = defaultdict(list)
        for row in d3_rows:
            d3_by_protocol[row.protocol].append(row)
        
        for protocol, protocol_rows in d3_by_protocol.items():
            f.write(f"\n{protocol} Protocol:\n")
            model_stats = defaultdict(list)
            for row in protocol_rows:
                if row.macro_f1 is not None:
                    model_stats[row.model].append(row.macro_f1)
            
            for model, f1_scores in model_stats.items():
                if f1_scores:
                    mean_f1 = sum(f1_scores) / len(f1_scores)
                    std_f1 = (sum((x - mean_f1)**2 for x in f1_scores) / len(f1_scores))**0.5 if len(f1_scores) > 1 else 0.0
                    f.write(f"  {model}: {mean_f1:.3f}±{std_f1:.3f} ({len(f1_scores)} seeds)\n")
        
        # D4 Analysis
        f.write("\nD4 SIM2REAL ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        d4_by_model = defaultdict(list)
        for row in d4_rows:
            d4_by_model[row.model].append(row)
        
        for model, model_rows in d4_by_model.items():
            f.write(f"\n{model} Model:\n")
            
            # Group by transfer method
            method_data = defaultdict(list)
            for row in model_rows:
                method_data[row.transfer_method].append(row)
            
            for method, method_rows in method_data.items():
                f.write(f"  {method}:\n")
                
                # Sort by label ratio
                method_rows.sort(key=lambda r: r.label_ratio)
                
                for row in method_rows:
                    if row.target_f1 is not None:
                        improvement = row.improvement if row.improvement is not None else 0.0
                        f.write(f"    {row.label_ratio*100:5.1f}% labels: {row.target_f1:.3f} F1 (+{improvement:.3f})\n")
    
    print(f"\nAcceptance report written to {report_path}")
    
    if success:
        print("✓ D3/D4 ACCEPTANCE: PASSED")
        return 0
    else:
        print("✗ D3/D4 ACCEPTANCE: FAILED")
        print("\nIssues found:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 2


if __name__ == "__main__":
    sys.exit(main())