#!/usr/bin/env python3
"""
D5 & D6 Acceptance Criteria Validation Script
Validates experiment results against acceptance criteria
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_results(pattern: str) -> pd.DataFrame:
    """Load all JSON results matching pattern"""
    import glob
    results = []
    
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract key metrics
                result = {
                    'file': file_path,
                    'model': data.get('meta', {}).get('model', 'unknown'),
                    'seed': data.get('meta', {}).get('seed', -1),
                    'macro_f1': data.get('metrics', {}).get('macro_f1', 0.0),
                    'ece_raw': data.get('metrics', {}).get('ece_raw', 1.0),
                    'ece_cal': data.get('metrics', {}).get('ece_cal', 1.0),
                    'nll_raw': data.get('metrics', {}).get('nll_raw', 10.0),
                    'nll_cal': data.get('metrics', {}).get('nll_cal', 10.0),
                    'brier': data.get('metrics', {}).get('brier', 1.0),
                    'temperature': data.get('metrics', {}).get('temperature', 1.0),
                }
                results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    return pd.DataFrame(results)

def validate_d5_criteria(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate D5 acceptance criteria"""
    criteria = {}
    
    # Coverage check
    models = df['model'].unique()
    seeds_per_model = df.groupby('model')['seed'].nunique()
    criteria['coverage'] = all(seeds_per_model >= 3)
    
    # Enhanced vs baseline performance
    if 'enhanced' in models and len(models) > 1:
        enhanced_f1 = df[df['model'] == 'enhanced']['macro_f1'].mean()
        baseline_f1 = df[df['model'] != 'enhanced']['macro_f1'].mean()
        criteria['enhanced_superior'] = enhanced_f1 >= baseline_f1 * 1.05  # 5% improvement
    
    # Calibration improvement
    if 'ece_raw' in df.columns and 'ece_cal' in df.columns:
        ece_improvement = (df['ece_raw'] - df['ece_cal']) / df['ece_raw']
        criteria['calibration_improvement'] = ece_improvement.mean() >= 0.05  # 5% relative improvement
    
    # ECE threshold
    criteria['ece_threshold'] = df['ece_cal'].mean() <= 0.15
    
    return criteria

def validate_d6_criteria(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate D6 acceptance criteria"""
    criteria = {}
    
    # Robustness check (if multiple seeds)
    if df['seed'].nunique() >= 2:
        seed_consistency = df.groupby('model')['macro_f1'].std()
        criteria['robustness'] = seed_consistency.mean() <= 0.05  # 5% std threshold
    
    # Enhanced vs baseline
    if 'enhanced' in df['model'].unique() and len(df['model'].unique()) > 1:
        enhanced_f1 = df[df['model'] == 'enhanced']['macro_f1'].mean()
        baseline_f1 = df[df['model'] != 'enhanced']['macro_f1'].mean()
        criteria['enhanced_advantage'] = enhanced_f1 >= baseline_f1 * 1.05
    
    # Calibration quality
    criteria['calibration_quality'] = df['ece_cal'].mean() <= 0.15
    
    return criteria

def generate_report(df: pd.DataFrame, d5_criteria: Dict[str, bool], d6_criteria: Dict[str, bool]) -> str:
    """Generate acceptance report"""
    report = []
    report.append("=" * 60)
    report.append("D5 & D6 ACCEPTANCE CRITERIA VALIDATION REPORT")
    report.append("=" * 60)
    
    # Summary statistics
    report.append(f"\n📊 EXPERIMENT SUMMARY:")
    report.append(f"  Total runs: {len(df)}")
    report.append(f"  Models: {', '.join(df['model'].unique())}")
    report.append(f"  Seeds per model: {df.groupby('model')['seed'].nunique().to_dict()}")
    
    # Performance summary
    report.append(f"\n📈 PERFORMANCE SUMMARY:")
    perf_summary = df.groupby('model').agg({
        'macro_f1': ['mean', 'std'],
        'ece_cal': ['mean', 'std']
    }).round(3)
    report.append(perf_summary.to_string())
    
    # D5 Criteria
    report.append(f"\n🔬 D5 ABLATION STUDY CRITERIA:")
    for criterion, passed in d5_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report.append(f"  {criterion}: {status}")
    
    # D6 Criteria
    report.append(f"\n🛡️ D6 ROBUSTNESS CRITERIA:")
    for criterion, passed in d6_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report.append(f"  {criterion}: {status}")
    
    # Overall assessment
    d5_pass_rate = sum(d5_criteria.values()) / len(d5_criteria)
    d6_pass_rate = sum(d6_criteria.values()) / len(d6_criteria)
    
    report.append(f"\n📋 OVERALL ASSESSMENT:")
    report.append(f"  D5 Pass Rate: {d5_pass_rate:.1%} ({sum(d5_criteria.values())}/{len(d5_criteria)})")
    report.append(f"  D6 Pass Rate: {d6_pass_rate:.1%} ({sum(d6_criteria.values())}/{len(d6_criteria)})")
    
    if d5_pass_rate >= 0.8 and d6_pass_rate >= 0.8:
        report.append(f"  🎉 OVERALL STATUS: ACCEPTED")
    else:
        report.append(f"  ⚠️ OVERALL STATUS: NEEDS ATTENTION")
    
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    # Load results from both D5 and D6 directories
    print("Loading experiment results...")
    df_d5 = load_results("results_gpu/d5/*.json")
    df_d6 = load_results("results_gpu/d6/*.json")
    
    # Combine results
    df = pd.concat([df_d5, df_d6], ignore_index=True)
    
    if df.empty:
        print("❌ No results found! Please run experiments first.")
        return 1
    
    print(f"✅ Loaded {len(df)} results")
    print(f"  - D5 results: {len(df_d5)} files")
    print(f"  - D6 results: {len(df_d6)} files")
    
    # Validate criteria
    print("Validating acceptance criteria...")
    d5_criteria = validate_d5_criteria(df)
    d6_criteria = validate_d6_criteria(df)
    
    # Generate report
    report = generate_report(df, d5_criteria, d6_criteria)
    print(report)
    
    # Save report
    report_file = "results/d5_d6_acceptance_report.txt"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
