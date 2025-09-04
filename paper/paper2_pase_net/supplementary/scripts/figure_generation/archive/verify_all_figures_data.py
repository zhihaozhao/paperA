#!/usr/bin/env python3
"""
Comprehensive data verification for all figures in the paper
Checks if data is from real experiments or fabricated
"""

import json
import os
from pathlib import Path
import glob
import numpy as np

# Define paths
RESULTS_DIR = Path("/workspace/results")
RESULTS_GPU_DIR = Path("/workspace/results_gpu")
PLOTS_DIR = Path("/workspace/paper/enhanced/plots")

def check_file_for_hardcoded_data(filepath):
    """Check if a Python file contains hardcoded performance data"""
    hardcoded_indicators = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines, 1):
        # Check for hardcoded arrays
        if 'np.array([' in line or 'performance_matrix' in line:
            # Check if followed by hardcoded numbers
            if any(x in line for x in ['0.8', '0.9', '0.7', '[0.', '[1.']):
                hardcoded_indicators.append(f"Line {i}: {line.strip()}")
        
        # Check for manual data creation
        if any(phrase in line.lower() for phrase in 
               ['simulate', 'fake', 'example', 'dummy', 'hardcoded', 'manual']):
            hardcoded_indicators.append(f"Line {i}: {line.strip()}")
            
        # Check for suspicious perfect scores
        if '1.0' in line and 'f1' in line.lower():
            hardcoded_indicators.append(f"Line {i} (Perfect score): {line.strip()}")
    
    return hardcoded_indicators

def check_data_source(script_file):
    """Check if script loads data from results folders"""
    uses_real_data = False
    data_sources = []
    
    with open(script_file, 'r') as f:
        content = f.read()
    
    # Check for results folder references
    if 'results/' in content or 'results_gpu/' in content:
        uses_real_data = True
        if 'results/' in content:
            data_sources.append('results/')
        if 'results_gpu/' in content:
            data_sources.append('results_gpu/')
    
    # Check for JSON file loading
    if 'json.load' in content or 'pd.read_json' in content:
        uses_real_data = True
        data_sources.append('JSON files')
    
    return uses_real_data, data_sources

def verify_figure_data():
    """Main verification function"""
    print("="*80)
    print("DATA INTEGRITY VERIFICATION REPORT FOR ALL FIGURES")
    print("="*80)
    
    # Define figures and their generating scripts
    figures = {
        'Figure 1 (System Architecture)': 'scr1_system_architecture.py',
        'Figure 2 (Physics Modeling)': 'scr2_physics_modeling.py',
        'Figure 3 (Cross-Domain)': 'scr3_cross_domain.py',
        'Figure 4 (Calibration)': 'scr4_calibration.py',
        'Figure 5 (Label Efficiency)': 'scr5_label_efficiency.py',
        'Figure 6 (Interpretability)': 'scr6_interpretability.py',
    }
    
    issues_found = []
    
    for fig_name, script_name in figures.items():
        script_path = PLOTS_DIR / script_name
        
        print(f"\n{'-'*60}")
        print(f"Checking: {fig_name}")
        print(f"Script: {script_name}")
        print(f"{'-'*60}")
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            issues_found.append(f"{fig_name}: Script missing")
            continue
        
        # Check for hardcoded data
        hardcoded = check_file_for_hardcoded_data(script_path)
        if hardcoded:
            print(f"⚠️  HARDCODED DATA FOUND:")
            for item in hardcoded[:5]:  # Show first 5
                print(f"   {item}")
            if len(hardcoded) > 5:
                print(f"   ... and {len(hardcoded)-5} more instances")
            issues_found.append(f"{fig_name}: {len(hardcoded)} hardcoded values")
        
        # Check data source
        uses_real, sources = check_data_source(script_path)
        if uses_real:
            print(f"✅ Loads data from: {', '.join(sources)}")
        else:
            print(f"❌ NO REAL DATA SOURCE FOUND")
            issues_found.append(f"{fig_name}: No real data source")
    
    # Check for specific known issues
    print(f"\n{'='*80}")
    print("SPECIFIC DATA INTEGRITY CHECKS")
    print(f"{'='*80}")
    
    # Check if Conformer experiments exist
    conformer_files = list(RESULTS_GPU_DIR.glob("**/paperA_conformer*.json")) + \
                     list(RESULTS_DIR.glob("**/paperA_conformer*.json"))
    print(f"\nConformer experiments found: {len(conformer_files)}")
    if len(conformer_files) == 0:
        print("❌ CRITICAL: No Conformer experiments found but included in figures!")
        issues_found.append("CRITICAL: Conformer data is fabricated")
    
    # Check for perfect scores
    perfect_scores = []
    for results_dir in [RESULTS_DIR, RESULTS_GPU_DIR]:
        for json_file in results_dir.glob("**/*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if 'metrics' in data and 'macro_f1' in data['metrics']:
                    if data['metrics']['macro_f1'] >= 0.99:
                        perfect_scores.append(json_file.name)
            except:
                pass
    
    if perfect_scores:
        print(f"\n⚠️  Suspiciously perfect scores (≥99%) in {len(perfect_scores)} files:")
        for f in perfect_scores[:5]:
            print(f"   {f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ISSUES")
    print(f"{'='*80}")
    
    if issues_found:
        print(f"\n❌ {len(issues_found)} ISSUES FOUND:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("\n✅ No major issues found (but manual review still recommended)")
    
    return issues_found

def check_specific_values():
    """Check specific suspicious values mentioned in figures"""
    print(f"\n{'='*80}")
    print("CHECKING SPECIFIC CLAIMED VALUES")
    print(f"{'='*80}")
    
    # Check Table 1 values
    print("\nTable 1 claimed values:")
    claimed_values = {
        'PASE-Net LOSO': 83.0,
        'PASE-Net LORO': 83.0,
        'CNN LOSO': 79.4,
        'BiLSTM LOSO': 81.2,
        'Conformer LOSO': 82.1
    }
    
    for metric, value in claimed_values.items():
        print(f"  {metric}: {value}% - ", end="")
        # Try to find this value in actual results
        found = False
        # This would need actual checking logic
        print("⚠️  Cannot verify without complete data")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Run verification
    issues = verify_figure_data()
    check_specific_values()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    if len(issues) > 0:
        print("\n🚨 CRITICAL DATA INTEGRITY ISSUES DETECTED!")
        print("DO NOT SUBMIT THE PAPER WITHOUT ADDRESSING THESE ISSUES!")
        print("\nRecommendations:")
        print("1. Complete all missing experiments")
        print("2. Use only real experimental data in figures")
        print("3. Clearly mark any illustrative/example figures")
        print("4. Be transparent about data limitations")
    else:
        print("\n✅ Basic checks passed, but manual verification still needed")
    
    print("\n" + "="*80)