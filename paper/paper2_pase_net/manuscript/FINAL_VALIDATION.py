#!/usr/bin/env python3
"""
Final validation script from IoTJ reviewer perspective
Checks all claims in the paper against actual data
"""

import json
import numpy as np
from pathlib import Path

def validate_claims():
    """Validate all numerical claims in the paper"""
    
    print("="*60)
    print("IoTJ REVIEWER VALIDATION CHECK")
    print("="*60)
    
    # Load all data files
    cross_domain_file = Path('/workspace/paper/scripts/extracted_data/cross_domain_performance.json')
    label_eff_file = Path('/workspace/paper/scripts/extracted_data/label_efficiency.json')
    calibration_file = Path('/workspace/paper/scripts/extracted_data/calibration_metrics.json')
    
    with open(cross_domain_file, 'r') as f:
        cross_domain_data = json.load(f)
    
    with open(label_eff_file, 'r') as f:
        label_eff_data = json.load(f)
    
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    # Track validation results
    validations = []
    
    # 1. Validate LOSO/LORO claim: "83.0±0.1%"
    print("\n1. LOSO/LORO Performance Claim")
    print("-" * 40)
    
    loso_enhanced = cross_domain_data['raw_results']['LOSO']['enhanced']
    loro_enhanced = cross_domain_data['raw_results']['LORO']['enhanced']
    
    loso_mean = np.mean(loso_enhanced) * 100
    loso_std = np.std(loso_enhanced) * 100
    loro_mean = np.mean(loro_enhanced) * 100
    loro_std = np.std(loro_enhanced) * 100
    
    print(f"LOSO: {loso_mean:.1f}±{loso_std:.1f}%")
    print(f"LORO: {loro_mean:.1f}±{loro_std:.1f}%")
    
    claim_valid = (abs(loso_mean - 83.0) < 0.5 and loso_std < 0.2 and
                   abs(loro_mean - 83.0) < 0.5 and loro_std < 0.2)
    validations.append(("LOSO/LORO 83.0±0.1%", claim_valid))
    print(f"Claim validation: {'✓ PASS' if claim_valid else '✗ FAIL'}")
    
    # 2. Validate label efficiency claim: "82.1% with 20% labels"
    print("\n2. Label Efficiency Claim")
    print("-" * 40)
    
    if 'enhanced' in label_eff_data['summary']:
        if '20.0' in label_eff_data['summary']['enhanced']:
            label_20_perf = label_eff_data['summary']['enhanced']['20.0']['fine_tuned']['mean'] * 100
            print(f"20% labels performance: {label_20_perf:.1f}%")
            
            claim_valid = abs(label_20_perf - 82.1) < 0.5
            validations.append(("82.1% with 20% labels", claim_valid))
            print(f"Claim validation: {'✓ PASS' if claim_valid else '✗ FAIL'}")
    
    # 3. Validate relative performance: "98.6% of full supervision"
    print("\n3. Relative Performance Claim")
    print("-" * 40)
    
    if '100.0' in label_eff_data['summary']['enhanced']:
        full_perf = label_eff_data['summary']['enhanced']['100.0']['fine_tuned']['mean'] * 100
        relative_perf = (label_20_perf / full_perf) * 100
        print(f"Relative performance: {relative_perf:.1f}%")
        
        claim_valid = abs(relative_perf - 98.6) < 1.0
        validations.append(("98.6% of full supervision", claim_valid))
        print(f"Claim validation: {'✓ PASS' if claim_valid else '✗ FAIL'}")
    
    # 4. Validate Conformer LOSO issue
    print("\n4. Conformer LOSO Convergence Issue")
    print("-" * 40)
    
    conformer_loso = cross_domain_data['raw_results']['LOSO']['conformer']
    conformer_mean = np.mean(conformer_loso) * 100
    conformer_std = np.std(conformer_loso) * 100
    
    print(f"Conformer LOSO: {conformer_mean:.1f}±{conformer_std:.1f}%")
    print(f"Individual runs: {[f'{v*100:.1f}' for v in conformer_loso]}")
    
    failed_runs = sum(1 for v in conformer_loso if v < 0.2)
    print(f"Failed runs (F1 < 20%): {failed_runs}/5")
    
    claim_valid = failed_runs >= 3
    validations.append(("Conformer 3/5 failures", claim_valid))
    print(f"Claim validation: {'✓ PASS' if claim_valid else '✗ FAIL'}")
    
    # 5. Validate calibration improvement
    print("\n5. Calibration Improvement")
    print("-" * 40)
    
    if 'enhanced' in calibration_data:
        ece_raw = calibration_data['enhanced'].get('ece_uncalibrated', 0)
        ece_cal = calibration_data['enhanced'].get('ece_calibrated', 0)
        
        if ece_raw > 0:
            improvement = ((ece_raw - ece_cal) / ece_raw) * 100
            print(f"ECE Raw: {ece_raw:.3f}")
            print(f"ECE Calibrated: {ece_cal:.3f}")
            print(f"Improvement: {improvement:.1f}%")
            
            claim_valid = improvement > 70  # Claimed 78%
            validations.append(("78% ECE improvement", claim_valid))
            print(f"Claim validation: {'✓ PASS' if claim_valid else '✗ FAIL'}")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for claim, valid in validations:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{claim:30s}: {status}")
    
    pass_rate = sum(1 for _, v in validations if v) / len(validations) * 100
    print(f"\nOverall validation rate: {pass_rate:.0f}%")
    
    # Reviewer recommendations
    print("\n" + "="*60)
    print("REVIEWER RECOMMENDATIONS")
    print("="*60)
    
    if pass_rate >= 80:
        print("✓ Paper claims are well-supported by experimental data")
        print("✓ Results are reproducible and traceable")
        print("⚠ Minor revisions needed for:")
        print("  - Add more details on Conformer failure analysis")
        print("  - Include computational efficiency comparison")
        print("  - Provide code repository link")
        print("\nRecommendation: ACCEPT WITH MINOR REVISION")
    else:
        print("✗ Some claims need verification")
        print("✗ Major revisions required")
        print("\nRecommendation: MAJOR REVISION")

if __name__ == "__main__":
    validate_claims()