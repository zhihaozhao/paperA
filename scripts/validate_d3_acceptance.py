#!/usr/bin/env python3
"""
D3 Cross-Domain Experiments Validation Script
Validates LOSO/LORO results against acceptance criteria
"""

import argparse
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class D3Validator:
    """Validator for D3 Cross-Domain Experiments"""
    
    def __init__(self, protocol: str):
        self.protocol = protocol.upper()
        
        # D3 Acceptance Criteria (based on cross-domain literature standards)
        self.criteria = {
            'min_falling_f1': 0.75,      # Cross-domain average
            'min_macro_f1': 0.80,        # Overall performance
            'max_ece_after_cal': 0.15,   # Calibration quality
            'min_enhanced_advantage': 0.05,  # Enhanced > baselines by 5% Falling F1
            'min_completion_rate': 0.90, # ≥90% of folds complete
            'required_models': ["enhanced", "cnn", "bilstm", "conformer_lite"],
            'required_seeds': [0, 1, 2, 3, 4]
        }
    
    def load_results(self, results_dir: str) -> Dict:
        """Load all results from the specified directory"""
        results_path = Path(results_dir)
        pattern = f"{self.protocol.lower()}_*.json"
        result_files = list(results_path.glob(pattern))
        
        if not result_files:
            raise FileNotFoundError(f"No {self.protocol} results found in {results_dir}")
        
        results = {}
        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract identifiers
                model = data.get('model', 'unknown')
                seed = data.get('seed', -1)
                key = f"{model}_seed{seed}"
                
                results[key] = data
                
            except Exception as e:
                print(f"[WARNING] Failed to load {file_path}: {e}")
        
        return results
    
    def validate_coverage(self, results: Dict) -> Dict:
        """Validate experimental coverage (models and seeds)"""
        validation = {
            'models_found': set(),
            'seeds_found': set(),
            'missing_models': set(),
            'missing_seeds': set(),
            'total_experiments': len(results)
        }
        
        for key, result in results.items():
            model = result.get('model', 'unknown')
            seed = result.get('seed', -1)
            
            validation['models_found'].add(model)
            validation['seeds_found'].add(seed)
        
        validation['missing_models'] = set(self.criteria['required_models']) - validation['models_found']
        validation['missing_seeds'] = set(self.criteria['required_seeds']) - validation['seeds_found']
        
        # Convert sets to lists for JSON serialization
        for key in ['models_found', 'seeds_found', 'missing_models', 'missing_seeds']:
            validation[key] = list(validation[key])
        
        validation['coverage_complete'] = (
            len(validation['missing_models']) == 0 and 
            len(validation['missing_seeds']) == 0
        )
        
        return validation
    
    def validate_performance(self, results: Dict) -> Dict:
        """Validate performance against criteria"""
        performance_stats = {
            'by_model': {},
            'overall': {
                'falling_f1': [],
                'macro_f1': [],
                'ece_cal': [],
                'completion_rates': []
            }
        }
        
        for key, result in results.items():
            model = result.get('model', 'unknown')
            fold_results = result.get('fold_results', [])
            aggregate_stats = result.get('aggregate_stats', {})
            
            if model not in performance_stats['by_model']:
                performance_stats['by_model'][model] = {
                    'experiments': 0,
                    'falling_f1': [],
                    'macro_f1': [],
                    'ece_cal': [],
                    'n_folds': []
                }
            
            performance_stats['by_model'][model]['experiments'] += 1
            
            # Extract metrics from aggregate stats
            if 'falling_f1' in aggregate_stats:
                falling_f1 = aggregate_stats['falling_f1']['mean']
                performance_stats['by_model'][model]['falling_f1'].append(falling_f1)
                performance_stats['overall']['falling_f1'].append(falling_f1)
            
            if 'macro_f1' in aggregate_stats:
                macro_f1 = aggregate_stats['macro_f1']['mean']
                performance_stats['by_model'][model]['macro_f1'].append(macro_f1)
                performance_stats['overall']['macro_f1'].append(macro_f1)
            
            # Check calibration from fold results
            ece_cal_values = []
            for fold in fold_results:
                cal_data = fold.get('calibration', {})
                if 'ece_cal' in cal_data:
                    ece_cal_values.append(cal_data['ece_cal'])
            
            if ece_cal_values:
                avg_ece_cal = np.mean(ece_cal_values)
                performance_stats['by_model'][model]['ece_cal'].append(avg_ece_cal)
                performance_stats['overall']['ece_cal'].append(avg_ece_cal)
            
            # Completion rate
            expected_folds = result.get('benchmark_metadata', {}).get('n_subjects', 0)
            if self.protocol == 'LORO':
                expected_folds = result.get('benchmark_metadata', {}).get('n_rooms', 0)
            
            if expected_folds > 0:
                completion_rate = len(fold_results) / expected_folds
                performance_stats['by_model'][model]['n_folds'].append(len(fold_results))
                performance_stats['overall']['completion_rates'].append(completion_rate)
        
        return performance_stats
    
    def check_acceptance_criteria(self, performance_stats: Dict, coverage: Dict) -> Dict:
        """Check all acceptance criteria"""
        checks = {
            'coverage_check': coverage['coverage_complete'],
            'performance_checks': {},
            'overall_pass': False
        }
        
        # Performance criteria checks
        overall = performance_stats['overall']
        
        if overall['falling_f1']:
            avg_falling_f1 = np.mean(overall['falling_f1'])
            checks['performance_checks']['falling_f1_pass'] = avg_falling_f1 >= self.criteria['min_falling_f1']
            checks['performance_checks']['falling_f1_value'] = avg_falling_f1
        
        if overall['macro_f1']:
            avg_macro_f1 = np.mean(overall['macro_f1'])
            checks['performance_checks']['macro_f1_pass'] = avg_macro_f1 >= self.criteria['min_macro_f1']
            checks['performance_checks']['macro_f1_value'] = avg_macro_f1
        
        if overall['ece_cal']:
            avg_ece_cal = np.mean(overall['ece_cal'])
            checks['performance_checks']['ece_pass'] = avg_ece_cal <= self.criteria['max_ece_after_cal']
            checks['performance_checks']['ece_value'] = avg_ece_cal
        
        if overall['completion_rates']:
            avg_completion = np.mean(overall['completion_rates'])
            checks['performance_checks']['completion_pass'] = avg_completion >= self.criteria['min_completion_rate']
            checks['performance_checks']['completion_value'] = avg_completion
        
        # Enhanced model advantage check
        if 'enhanced' in performance_stats['by_model']:
            enhanced_f1 = np.mean(performance_stats['by_model']['enhanced']['falling_f1'])
            baseline_f1s = []
            
            for model in ['cnn', 'bilstm', 'conformer_lite']:
                if model in performance_stats['by_model']:
                    baseline_f1s.extend(performance_stats['by_model'][model]['falling_f1'])
            
            if baseline_f1s:
                avg_baseline_f1 = np.mean(baseline_f1s)
                advantage = enhanced_f1 - avg_baseline_f1
                checks['performance_checks']['enhanced_advantage_pass'] = advantage >= self.criteria['min_enhanced_advantage']
                checks['performance_checks']['enhanced_advantage_value'] = advantage
        
        # Overall pass check
        performance_passes = [v for k, v in checks['performance_checks'].items() if k.endswith('_pass')]
        checks['overall_pass'] = (
            checks['coverage_check'] and 
            all(performance_passes) and 
            len(performance_passes) >= 4  # At least 4 criteria checked
        )
        
        return checks
    
    def generate_report(self, results: Dict, coverage: Dict, 
                       performance_stats: Dict, acceptance_checks: Dict) -> str:
        """Generate validation report"""
        report_lines = [
            f"# D3 {self.protocol} Validation Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Experimental Coverage",
            f"- Total experiments: {coverage['total_experiments']}",
            f"- Models found: {coverage['models_found']}",
            f"- Seeds found: {coverage['seeds_found']}",
            f"- Missing models: {coverage['missing_models']}",
            f"- Missing seeds: {coverage['missing_seeds']}",
            f"- Coverage complete: {'✅' if coverage['coverage_complete'] else '❌'}",
            "",
            "## Performance Summary",
        ]
        
        # Overall statistics
        overall = performance_stats['overall']
        if overall['falling_f1']:
            falling_f1_mean = np.mean(overall['falling_f1'])
            falling_f1_std = np.std(overall['falling_f1'])
            report_lines.append(f"- Falling F1: {falling_f1_mean:.3f}±{falling_f1_std:.3f}")
        
        if overall['macro_f1']:
            macro_f1_mean = np.mean(overall['macro_f1'])
            macro_f1_std = np.std(overall['macro_f1'])
            report_lines.append(f"- Macro F1: {macro_f1_mean:.3f}±{macro_f1_std:.3f}")
        
        if overall['ece_cal']:
            ece_mean = np.mean(overall['ece_cal'])
            ece_std = np.std(overall['ece_cal'])
            report_lines.append(f"- ECE (calibrated): {ece_mean:.3f}±{ece_std:.3f}")
        
        if overall['completion_rates']:
            completion_mean = np.mean(overall['completion_rates'])
            report_lines.append(f"- Completion rate: {completion_mean:.1%}")
        
        # Per-model breakdown
        report_lines.extend(["", "## Per-Model Results"])
        for model, stats in performance_stats['by_model'].items():
            if stats['falling_f1']:
                f1_mean = np.mean(stats['falling_f1'])
                f1_std = np.std(stats['falling_f1'])
                report_lines.append(f"- {model}: Falling F1 = {f1_mean:.3f}±{f1_std:.3f} (n={len(stats['falling_f1'])})")
        
        # Acceptance criteria
        report_lines.extend(["", "## Acceptance Criteria"])
        checks = acceptance_checks['performance_checks']
        
        for criterion, threshold in [
            ('falling_f1', self.criteria['min_falling_f1']),
            ('macro_f1', self.criteria['min_macro_f1']),
            ('ece', self.criteria['max_ece_after_cal']),
            ('completion', self.criteria['min_completion_rate']),
            ('enhanced_advantage', self.criteria['min_enhanced_advantage'])
        ]:
            key_pass = f"{criterion}_pass"
            key_value = f"{criterion}_value"
            
            if key_pass in checks:
                status = "✅" if checks[key_pass] else "❌"
                value = checks.get(key_value, 0.0)
                
                if criterion == 'ece':
                    report_lines.append(f"- {criterion}: {value:.3f} ≤ {threshold} {status}")
                elif criterion == 'completion':
                    report_lines.append(f"- {criterion}: {value:.1%} ≥ {threshold:.1%} {status}")
                else:
                    report_lines.append(f"- {criterion}: {value:.3f} ≥ {threshold} {status}")
        
        # Overall result
        overall_status = "✅ PASS" if acceptance_checks['overall_pass'] else "❌ FAIL"
        report_lines.extend([
            "",
            f"## Overall Result: {overall_status}",
            ""
        ])
        
        if not acceptance_checks['overall_pass']:
            report_lines.extend([
                "### Issues to Address:",
                "- Check failed criteria above",
                "- Ensure all required models and seeds are run",
                "- Verify benchmark dataset quality",
                "- Consider hyperparameter tuning if performance is low",
                ""
            ])
        else:
            report_lines.extend([
                "### Next Steps:",
                f"- ✅ D3 {self.protocol} validation passed",
                "- Continue to next experimental phase",
                "- Prepare results for publication",
                ""
            ])
        
        return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Validate D3 Cross-Domain Experiments")
    parser.add_argument("--protocol", type=str, required=True, choices=["loso", "loro"])
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Results directory (defaults to results/d3/{protocol})")
    parser.add_argument("--output_report", type=str, default=None,
                       help="Output report file (defaults to validation report in results dir)")
    
    args = parser.parse_args()
    
    # Default paths
    if args.results_dir is None:
        args.results_dir = f"results/d3/{args.protocol}"
    
    if args.output_report is None:
        args.output_report = f"{args.results_dir}/d3_{args.protocol}_validation_report.md"
    
    print(f"[INFO] Validating D3 {args.protocol.upper()} experiments")
    print(f"[INFO] Results directory: {args.results_dir}")
    
    # Initialize validator
    validator = D3Validator(args.protocol)
    
    try:
        # Load results
        results = validator.load_results(args.results_dir)
        print(f"[INFO] Loaded {len(results)} experiment results")
        
        # Validate coverage
        coverage = validator.validate_coverage(results)
        
        # Validate performance
        performance_stats = validator.validate_performance(results)
        
        # Check acceptance criteria
        acceptance_checks = validator.check_acceptance_criteria(performance_stats, coverage)
        
        # Generate report
        report = validator.generate_report(results, coverage, performance_stats, acceptance_checks)
        
        # Save report
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Print summary
        print(f"\n[SUMMARY] D3 {args.protocol.upper()} Validation Results:")
        print(f"- Experiments: {len(results)}")
        print(f"- Coverage: {'✅' if coverage['coverage_complete'] else '❌'}")
        print(f"- Performance: {'✅' if acceptance_checks['overall_pass'] else '❌'}")
        print(f"- Report saved: {args.output_report}")
        
        if acceptance_checks['overall_pass']:
            print(f"\n🎉 D3 {args.protocol.upper()} experiments PASSED validation!")
            return 0
        else:
            print(f"\n❌ D3 {args.protocol.upper()} experiments FAILED validation")
            print("Please check the detailed report for specific issues")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())