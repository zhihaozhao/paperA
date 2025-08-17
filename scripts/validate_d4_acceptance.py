#!/usr/bin/env python3
"""
D4 Sim2Real Experiments Validation Script
Validates label efficiency and transfer learning results against acceptance criteria
"""

import argparse
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

class D4Validator:
    """Validator for D4 Sim2Real Experiments"""
    
    def __init__(self):
        # D4 Acceptance Criteria (based on Sim2Real literature standards)
        self.criteria = {
            'min_zero_shot_falling_f1': 0.60,    # Zero-shot baseline
            'min_zero_shot_macro_f1': 0.70,      # Zero-shot overall
            'min_transfer_gain': 0.15,           # Fine-tuning improvement over zero-shot
            'max_sim2real_ece_gap': 0.10,        # Calibration degradation limit
            'target_label_efficiency': {         # Label efficiency targets
                '10_percent': 0.90,              # 10% labels → 90% of full performance
                '20_percent': 0.95               # 20% labels → 95% of full performance
            },
            'min_seeds_per_config': 3,           # Statistical robustness
            'required_models': ["enhanced", "cnn", "bilstm", "conformer_lite"],
            'required_methods': ["zero_shot", "linear_probe", "fine_tune", "temp_scale"],
            'required_ratios': [0.01, 0.05, 0.10, 0.15, 0.20, 0.50, 1.00]
        }
    
    def load_results(self, results_dir: str) -> Dict:
        """Load all Sim2Real results"""
        results_path = Path(results_dir)
        result_files = list(results_path.glob("sim2real_*.json"))
        
        if not result_files:
            raise FileNotFoundError(f"No Sim2Real results found in {results_dir}")
        
        results = {}
        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract identifiers
                model = data.get('model', 'unknown')
                method = data.get('transfer_method', 'unknown')
                ratio = data.get('label_ratio', 0.0)
                seed = data.get('seed', -1)
                
                key = f"{model}_{method}_ratio{ratio:.2f}_seed{seed}"
                results[key] = data
                
            except Exception as e:
                print(f"[WARNING] Failed to load {file_path}: {e}")
        
        return results
    
    def validate_coverage(self, results: Dict) -> Dict:
        """Validate experimental coverage"""
        found = {
            'models': set(),
            'methods': set(),
            'ratios': set(),
            'seeds': set()
        }
        
        config_coverage = defaultdict(set)  # (model, method, ratio) -> seeds
        
        for key, result in results.items():
            model = result.get('model', 'unknown')
            method = result.get('transfer_method', 'unknown')
            ratio = result.get('label_ratio', 0.0)
            seed = result.get('seed', -1)
            
            found['models'].add(model)
            found['methods'].add(method)
            found['ratios'].add(ratio)
            found['seeds'].add(seed)
            
            config_key = (model, method, ratio)
            config_coverage[config_key].add(seed)
        
        # Check completeness
        validation = {
            'total_experiments': len(results),
            'models_found': list(found['models']),
            'methods_found': list(found['methods']),
            'ratios_found': sorted(list(found['ratios'])),
            'seeds_found': sorted(list(found['seeds'])),
            'missing_models': list(set(self.criteria['required_models']) - found['models']),
            'missing_methods': list(set(self.criteria['required_methods']) - found['methods']),
            'missing_ratios': list(set(self.criteria['required_ratios']) - found['ratios']),
            'config_coverage': {}
        }
        
        # Check per-configuration seed coverage
        insufficient_seeds = []
        for config_key, seeds in config_coverage.items():
            model, method, ratio = config_key
            n_seeds = len(seeds)
            
            config_str = f"{model}_{method}_ratio{ratio:.2f}"
            validation['config_coverage'][config_str] = {
                'seeds': sorted(list(seeds)),
                'n_seeds': n_seeds,
                'sufficient': n_seeds >= self.criteria['min_seeds_per_config']
            }
            
            if n_seeds < self.criteria['min_seeds_per_config']:
                insufficient_seeds.append(config_str)
        
        validation['insufficient_seed_configs'] = insufficient_seeds
        validation['coverage_complete'] = (
            len(validation['missing_models']) == 0 and
            len(validation['missing_methods']) == 0 and
            len(insufficient_seeds) == 0
        )
        
        return validation
    
    def validate_label_efficiency(self, results: Dict) -> Dict:
        """Validate label efficiency criteria"""
        efficiency_stats = defaultdict(lambda: defaultdict(list))
        
        # Group results by model and method
        for key, result in results.items():
            model = result.get('model', 'unknown')
            method = result.get('transfer_method', 'unknown')
            ratio = result.get('label_ratio', 0.0)
            metrics = result.get('metrics', {})
            
            falling_f1 = metrics.get('falling_f1', 0.0)
            macro_f1 = metrics.get('macro_f1', 0.0)
            
            efficiency_stats[model][f"{method}_ratio{ratio:.2f}"].append({
                'falling_f1': falling_f1,
                'macro_f1': macro_f1,
                'ratio': ratio,
                'method': method
            })
        
        # Compute efficiency metrics
        efficiency_validation = {
            'by_model': {},
            'efficiency_targets_met': {}
        }
        
        for model, methods_data in efficiency_stats.items():
            model_stats = {
                'zero_shot_baseline': {},
                'best_method_at_ratio': {},
                'full_supervision_reference': {},
                'efficiency_scores': {}
            }
            
            # Get zero-shot baseline
            zero_shot_100 = methods_data.get('zero_shot_ratio1.00', [])
            if zero_shot_100:
                model_stats['zero_shot_baseline'] = {
                    'falling_f1': np.mean([r['falling_f1'] for r in zero_shot_100]),
                    'macro_f1': np.mean([r['macro_f1'] for r in zero_shot_100])
                }
            
            # Get full supervision reference (best method at 100%)
            full_sup_results = []
            for method in ['fine_tune', 'linear_probe', 'zero_shot']:
                method_100 = methods_data.get(f'{method}_ratio1.00', [])
                if method_100:
                    avg_f1 = np.mean([r['falling_f1'] for r in method_100])
                    full_sup_results.append(avg_f1)
            
            if full_sup_results:
                model_stats['full_supervision_reference'] = {
                    'falling_f1': max(full_sup_results)
                }
            
            # Check efficiency targets
            for target_ratio, target_performance in [(0.10, 0.90), (0.20, 0.95)]:
                best_f1_at_ratio = 0
                best_method_at_ratio = None
                
                for method in self.criteria['required_methods']:
                    if method == 'zero_shot':
                        continue  # Skip zero-shot for efficiency analysis
                        
                    method_key = f"{method}_ratio{target_ratio:.2f}"
                    if method_key in methods_data:
                        avg_f1 = np.mean([r['falling_f1'] for r in methods_data[method_key]])
                        if avg_f1 > best_f1_at_ratio:
                            best_f1_at_ratio = avg_f1
                            best_method_at_ratio = method
                
                if best_f1_at_ratio > 0 and model_stats['full_supervision_reference']:
                    full_sup_f1 = model_stats['full_supervision_reference']['falling_f1']
                    efficiency_ratio = best_f1_at_ratio / full_sup_f1
                    
                    model_stats['efficiency_scores'][f'{int(target_ratio*100)}percent'] = {
                        'achieved_f1': best_f1_at_ratio,
                        'target_f1': full_sup_f1 * target_performance,
                        'efficiency_ratio': efficiency_ratio,
                        'target_met': efficiency_ratio >= target_performance,
                        'best_method': best_method_at_ratio
                    }
            
            efficiency_validation['by_model'][model] = model_stats
        
        return efficiency_validation
    
    def validate_transfer_methods(self, results: Dict) -> Dict:
        """Validate transfer learning method effectiveness"""
        method_validation = {}
        
        # Group by model and analyze transfer gains
        model_results = defaultdict(lambda: defaultdict(list))
        
        for key, result in results.items():
            model = result.get('model', 'unknown')
            method = result.get('transfer_method', 'unknown')
            metrics = result.get('metrics', {})
            
            model_results[model][method].append({
                'falling_f1': metrics.get('falling_f1', 0.0),
                'macro_f1': metrics.get('macro_f1', 0.0),
                'ece': metrics.get('ece', 0.0)
            })
        
        for model, methods in model_results.items():
            if 'zero_shot' in methods and 'fine_tune' in methods:
                zero_shot_f1 = np.mean([r['falling_f1'] for r in methods['zero_shot']])
                fine_tune_f1 = np.mean([r['falling_f1'] for r in methods['fine_tune']])
                
                transfer_gain = fine_tune_f1 - zero_shot_f1
                
                method_validation[model] = {
                    'zero_shot_f1': zero_shot_f1,
                    'fine_tune_f1': fine_tune_f1,
                    'transfer_gain': transfer_gain,
                    'transfer_gain_pass': transfer_gain >= self.criteria['min_transfer_gain']
                }
        
        return method_validation
    
    def check_acceptance_criteria(self, coverage: Dict, efficiency: Dict, 
                                 transfer_methods: Dict) -> Dict:
        """Check all D4 acceptance criteria"""
        checks = {
            'coverage_check': coverage['coverage_complete'],
            'zero_shot_checks': {},
            'efficiency_checks': {},
            'transfer_checks': {},
            'overall_pass': False
        }
        
        # Zero-shot baseline checks
        zero_shot_f1s = []
        for model, stats in efficiency['by_model'].items():
            baseline = stats.get('zero_shot_baseline', {})
            if 'falling_f1' in baseline:
                zero_shot_f1s.append(baseline['falling_f1'])
        
        if zero_shot_f1s:
            avg_zero_shot_f1 = np.mean(zero_shot_f1s)
            checks['zero_shot_checks'] = {
                'falling_f1_pass': avg_zero_shot_f1 >= self.criteria['min_zero_shot_falling_f1'],
                'falling_f1_value': avg_zero_shot_f1
            }
        
        # Label efficiency checks
        efficiency_passes = []
        for model, stats in efficiency['by_model'].items():
            for target, target_data in stats.get('efficiency_scores', {}).items():
                if target_data.get('target_met', False):
                    efficiency_passes.append(True)
                else:
                    efficiency_passes.append(False)
        
        checks['efficiency_checks'] = {
            'efficiency_targets_met': sum(efficiency_passes),
            'efficiency_targets_total': len(efficiency_passes),
            'efficiency_pass': len(efficiency_passes) > 0 and np.mean(efficiency_passes) >= 0.5
        }
        
        # Transfer method checks
        transfer_passes = []
        for model, stats in transfer_methods.items():
            if stats.get('transfer_gain_pass', False):
                transfer_passes.append(True)
            else:
                transfer_passes.append(False)
        
        checks['transfer_checks'] = {
            'models_with_transfer_gain': sum(transfer_passes),
            'total_models_checked': len(transfer_passes),
            'transfer_pass': len(transfer_passes) > 0 and np.mean(transfer_passes) >= 0.75
        }
        
        # Overall pass check
        checks['overall_pass'] = (
            checks['coverage_check'] and
            checks.get('zero_shot_checks', {}).get('falling_f1_pass', False) and
            checks['efficiency_checks']['efficiency_pass'] and
            checks['transfer_checks']['transfer_pass']
        )
        
        return checks
    
    def generate_report(self, results: Dict, coverage: Dict, efficiency: Dict,
                       transfer_methods: Dict, acceptance_checks: Dict) -> str:
        """Generate comprehensive D4 validation report"""
        report_lines = [
            "# D4 Sim2Real Validation Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Experimental Coverage",
            f"- Total experiments: {coverage['total_experiments']}",
            f"- Models: {coverage['models_found']} (missing: {coverage['missing_models']})",
            f"- Methods: {coverage['methods_found']} (missing: {coverage['missing_methods']})",
            f"- Label ratios: {len(coverage['ratios_found'])}/{len(self.criteria['required_ratios'])}",
            f"- Coverage complete: {'✅' if coverage['coverage_complete'] else '❌'}",
            "",
            "## Zero-Shot Baseline Performance"
        ]
        
        # Zero-shot results
        zero_shot_checks = acceptance_checks.get('zero_shot_checks', {})
        if 'falling_f1_value' in zero_shot_checks:
            f1_value = zero_shot_checks['falling_f1_value']
            f1_pass = zero_shot_checks['falling_f1_pass']
            threshold = self.criteria['min_zero_shot_falling_f1']
            status = "✅" if f1_pass else "❌"
            report_lines.append(f"- Zero-shot Falling F1: {f1_value:.3f} ≥ {threshold} {status}")
        
        # Label efficiency analysis
        report_lines.extend(["", "## Label Efficiency Analysis"])
        
        efficiency_checks = acceptance_checks['efficiency_checks']
        targets_met = efficiency_checks['efficiency_targets_met']
        targets_total = efficiency_checks['efficiency_targets_total']
        
        report_lines.append(f"- Efficiency targets met: {targets_met}/{targets_total}")
        
        # Per-model efficiency breakdown
        for model, stats in efficiency['by_model'].items():
            report_lines.append(f"\n### {model.capitalize()} Model")
            
            baseline = stats.get('zero_shot_baseline', {})
            if 'falling_f1' in baseline:
                report_lines.append(f"- Zero-shot baseline: {baseline['falling_f1']:.3f}")
            
            for target, target_data in stats.get('efficiency_scores', {}).items():
                achieved = target_data['achieved_f1']
                target_val = target_data['target_f1']
                efficiency = target_data['efficiency_ratio']
                met = target_data['target_met']
                method = target_data['best_method']
                
                status = "✅" if met else "❌"
                report_lines.append(
                    f"- {target}: {achieved:.3f}/{target_val:.3f} = {efficiency:.1%} via {method} {status}"
                )
        
        # Transfer learning effectiveness
        report_lines.extend(["", "## Transfer Learning Effectiveness"])
        
        transfer_checks = acceptance_checks['transfer_checks']
        models_with_gain = transfer_checks['models_with_transfer_gain']
        total_models = transfer_checks['total_models_checked']
        
        report_lines.append(f"- Models with significant transfer gain: {models_with_gain}/{total_models}")
        
        for model, stats in transfer_methods.items():
            gain = stats['transfer_gain']
            gain_pass = stats['transfer_gain_pass']
            threshold = self.criteria['min_transfer_gain']
            status = "✅" if gain_pass else "❌"
            
            report_lines.append(f"- {model}: +{gain:.3f} ≥ {threshold} {status}")
        
        # Acceptance criteria summary
        report_lines.extend(["", "## Acceptance Criteria"])
        
        criteria_results = [
            ("Coverage", acceptance_checks['coverage_check']),
            ("Zero-shot baseline", zero_shot_checks.get('falling_f1_pass', False)),
            ("Label efficiency", efficiency_checks['efficiency_pass']),
            ("Transfer effectiveness", transfer_checks['transfer_pass'])
        ]
        
        for criterion, passed in criteria_results:
            status = "✅" if passed else "❌"
            report_lines.append(f"- {criterion}: {status}")
        
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
                "- Complete missing experimental configurations",
                "- Check model loading and transfer learning implementation",
                "- Verify benchmark dataset quality and preprocessing",
                "- Consider hyperparameter tuning for adaptation methods",
                ""
            ])
        else:
            report_lines.extend([
                "### Next Steps:",
                "- ✅ D4 Sim2Real validation passed",
                "- Generate label efficiency plots",
                "- Prepare results for publication",
                "- Compare with D3 cross-domain results",
                ""
            ])
        
        return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Validate D4 Sim2Real Experiments")
    parser.add_argument("--results_dir", type=str, default="results/d4/sim2real",
                       help="Sim2Real results directory")
    parser.add_argument("--output_report", type=str, default=None,
                       help="Output report file")
    
    args = parser.parse_args()
    
    if args.output_report is None:
        args.output_report = f"{args.results_dir}/d4_sim2real_validation_report.md"
    
    print("[INFO] Validating D4 Sim2Real experiments")
    print(f"[INFO] Results directory: {args.results_dir}")
    
    # Initialize validator
    validator = D4Validator()
    
    try:
        # Load results
        results = validator.load_results(args.results_dir)
        print(f"[INFO] Loaded {len(results)} experiment results")
        
        # Validate coverage
        coverage = validator.validate_coverage(results)
        
        # Validate label efficiency
        efficiency = validator.validate_label_efficiency(results)
        
        # Validate transfer methods
        transfer_methods = validator.validate_transfer_methods(results)
        
        # Check acceptance criteria
        acceptance_checks = validator.check_acceptance_criteria(
            coverage, efficiency, transfer_methods
        )
        
        # Generate report
        report = validator.generate_report(
            results, coverage, efficiency, transfer_methods, acceptance_checks
        )
        
        # Save report
        output_path = Path(args.output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Print summary
        print(f"\n[SUMMARY] D4 Sim2Real Validation Results:")
        print(f"- Experiments: {len(results)}")
        print(f"- Coverage: {'✅' if coverage['coverage_complete'] else '❌'}")
        print(f"- Performance: {'✅' if acceptance_checks['overall_pass'] else '❌'}")
        print(f"- Report saved: {args.output_report}")
        
        if acceptance_checks['overall_pass']:
            print("\n🎉 D4 Sim2Real experiments PASSED validation!")
            return 0
        else:
            print("\n❌ D4 Sim2Real experiments FAILED validation")
            print("Please check the detailed report for specific issues")
            return 1
            
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())