#!/usr/bin/env python3
"""
D2 Experiment Results Analysis Script
Analyzes 540 D2 experiment results and generates statistics for paper
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class D2ResultsAnalyzer:
    """Analyzer for D2 experiment results"""
    
    def __init__(self, results_dir: str = "results_gpu/d2"):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.summary_stats = {}
        
    def load_all_results(self) -> Dict:
        """Load all D2 experiment results"""
        result_files = list(self.results_dir.glob("paperA_*.json"))
        print(f"[INFO] Found {len(result_files)} D2 result files")
        
        results = {}
        failed_files = []
        
        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                # Extract experiment identifiers
                filename = file_path.stem
                # paperA_enhanced_hard_s0_cla0p0_env0p0_lab0p0
                parts = filename.split('_')
                model = parts[1]
                difficulty = parts[2]
                seed = int(parts[3][1:])  # s0 -> 0
                
                # Extract parameter values
                cla_overlap = float(parts[4][3:].replace('p', '.'))  # cla0p4 -> 0.4
                env_burst = float(parts[5][3:].replace('p', '.'))   # env0p1 -> 0.1
                lab_noise = float(parts[6][3:].replace('p', '.'))   # lab0p05 -> 0.05
                
                key = f"{model}_{difficulty}_s{seed}_cla{cla_overlap}_env{env_burst}_lab{lab_noise}"
                
                results[key] = {
                    'filename': filename,
                    'model': model,
                    'difficulty': difficulty,
                    'seed': seed,
                    'class_overlap': cla_overlap,
                    'env_burst_rate': env_burst,
                    'label_noise_prob': lab_noise,
                    'metrics': data.get('metrics', {}),
                    'meta': data.get('meta', {}),
                    'args': data.get('args', {})
                }
                
            except Exception as e:
                failed_files.append((file_path, str(e)))
        
        if failed_files:
            print(f"[WARNING] Failed to load {len(failed_files)} files:")
            for file_path, error in failed_files[:5]:  # Show first 5 errors
                print(f"  {file_path}: {error}")
        
        self.results = results
        return results
    
    def compute_summary_statistics(self) -> Dict:
        """Compute summary statistics across all experiments"""
        if not self.results:
            self.load_all_results()
        
        # Group results by different dimensions
        by_model = defaultdict(list)
        by_difficulty = defaultdict(list) 
        by_parameter_combo = defaultdict(list)
        
        all_metrics = []
        
        for key, result in self.results.items():
            model = result['model']
            difficulty = result['difficulty']
            metrics = result['metrics']
            
            # Parameter combination
            cla = result['class_overlap']
            env = result['env_burst_rate'] 
            lab = result['label_noise_prob']
            param_combo = f"cla{cla}_env{env}_lab{lab}"
            
            by_model[model].append(metrics)
            by_difficulty[difficulty].append(metrics)
            by_parameter_combo[param_combo].append(metrics)
            all_metrics.append(metrics)
        
        # Compute statistics
        summary = {
            'total_experiments': len(self.results),
            'by_model': self._compute_group_stats(by_model),
            'by_difficulty': self._compute_group_stats(by_difficulty),
            'by_parameter_combo': self._compute_group_stats(by_parameter_combo),
            'overall': self._compute_metrics_stats(all_metrics)
        }
        
        self.summary_stats = summary
        return summary
    
    def _compute_group_stats(self, grouped_results: Dict[str, List]) -> Dict:
        """Compute statistics for grouped results"""
        stats = {}
        for group_name, metrics_list in grouped_results.items():
            stats[group_name] = self._compute_metrics_stats(metrics_list)
        return stats
    
    def _compute_metrics_stats(self, metrics_list: List[Dict]) -> Dict:
        """Compute statistics for a list of metrics"""
        if not metrics_list:
            return {}
        
        # Extract metric values
        metric_arrays = {}
        for metrics in metrics_list:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if metric_name not in metric_arrays:
                        metric_arrays[metric_name] = []
                    metric_arrays[metric_name].append(value)
        
        # Compute statistics
        stats = {}
        for metric_name, values in metric_arrays.items():
            if values:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        return stats
    
    def extract_paper_statistics(self) -> Dict:
        """Extract key statistics for paper writing"""
        if not self.summary_stats:
            self.compute_summary_statistics()
        
        stats = self.summary_stats
        
        # Key paper statistics
        paper_stats = {
            'total_experiments': stats['total_experiments'],
            'models_evaluated': len(stats['by_model']),
            'parameter_combinations': len(stats['by_parameter_combo']),
            
            # Overall performance
            'overall_macro_f1': stats['overall'].get('macro_f1', {}),
            'overall_falling_f1': stats['overall'].get('falling_f1', {}),
            'overall_ece_raw': stats['overall'].get('ece_raw', {}),
            'overall_ece_cal': stats['overall'].get('ece_cal', {}),
            
            # Model comparison
            'enhanced_performance': stats['by_model'].get('enhanced', {}).get('macro_f1', {}),
            'cnn_performance': stats['by_model'].get('cnn', {}).get('macro_f1', {}),
            'bilstm_performance': stats['by_model'].get('bilstm', {}).get('macro_f1', {}),
            'conformer_performance': stats['by_model'].get('conformer_lite', {}).get('macro_f1', {}),
            
            # Calibration improvement
            'calibration_improvement': self._compute_calibration_improvement(),
            
            # Robustness analysis
            'robustness_analysis': self._analyze_robustness()
        }
        
        return paper_stats
    
    def _compute_calibration_improvement(self) -> Dict:
        """Compute calibration improvement statistics"""
        improvements = []
        
        for result in self.results.values():
            metrics = result['metrics']
            ece_raw = metrics.get('ece_raw', 0)
            ece_cal = metrics.get('ece_cal', 0)
            
            if ece_raw > 0:
                improvement = (ece_raw - ece_cal) / ece_raw
                improvements.append(improvement)
        
        if improvements:
            return {
                'mean_improvement': float(np.mean(improvements)),
                'std_improvement': float(np.std(improvements)),
                'median_improvement': float(np.median(improvements)),
                'n_experiments': len(improvements)
            }
        return {}
    
    def _analyze_robustness(self) -> Dict:
        """Analyze model robustness across parameter variations"""
        # Group by model and compute performance variance
        robustness = {}
        
        for model in ['enhanced', 'cnn', 'bilstm', 'conformer_lite']:
            model_results = [r for r in self.results.values() if r['model'] == model]
            
            if model_results:
                falling_f1s = [r['metrics'].get('falling_f1', 0) for r in model_results]
                macro_f1s = [r['metrics'].get('macro_f1', 0) for r in model_results]
                
                robustness[model] = {
                    'falling_f1_variance': float(np.var(falling_f1s)),
                    'macro_f1_variance': float(np.var(macro_f1s)),
                    'falling_f1_cv': float(np.std(falling_f1s) / np.mean(falling_f1s)) if np.mean(falling_f1s) > 0 else 0,
                    'n_experiments': len(model_results)
                }
        
        return robustness
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table for paper"""
        if not self.summary_stats:
            self.compute_summary_statistics()
        
        # Create model comparison table
        models = ['enhanced', 'cnn', 'bilstm', 'conformer_lite']
        metrics = ['macro_f1', 'falling_f1', 'ece_raw', 'ece_cal']
        
        table_data = []
        for model in models:
            row = {'Model': model.capitalize()}
            model_stats = self.summary_stats['by_model'].get(model, {})
            
            for metric in metrics:
                metric_stats = model_stats.get(metric, {})
                if metric_stats:
                    mean_val = metric_stats.get('mean', 0)
                    std_val = metric_stats.get('std', 0)
                    row[metric] = f"{mean_val:.3f}±{std_val:.3f}"
                else:
                    row[metric] = "N/A"
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def save_summary_csv(self, output_path: str = "results/d2_summary_analysis.csv"):
        """Save summary statistics to CSV"""
        summary_table = self.create_summary_table()
        summary_table.to_csv(output_path, index=False)
        print(f"[INFO] Summary table saved to {output_path}")
        
        # Also save detailed statistics
        paper_stats = self.extract_paper_statistics()
        stats_file = output_path.replace('.csv', '_detailed.json')
        
        with open(stats_file, 'w') as f:
            json.dump(paper_stats, f, indent=2)
        print(f"[INFO] Detailed statistics saved to {stats_file}")
        
        return summary_table

def main():
    """Main analysis function"""
    print("[INFO] Analyzing D2 experiment results...")
    
    # Initialize analyzer
    analyzer = D2ResultsAnalyzer()
    
    # Load and analyze results
    results = analyzer.load_all_results()
    print(f"[INFO] Loaded {len(results)} experiments successfully")
    
    # Compute statistics
    summary_stats = analyzer.compute_summary_statistics()
    
    # Extract paper statistics
    paper_stats = analyzer.extract_paper_statistics()
    
    # Create and save summary
    summary_table = analyzer.save_summary_csv()
    
    # Print key findings
    print("\n" + "="*60)
    print("D2 EXPERIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total experiments: {paper_stats['total_experiments']}")
    print(f"Models evaluated: {paper_stats['models_evaluated']}")
    print(f"Parameter combinations: {paper_stats['parameter_combinations']}")
    
    print("\nOverall Performance:")
    overall_macro = paper_stats['overall_macro_f1']
    overall_falling = paper_stats['overall_falling_f1']
    print(f"  Macro F1: {overall_macro.get('mean', 0):.3f}±{overall_macro.get('std', 0):.3f}")
    print(f"  Falling F1: {overall_falling.get('mean', 0):.3f}±{overall_falling.get('std', 0):.3f}")
    
    print("\nCalibration Improvement:")
    cal_imp = paper_stats['calibration_improvement']
    if cal_imp:
        print(f"  Mean ECE reduction: {cal_imp.get('mean_improvement', 0):.1%}")
        print(f"  Median ECE reduction: {cal_imp.get('median_improvement', 0):.1%}")
    
    print("\nModel Performance Summary:")
    print(summary_table.to_string(index=False))
    
    print("\n[SUCCESS] D2 analysis completed! Results ready for paper integration.")

if __name__ == "__main__":
    main()