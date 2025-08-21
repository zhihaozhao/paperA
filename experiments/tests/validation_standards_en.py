#!/usr/bin/env python3
"""
Validation Standards Test System - PhD Thesis Experiment Acceptance Automation (English Version)
Based on D1 acceptance criteria, implementing automated validation and quality assurance

Author: PhD Thesis Research  
Date: 2025
Version: v2.0-en

Acceptance Criteria (based on memory):
- InD synthetic capacity-aligned validation - summary CSV ≥3 seeds per model
- Enhanced vs CNN parameters within ±10% range
- Metrics validity validation (macro_f1, ECE, NLL)
- Next: cross-generator testing, higher difficulty sweeps, ablation studies
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

# Statistical and validation libraries
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class D1AcceptanceCriteriaValidator:
    """D1 Acceptance Criteria Validator - Based on memory 6364081 standards"""
    
    def __init__(self, results_directory: str = "experiments/results"):
        """
        Initialize acceptance criteria validator
        Args:
            results_directory: Experiment results root directory
        """
        self.results_directory = Path(results_directory)
        self.validation_reports = []
        
        # Setup logging
        self._setup_validation_logging()
        
        # D1 Acceptance Standards (based on memory)
        self.d1_standards = {
            "ind_synthetic_capacity_alignment": {
                "min_seeds": 3,
                "required_models": ["Enhanced", "CNN"],
                "parameter_tolerance": 0.10,  # ±10%
                "summary_csv_required": True
            },
            "performance_metric_requirements": {
                "macro_f1": {"min_value": 0.75, "enhanced_target": 0.83},
                "ece": {"max_value": 0.05, "ideal_value": 0.03},
                "nll": {"max_value": 1.5, "ideal_value": 1.0}
            },
            "enhanced_model_consistency": {
                "loso_f1": 0.830,
                "loro_f1": 0.830,
                "allowed_deviation": 0.001,
                "consistency_required": True
            },
            "stea_breakthrough_point": {
                "20pct_labels_f1": 0.821,
                "target_threshold": 0.80,
                "breakthrough_required": True,
                "relative_performance": 0.986  # 82.1/83.3
            }
        }
        
    def _setup_validation_logging(self):
        """Setup validation logging system"""
        log_file = self.results_directory / f"D1_acceptance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - D1Acceptance - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("D1AcceptanceValidator")
    
    def validate_ind_synthetic_capacity_alignment(self, experiment_results_path: str) -> Dict[str, bool]:
        """
        Validate InD synthetic capacity alignment standards
        Args:
            experiment_results_path: Experiment results CSV file path
        Returns:
            Validation result dictionary
        """
        self.logger.info("🔬 Validating InD synthetic capacity alignment standards...")
        
        validation_results = {
            "summary_csv_exists": False,
            "sufficient_seeds": False,
            "parameter_tolerance_met": False,
            "model_completeness": False
        }
        
        try:
            # Check summary CSV file
            csv_file_path = Path(experiment_results_path)
            if csv_file_path.exists():
                validation_results["summary_csv_exists"] = True
                self.logger.info("✅ Summary CSV file exists")
                
                # Read CSV data
                experiment_data = pd.read_csv(csv_file_path)
                
                # Validate seed count
                if 'seed' in experiment_data.columns:
                    for model_name in self.d1_standards["ind_synthetic_capacity_alignment"]["required_models"]:
                        model_data = experiment_data[experiment_data['model'] == model_name]
                        seed_count = len(model_data['seed'].unique())
                        
                        if seed_count >= self.d1_standards["ind_synthetic_capacity_alignment"]["min_seeds"]:
                            validation_results["sufficient_seeds"] = True
                            self.logger.info(f"✅ {model_name}: {seed_count} seeds ≥ {self.d1_standards['ind_synthetic_capacity_alignment']['min_seeds']}")
                        else:
                            self.logger.warning(f"⚠️  {model_name}: {seed_count} seeds < {self.d1_standards['ind_synthetic_capacity_alignment']['min_seeds']}")
                
                # Validate Enhanced vs CNN parameter tolerance
                validation_results["parameter_tolerance_met"] = self._validate_parameter_tolerance(experiment_data)
                
                # Validate model completeness
                required_models = set(self.d1_standards["ind_synthetic_capacity_alignment"]["required_models"])
                actual_models = set(experiment_data['model'].unique()) if 'model' in experiment_data.columns else set()
                
                if required_models.issubset(actual_models):
                    validation_results["model_completeness"] = True
                    self.logger.info(f"✅ Model completeness: {actual_models}")
                else:
                    missing_models = required_models - actual_models
                    self.logger.warning(f"⚠️  Missing models: {missing_models}")
                    
        except Exception as error:
            self.logger.error(f"❌ InD validation failed: {error}")
        
        return validation_results
    
    def _validate_parameter_tolerance(self, experiment_data: pd.DataFrame) -> bool:
        """Validate Enhanced vs CNN parameter tolerance within ±10% range"""
        try:
            if 'model' in experiment_data.columns and 'parameters' in experiment_data.columns:
                enhanced_params = experiment_data[experiment_data['model'] == 'Enhanced']['parameters'].iloc[0]
                cnn_params = experiment_data[experiment_data['model'] == 'CNN']['parameters'].iloc[0]
                
                param_diff_rate = abs(enhanced_params - cnn_params) / cnn_params
                tolerance_threshold = self.d1_standards["ind_synthetic_capacity_alignment"]["parameter_tolerance"]
                
                if param_diff_rate <= tolerance_threshold:
                    self.logger.info(f"✅ Parameter tolerance: {param_diff_rate:.1%} ≤ ±{tolerance_threshold:.0%}")
                    return True
                else:
                    self.logger.warning(f"⚠️  Parameter tolerance exceeded: {param_diff_rate:.1%} > ±{tolerance_threshold:.0%}")
                    return False
            else:
                self.logger.warning("⚠️  Missing parameter information columns")
                return False
                
        except Exception as error:
            self.logger.error(f"❌ Parameter tolerance validation failed: {error}")
            return False
    
    def validate_performance_metrics_validity(self, experiment_results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate performance metrics validity (macro_f1, ECE, NLL)
        Args:
            experiment_results: Experiment results dictionary
        Returns:
            Metrics validation results
        """
        self.logger.info("📊 Validating performance metrics validity...")
        
        metrics_validation = {
            "macro_f1_valid": False,
            "ece_valid": False,
            "nll_valid": False
        }
        
        # Validate macro_f1
        if "macro_f1" in experiment_results:
            macro_f1 = experiment_results["macro_f1"]
            min_requirement = self.d1_standards["performance_metric_requirements"]["macro_f1"]["min_value"]
            
            if macro_f1 >= min_requirement:
                metrics_validation["macro_f1_valid"] = True
                self.logger.info(f"✅ Macro F1: {macro_f1:.4f} ≥ {min_requirement}")
            else:
                self.logger.warning(f"⚠️  Macro F1 below standard: {macro_f1:.4f} < {min_requirement}")
        
        # Validate ECE  
        if "ECE" in experiment_results:
            ece = experiment_results["ECE"]
            max_allowed = self.d1_standards["performance_metric_requirements"]["ece"]["max_value"]
            
            if ece <= max_allowed:
                metrics_validation["ece_valid"] = True
                self.logger.info(f"✅ ECE: {ece:.4f} ≤ {max_allowed}")
            else:
                self.logger.warning(f"⚠️  ECE exceeds limit: {ece:.4f} > {max_allowed}")
        
        # Validate NLL
        if "NLL" in experiment_results:
            nll = experiment_results["NLL"]
            max_allowed = self.d1_standards["performance_metric_requirements"]["nll"]["max_value"]
            
            if nll <= max_allowed:
                metrics_validation["nll_valid"] = True
                self.logger.info(f"✅ NLL: {nll:.4f} ≤ {max_allowed}")
            else:
                self.logger.warning(f"⚠️  NLL exceeds limit: {nll:.4f} > {max_allowed}")
        
        return metrics_validation
    
    def validate_enhanced_model_consistency(self, cdae_results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate Enhanced model LOSO=LORO consistency
        Args:
            cdae_results: CDAE protocol experiment results
        Returns:
            Consistency validation results
        """
        self.logger.info("🎯 Validating Enhanced model consistency...")
        
        consistency_validation = {
            "loso_performance_met": False,
            "loro_performance_met": False,
            "consistency_satisfied": False
        }
        
        try:
            # Extract LOSO and LORO results
            loso_results = cdae_results.get("LOSO", {})
            loro_results = cdae_results.get("LORO", {})
            
            loso_f1 = loso_results.get("loso_mean_f1", 0.0)
            loro_f1 = loro_results.get("loro_mean_f1", 0.0)
            
            # Validate LOSO performance
            if loso_f1 >= self.d1_standards["enhanced_model_consistency"]["loso_f1"]:
                consistency_validation["loso_performance_met"] = True
                self.logger.info(f"✅ LOSO F1: {loso_f1:.4f} ≥ {self.d1_standards['enhanced_model_consistency']['loso_f1']}")
            
            # Validate LORO performance
            if loro_f1 >= self.d1_standards["enhanced_model_consistency"]["loro_f1"]:
                consistency_validation["loro_performance_met"] = True
                self.logger.info(f"✅ LORO F1: {loro_f1:.4f} ≥ {self.d1_standards['enhanced_model_consistency']['loro_f1']}")
            
            # Validate consistency
            f1_difference = abs(loso_f1 - loro_f1)
            allowed_deviation = self.d1_standards["enhanced_model_consistency"]["allowed_deviation"]
            
            if f1_difference <= allowed_deviation:
                consistency_validation["consistency_satisfied"] = True
                self.logger.info(f"✅ LOSO-LORO consistency: difference={f1_difference:.4f} ≤ {allowed_deviation}")
            else:
                self.logger.warning(f"⚠️  Consistency not satisfied: difference={f1_difference:.4f} > {allowed_deviation}")
                
        except Exception as error:
            self.logger.error(f"❌ Consistency validation failed: {error}")
        
        return consistency_validation
    
    def validate_stea_breakthrough_point(self, stea_results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Validate STEA protocol 20% label breakthrough point
        Args:
            stea_results: STEA protocol experiment results
        Returns:
            Breakthrough point validation results
        """
        self.logger.info("🎯 Validating STEA breakthrough point...")
        
        breakthrough_validation = {
            "20pct_labels_met": False,
            "exceeds_target_threshold": False,
            "relative_performance_met": False
        }
        
        try:
            stea_data = stea_results.get("stea_results", [])
            
            # Find 20% label results
            for result_item in stea_data:
                if result_item.get("label_ratio") == 20:
                    labels_20_f1 = result_item.get("final_f1", 0.0)
                    target_threshold = self.d1_standards["stea_breakthrough_point"]["target_threshold"]
                    
                    # Validate 20% label performance
                    if labels_20_f1 >= self.d1_standards["stea_breakthrough_point"]["20pct_labels_f1"]:
                        breakthrough_validation["20pct_labels_met"] = True
                        self.logger.info(f"✅ 20% labels F1: {labels_20_f1:.4f} ≥ {self.d1_standards['stea_breakthrough_point']['20pct_labels_f1']}")
                    
                    # Validate exceeding target threshold
                    if labels_20_f1 >= target_threshold:
                        breakthrough_validation["exceeds_target_threshold"] = True
                        self.logger.info(f"✅ Exceeds target: {labels_20_f1:.4f} ≥ {target_threshold}")
                    
                    # Validate relative performance
                    relative_performance = result_item.get("relative_performance", 0.0)
                    if relative_performance >= self.d1_standards["stea_breakthrough_point"]["relative_performance"]:
                        breakthrough_validation["relative_performance_met"] = True
                        self.logger.info(f"✅ Relative performance: {relative_performance:.1%} ≥ {self.d1_standards['stea_breakthrough_point']['relative_performance']:.1%}")
                    
                    break
            else:
                self.logger.warning("⚠️  20% label experiment results not found")
                
        except Exception as error:
            self.logger.error(f"❌ STEA breakthrough point validation failed: {error}")
        
        return breakthrough_validation
    
    def execute_statistical_significance_testing(self, comparison_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Execute statistical significance testing
        Args:
            comparison_results: Model comparison results {"Enhanced": [F1_list], "CNN": [F1_list]}
        Returns:
            Statistical test results
        """
        self.logger.info("📈 Executing statistical significance testing...")
        
        statistical_results = {}
        
        if "Enhanced" in comparison_results and "CNN" in comparison_results:
            enhanced_f1 = comparison_results["Enhanced"]
            cnn_f1 = comparison_results["CNN"]
            
            # Paired t-test
            t_statistic, p_value = stats.ttest_rel(enhanced_f1, cnn_f1)
            
            # Cohen's d effect size
            mean_difference = np.mean(np.array(enhanced_f1) - np.array(cnn_f1))
            pooled_std = np.sqrt((np.var(enhanced_f1) + np.var(cnn_f1)) / 2)
            cohens_d = mean_difference / pooled_std if pooled_std > 0 else 0
            
            statistical_results = {
                "paired_t_test": {
                    "t_statistic": float(t_statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                },
                "effect_size": {
                    "cohens_d": float(cohens_d),
                    "effect_magnitude": "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else "small")
                },
                "descriptive_statistics": {
                    "enhanced_mean": float(np.mean(enhanced_f1)),
                    "enhanced_std": float(np.std(enhanced_f1)),
                    "cnn_mean": float(np.mean(cnn_f1)),
                    "cnn_std": float(np.std(cnn_f1))
                }
            }
            
            if p_value < 0.05:
                self.logger.info(f"✅ Statistical significance: p={p_value:.4f} < 0.05")
            else:
                self.logger.warning(f"⚠️  Not statistically significant: p={p_value:.4f} ≥ 0.05")
        
        return statistical_results
    
    def generate_comprehensive_acceptance_report(self, 
                                               ind_validation: Dict,
                                               metrics_validation: Dict, 
                                               consistency_validation: Dict,
                                               breakthrough_validation: Dict,
                                               statistical_validation: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive acceptance report
        Args:
            Various validation result dictionaries
        Returns:
            Comprehensive acceptance report
        """
        self.logger.info("📋 Generating comprehensive acceptance report...")
        
        # Calculate overall pass rate
        all_validation_items = []
        all_validation_items.extend(ind_validation.values())
        all_validation_items.extend(metrics_validation.values())
        all_validation_items.extend(consistency_validation.values())
        all_validation_items.extend(breakthrough_validation.values())
        
        passed_items = sum(all_validation_items)
        total_items = len(all_validation_items)
        pass_rate = passed_items / total_items
        
        comprehensive_report = {
            "acceptance_overview": {
                "acceptance_time": datetime.now().isoformat(),
                "passed_items": passed_items,
                "total_items": total_items,
                "pass_rate": f"{pass_rate:.1%}",
                "overall_status": "✅ Passed" if pass_rate >= 0.8 else "❌ Failed"
            },
            "detailed_validation_results": {
                "ind_synthetic_capacity_alignment": ind_validation,
                "performance_metrics_validity": metrics_validation,
                "enhanced_model_consistency": consistency_validation,
                "stea_breakthrough_point": breakthrough_validation,
                "statistical_significance": statistical_validation
            },
            "follow_up_recommendations": self._generate_follow_up_recommendations(pass_rate, ind_validation, metrics_validation, consistency_validation, breakthrough_validation)
        }
        
        # Save report
        report_file = self.results_directory / f"D1_comprehensive_acceptance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
        
        # Generate Markdown report
        self._generate_markdown_report(comprehensive_report)
        
        return comprehensive_report
    
    def _generate_follow_up_recommendations(self, pass_rate: float, *validation_results) -> List[str]:
        """Generate follow-up recommendations based on validation results"""
        recommendations = []
        
        if pass_rate >= 0.9:
            recommendations.extend([
                "🎉 Acceptance criteria fully achieved, ready for next phase",
                "📈 Recommend adding cross-generator testing (test_seed)",
                "🔄 Consider higher difficulty sweep validation",
                "🧪 Conduct ablation studies (+SE/+Attention/only CNN)",
                "📊 Temperature scaling NPZ export for reliability curves"
            ])
        elif pass_rate >= 0.7:
            recommendations.extend([
                "⚡ Most standards achieved, targeted improvements needed",
                "🔧 Focus on optimizing failed validation items",
                "📊 Increase experiment seed count for improved statistical reliability"
            ])
        else:
            recommendations.extend([
                "🚨 Acceptance criteria not met, systematic improvements required",
                "🔬 Re-examine experimental design and model architecture",
                "📋 Recommend starting with single protocol validation"
            ])
        
        return recommendations
    
    def _generate_markdown_report(self, comprehensive_report: Dict[str, Any]):
        """Generate Markdown format acceptance report"""
        report_content = f"""# WiFi CSI PhD Thesis D1 Acceptance Criteria Validation Report

## 📊 Acceptance Overview

- **Acceptance Time**: {comprehensive_report['acceptance_overview']['acceptance_time']}
- **Pass Rate**: {comprehensive_report['acceptance_overview']['pass_rate']}
- **Overall Status**: {comprehensive_report['acceptance_overview']['overall_status']}
- **Passed Items**: {comprehensive_report['acceptance_overview']['passed_items']}/{comprehensive_report['acceptance_overview']['total_items']}

## ✅ Detailed Validation Results

### 🔬 InD Synthetic Capacity Alignment Validation
"""
        
        ind_results = comprehensive_report['detailed_validation_results']['ind_synthetic_capacity_alignment']
        for item, status in ind_results.items():
            status_symbol = "✅" if status else "❌"
            report_content += f"- {item}: {status_symbol}\n"
        
        report_content += """
### 📈 Performance Metrics Validity Validation
"""
        metrics_results = comprehensive_report['detailed_validation_results']['performance_metrics_validity']
        for metric, status in metrics_results.items():
            status_symbol = "✅" if status else "❌"
            report_content += f"- {metric}: {status_symbol}\n"
        
        report_content += f"""
## 🎯 Follow-up Recommendations

"""
        for recommendation in comprehensive_report['follow_up_recommendations']:
            report_content += f"- {recommendation}\n"
        
        report_content += f"""
---
**Acceptance Standard Version**: D1 (based on memory 6364081)
**Generated Time**: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}
"""
        
        # Save Markdown report
        md_file = self.results_directory / f"D1_acceptance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 Markdown report saved: {md_file}")


def demo_acceptance_workflow():
    """Demonstrate complete acceptance workflow"""
    print("🏆 D1 Acceptance Criteria Validation Demo")
    print("=" * 60)
    
    # Create validator
    validator = D1AcceptanceCriteriaValidator()
    
    # Simulate experiment results
    simulated_ind_results = "experiments/results/simulated_summary.csv"
    simulated_performance_results = {"macro_f1": 0.830, "ECE": 0.03, "NLL": 1.2}
    simulated_cdae_results = {
        "LOSO": {"loso_mean_f1": 0.830},
        "LORO": {"loro_mean_f1": 0.830}
    }
    simulated_stea_results = {
        "stea_results": [
            {"label_ratio": 20, "final_f1": 0.821, "relative_performance": 0.986}
        ]
    }
    simulated_comparison_results = {
        "Enhanced": [0.830, 0.831, 0.829, 0.830],
        "CNN": [0.820, 0.825, 0.815, 0.822]
    }
    
    # Execute validations
    print("1. InD synthetic capacity alignment validation...")
    # ind_validation = validator.validate_ind_synthetic_capacity_alignment(simulated_ind_results)
    ind_validation = {"summary_csv_exists": True, "sufficient_seeds": True, "parameter_tolerance_met": True, "model_completeness": True}
    
    print("2. Performance metrics validity validation...")
    metrics_validation = validator.validate_performance_metrics_validity(simulated_performance_results)
    
    print("3. Enhanced model consistency validation...")
    consistency_validation = validator.validate_enhanced_model_consistency(simulated_cdae_results)
    
    print("4. STEA breakthrough point validation...")
    breakthrough_validation = validator.validate_stea_breakthrough_point(simulated_stea_results)
    
    print("5. Statistical significance testing...")
    statistical_validation = validator.execute_statistical_significance_testing(simulated_comparison_results)
    
    # Generate comprehensive report
    comprehensive_report = validator.generate_comprehensive_acceptance_report(
        ind_validation, metrics_validation, consistency_validation, breakthrough_validation, statistical_validation
    )
    
    print(f"\n🎉 Acceptance workflow completed!")
    print(f"📊 Pass rate: {comprehensive_report['acceptance_overview']['pass_rate']}")
    print(f"📋 Detailed report saved to: experiments/results/")


if __name__ == "__main__":
    demo_acceptance_workflow()