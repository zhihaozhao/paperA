#!/usr/bin/env python3
"""
D2实验验收标准自动化检查脚本
用于验证D2实验结果是否满足所有验收标准
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os
from scipy import stats
from typing import Dict, List, Tuple


class D2AcceptanceValidator:
    """D2实验验收标准验证器"""
    
    def __init__(self, results_dir: str = "results_gpu/d2"):
        self.results_dir = Path(results_dir)
        self.report = []
        self.passed_checks = 0
        self.total_checks = 0
        
    def log(self, message: str, level: str = "INFO"):
        """记录验证信息"""
        prefix = {
            "INFO": "ℹ️",
            "PASS": "✅",
            "FAIL": "❌", 
            "WARN": "⚠️"
        }.get(level, "•")
        
        formatted = f"{prefix} {message}"
        self.report.append(formatted)
        print(formatted)
        
    def check(self, condition: bool, success_msg: str, failure_msg: str):
        """执行单项验收检查"""
        self.total_checks += 1
        if condition:
            self.passed_checks += 1
            self.log(success_msg, "PASS")
            return True
        else:
            self.log(failure_msg, "FAIL")
            return False
    
    def load_experiment_results(self) -> pd.DataFrame:
        """加载所有实验结果"""
        self.log("Loading experiment results...")
        
        json_files = list(self.results_dir.glob("*.json"))
        if not json_files:
            self.log(f"No JSON files found in {self.results_dir}", "FAIL")
            return pd.DataFrame()
        
        results = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取关键信息
                args = data.get('args', {})
                metrics = data.get('metrics', {})
                
                result = {
                    'file': json_file.name,
                    'model': args.get('model', 'unknown'),
                    'difficulty': args.get('difficulty', 'unknown'),
                    'seed': args.get('seed', 0),
                    'macro_f1': metrics.get('macro_f1', 0.0),
                    'falling_f1': metrics.get('falling_f1', 0.0),
                    'mutual_misclass': metrics.get('mutual_misclass', 0.0),
                    'ece': metrics.get('ece', 1.0),
                    'ece_raw': metrics.get('ece_raw', 1.0),
                    'ece_cal': metrics.get('ece_cal', 1.0),
                    'brier': metrics.get('brier', 1.0),
                    'temperature': metrics.get('temperature', 1.0),
                }
                results.append(result)
                
            except Exception as e:
                self.log(f"Error loading {json_file}: {e}", "WARN")
        
        df = pd.DataFrame(results)
        self.log(f"Loaded {len(df)} experiment results")
        return df
    
    def validate_basic_completeness(self, df: pd.DataFrame) -> bool:
        """验证基本完整性"""
        self.log("\n=== 基本完整性检查 ===")
        
        # 检查实验数量
        expected_experiments = 4 * 8 * 3  # 4模型 × 8种子 × 3难度
        actual_experiments = len(df)
        
        completeness_ok = self.check(
            actual_experiments >= expected_experiments * 0.9,  # 允许10%的失败率
            f"实验完整性: {actual_experiments}/{expected_experiments} (≥90%)",
            f"实验完整性不足: {actual_experiments}/{expected_experiments} (<90%)"
        )
        
        # 检查模型覆盖
        expected_models = {'enhanced', 'lstm', 'tcn', 'txf'}
        actual_models = set(df['model'].unique())
        models_ok = self.check(
            expected_models.issubset(actual_models),
            f"模型覆盖完整: {sorted(actual_models)}",
            f"模型覆盖不足: 期望{sorted(expected_models)}, 实际{sorted(actual_models)}"
        )
        
        # 检查难度级别覆盖
        expected_difficulties = {'easy', 'mid', 'hard'}
        actual_difficulties = set(df['difficulty'].unique())
        difficulties_ok = self.check(
            expected_difficulties.issubset(actual_difficulties),
            f"难度级别完整: {sorted(actual_difficulties)}",
            f"难度级别不足: 期望{sorted(expected_difficulties)}, 实际{sorted(actual_difficulties)}"
        )
        
        return completeness_ok and models_ok and difficulties_ok
    
    def validate_performance_metrics(self, df: pd.DataFrame) -> bool:
        """验证性能指标"""
        self.log("\n=== 性能指标验证 ===")
        
        # 1. Falling F1 < 0.99 (严格标准)
        falling_f1_max = df['falling_f1'].max()
        falling_f1_ok = self.check(
            falling_f1_max < 0.99,
            f"Falling F1未达完美: 最大值={falling_f1_max:.4f} < 0.99",
            f"Falling F1过于完美: 最大值={falling_f1_max:.4f} ≥ 0.99"
        )
        
        # 2. Mutual Misclass > 0 (必须有混淆)
        mutual_misclass_min = df['mutual_misclass'].min()
        mutual_misclass_ok = self.check(
            mutual_misclass_min > 0,
            f"存在类间混淆: 最小值={mutual_misclass_min:.6f} > 0",
            f"无类间混淆: 最小值={mutual_misclass_min:.6f} ≤ 0"
        )
        
        # 3. Macro F1 在合理范围 (0.80-0.98)
        macro_f1_range = (df['macro_f1'].min(), df['macro_f1'].max())
        macro_f1_ok = self.check(
            0.80 <= macro_f1_range[0] and macro_f1_range[1] <= 0.98,
            f"Macro F1范围合理: {macro_f1_range[0]:.3f}-{macro_f1_range[1]:.3f}",
            f"Macro F1范围异常: {macro_f1_range[0]:.3f}-{macro_f1_range[1]:.3f}"
        )
        
        return falling_f1_ok and mutual_misclass_ok and macro_f1_ok
    
    def validate_calibration_metrics(self, df: pd.DataFrame) -> bool:
        """验证校准指标"""
        self.log("\n=== 校准指标验证 ===")
        
        # 1. Enhanced模型ECE改善
        enhanced_df = df[df['model'] == 'enhanced']
        if len(enhanced_df) == 0:
            self.log("未找到Enhanced模型结果", "WARN")
            return False
        
        # 温度校准改善
        ece_improvement = (enhanced_df['ece_raw'] - enhanced_df['ece_cal']).mean()
        calibration_ok = self.check(
            ece_improvement > 0,
            f"温度校准平均改善: {ece_improvement:.4f} > 0",
            f"温度校准无改善: {ece_improvement:.4f} ≤ 0"
        )
        
        # 2. Enhanced vs 基线ECE对比
        baseline_models = ['lstm', 'tcn', 'txf']
        baseline_df = df[df['model'].isin(baseline_models)]
        
        if len(baseline_df) > 0:
            enhanced_ece_mean = enhanced_df['ece'].mean()
            baseline_ece_mean = baseline_df['ece'].mean()
            ece_advantage = baseline_ece_mean - enhanced_ece_mean
            
            ece_comparison_ok = self.check(
                ece_advantage > 0.01,  # Enhanced应该比基线低至少1%
                f"Enhanced ECE优势: {ece_advantage:.4f} > 0.01",
                f"Enhanced ECE优势不足: {ece_advantage:.4f} ≤ 0.01"
            )
        else:
            ece_comparison_ok = False
            self.log("缺少基线模型结果进行ECE对比", "WARN")
        
        return calibration_ok and ece_comparison_ok
    
    def validate_statistical_stability(self, df: pd.DataFrame) -> bool:
        """验证统计稳定性"""
        self.log("\n=== 统计稳定性验证 ===")
        
        # 每个(模型,难度)组合的种子数
        seed_counts = df.groupby(['model', 'difficulty']).size()
        min_seeds = seed_counts.min()
        
        seed_coverage_ok = self.check(
            min_seeds >= 3,
            f"种子覆盖充分: 最少{min_seeds}个种子/组合 ≥ 3",
            f"种子覆盖不足: 最少{min_seeds}个种子/组合 < 3"
        )
        
        # 检查变异系数 (CV = std/mean)
        stability_results = []
        for (model, difficulty), group in df.groupby(['model', 'difficulty']):
            if len(group) >= 3:
                cv_f1 = group['macro_f1'].std() / group['macro_f1'].mean()
                cv_ece = group['ece'].std() / group['ece'].mean()
                stability_results.extend([cv_f1, cv_ece])
        
        if stability_results:
            max_cv = max(stability_results)
            stability_ok = self.check(
                max_cv < 0.15,
                f"结果稳定性良好: 最大变异系数={max_cv:.3f} < 0.15",
                f"结果稳定性不足: 最大变异系数={max_cv:.3f} ≥ 0.15"
            )
        else:
            stability_ok = False
            self.log("无足够数据进行稳定性分析", "WARN")
        
        return seed_coverage_ok and stability_ok
    
    def validate_output_files(self) -> bool:
        """验证输出文件"""
        self.log("\n=== 输出文件验证 ===")
        
        # 检查必需的汇总文件
        summary_csv = Path("results/synth/summary.csv")
        summary_ok = self.check(
            summary_csv.exists(),
            f"汇总CSV文件存在: {summary_csv}",
            f"汇总CSV文件缺失: {summary_csv}"
        )
        
        # 检查图表文件
        plots_dir = Path("plots")
        expected_plots = [
            "fig_synth_bars.pdf",
            "fig_overlap_scatter.pdf"
        ]
        
        plots_ok = True
        for plot_file in expected_plots:
            plot_path = plots_dir / plot_file
            file_ok = self.check(
                plot_path.exists(),
                f"图表文件存在: {plot_path}",
                f"图表文件缺失: {plot_path}"
            )
            plots_ok = plots_ok and file_ok
        
        return summary_ok and plots_ok
    
    def generate_summary_statistics(self, df: pd.DataFrame):
        """生成汇总统计"""
        self.log("\n=== 实验结果摘要 ===")
        
        # 按模型统计
        model_stats = df.groupby('model').agg({
            'macro_f1': ['mean', 'std'],
            'falling_f1': ['mean', 'std'], 
            'ece': ['mean', 'std']
        }).round(4)
        
        self.log("按模型汇总统计:")
        for model in model_stats.index:
            f1_mean, f1_std = model_stats.loc[model, ('macro_f1', 'mean')], model_stats.loc[model, ('macro_f1', 'std')]
            ece_mean, ece_std = model_stats.loc[model, ('ece', 'mean')], model_stats.loc[model, ('ece', 'std')]
            self.log(f"  {model}: F1={f1_mean:.3f}±{f1_std:.3f}, ECE={ece_mean:.3f}±{ece_std:.3f}")
        
        # 按难度统计
        difficulty_stats = df.groupby('difficulty').agg({
            'macro_f1': 'mean',
            'falling_f1': 'mean',
            'mutual_misclass': 'mean'
        }).round(4)
        
        self.log("按难度汇总统计:")
        for difficulty in difficulty_stats.index:
            stats_row = difficulty_stats.loc[difficulty]
            self.log(f"  {difficulty}: F1={stats_row['macro_f1']:.3f}, "
                    f"Fall F1={stats_row['falling_f1']:.3f}, "
                    f"Misclass={stats_row['mutual_misclass']:.4f}")
    
    def run_full_validation(self) -> bool:
        """运行完整的验收验证"""
        self.log("🔍 开始D2实验验收标准检查")
        self.log(f"结果目录: {self.results_dir}")
        
        # 加载数据
        df = self.load_experiment_results()
        if df.empty:
            self.log("无法加载实验结果，验证失败", "FAIL")
            return False
        
        # 执行各项验证
        completeness_ok = self.validate_basic_completeness(df)
        performance_ok = self.validate_performance_metrics(df)
        calibration_ok = self.validate_calibration_metrics(df)
        stability_ok = self.validate_statistical_stability(df)
        output_ok = self.validate_output_files()
        
        # 生成摘要
        self.generate_summary_statistics(df)
        
        # 最终结果
        all_passed = all([completeness_ok, performance_ok, calibration_ok, stability_ok, output_ok])
        
        self.log(f"\n{'='*50}")
        self.log(f"验收结果: {self.passed_checks}/{self.total_checks} 项检查通过")
        
        if all_passed:
            self.log("🎉 D2实验验收通过！所有标准均已达成", "PASS")
        else:
            self.log("❌ D2实验验收失败，请检查上述失败项", "FAIL")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="D2实验验收标准检查")
    parser.add_argument("--results-dir", type=str, default="results_gpu/d2",
                       help="实验结果目录路径")
    parser.add_argument("--save-report", type=str, 
                       help="保存验收报告到指定文件")
    
    args = parser.parse_args()
    
    # 创建验证器并运行
    validator = D2AcceptanceValidator(args.results_dir)
    success = validator.run_full_validation()
    
    # 保存报告
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# D2实验验收报告\n\n")
            f.write(f"**生成时间**: {pd.Timestamp.now()}\n")
            f.write(f"**结果目录**: {args.results_dir}\n")
            f.write(f"**验收状态**: {'✅ 通过' if success else '❌ 失败'}\n\n")
            f.write("## 详细检查结果\n\n```\n")
            for line in validator.report:
                f.write(line + "\n")
            f.write("```\n")
        
        print(f"\n📄 验收报告已保存到: {report_path}")
    
    # 设置退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()