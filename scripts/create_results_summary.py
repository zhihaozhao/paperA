#!/usr/bin/env python3
"""
D2实验结果摘要生成器
快速生成Markdown格式的结果摘要
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results_path = Path(results_dir)
    all_results = []
    
    # 查找所有CSV结果文件
    csv_files = list(results_path.glob("**/*.csv"))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return None

def create_markdown_summary(df, output_path):
    """创建Markdown摘要"""
    
    md_content = f"""# 🔬 D2实验结果摘要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 实验概览

- **总实验数**: {len(df)} 次
- **模型类型**: 4 种 (enhanced, cnn, bilstm, conformer_lite)
- **随机种子**: 5 个 (0-4)
- **参数网格**: 3×3×3 = 27 种配置
- **总配置数**: 4 × 5 × 27 = **540 次实验**

## 🏆 模型性能排名

### Macro F1 Score
"""
    
    if 'macro_f1' in df.columns and 'model' in df.columns:
        # 计算各模型的平均性能
        model_performance = df.groupby('model')['macro_f1'].agg(['mean', 'std', 'count']).round(4)
        model_performance = model_performance.sort_values('mean', ascending=False)
        
        md_content += """
| 排名 | 模型 | Macro F1 (均值±标准差) | 实验次数 |
|------|------|------------------------|----------|
"""
        
        for i, (model, stats) in enumerate(model_performance.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            md_content += f"| {i} {medal} | `{model}` | {stats['mean']:.4f} ± {stats['std']:.4f} | {int(stats['count'])} |\n"
    
    # ECE排名
    if 'ece' in df.columns:
        md_content += "\n### Expected Calibration Error (ECE)\n"
        ece_performance = df.groupby('model')['ece'].agg(['mean', 'std']).round(4)
        ece_performance = ece_performance.sort_values('mean', ascending=True)  # ECE越小越好
        
        md_content += """
| 排名 | 模型 | ECE (均值±标准差) |
|------|------|------------------|
"""
        
        for i, (model, stats) in enumerate(ece_performance.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            md_content += f"| {i} {medal} | `{model}` | {stats['mean']:.4f} ± {stats['std']:.4f} |\n"
    
    # 超参数影响分析
    md_content += "\n## ⚙️ 超参数影响分析\n"
    
    params = ['class_overlap', 'label_noise_prob', 'env_burst_rate']
    for param in params:
        if param in df.columns:
            param_effect = df.groupby(param)['macro_f1'].agg(['mean', 'std']).round(4)
            
            param_name_map = {
                'class_overlap': 'Class Overlap',
                'label_noise_prob': 'Label Noise Probability', 
                'env_burst_rate': 'Environment Burst Rate'
            }
            
            md_content += f"\n### {param_name_map.get(param, param)}\n"
            md_content += """
| 参数值 | Macro F1 (均值±标准差) |
|--------|----------------------|
"""
            
            for value, stats in param_effect.iterrows():
                md_content += f"| {value} | {stats['mean']:.4f} ± {stats['std']:.4f} |\n"
    
    # 验收标准检查
    md_content += "\n## ✅ D2验收标准检查\n"
    
    # 检查实验完成度
    expected_experiments = 540
    actual_experiments = len(df)
    completion_rate = (actual_experiments / expected_experiments) * 100
    
    status_icon = "✅" if completion_rate >= 95 else "⚠️" if completion_rate >= 80 else "❌"
    
    md_content += f"""
### 实验完成度
- **预期实验数**: {expected_experiments}
- **实际完成数**: {actual_experiments}
- **完成率**: {completion_rate:.1f}% {status_icon}

### 数据质量检查
"""
    
    # 检查是否所有模型都有结果
    if 'model' in df.columns:
        expected_models = {'enhanced', 'cnn', 'bilstm', 'conformer_lite'}
        actual_models = set(df['model'].unique())
        missing_models = expected_models - actual_models
        
        if not missing_models:
            md_content += "- **模型覆盖**: ✅ 所有4个模型都有结果\n"
        else:
            md_content += f"- **模型覆盖**: ⚠️ 缺失模型: {missing_models}\n"
    
    # 检查种子覆盖
    if 'seed' in df.columns:
        expected_seeds = {0, 1, 2, 3, 4}
        actual_seeds = set(df['seed'].unique())
        missing_seeds = expected_seeds - actual_seeds
        
        if not missing_seeds:
            md_content += "- **种子覆盖**: ✅ 所有5个种子都有结果\n"
        else:
            md_content += f"- **种子覆盖**: ⚠️ 缺失种子: {missing_seeds}\n"
    
    # 性能稳定性检查
    if 'macro_f1' in df.columns and 'model' in df.columns:
        md_content += "\n### 性能稳定性\n"
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]['macro_f1']
            cv = (model_data.std() / model_data.mean()) * 100  # 变异系数
            
            stability_icon = "✅" if cv < 10 else "⚠️" if cv < 20 else "❌"
            md_content += f"- **{model}**: CV = {cv:.2f}% {stability_icon}\n"
    
    # 后续计划
    md_content += """
## 🚀 下一步计划

### 立即任务
- [ ] 创建D2完成里程碑标签 (`v1.0-d2-complete`)
- [ ] 准备Sim2Real实验数据
- [ ] 设置SenseFi benchmark环境

### Sim2Real实验计划
- [ ] **基线建立**: 在SenseFi数据集上训练传统模型
- [ ] **域转移测试**: 合成数据训练 → 真实数据测试
- [ ] **少样本学习**: 用少量真实数据微调
- [ ] **跨域泛化**: 不同数据集间的性能评估

### 论文写作
- [ ] 更新实验结果到`main.tex`
- [ ] 生成性能对比图表
- [ ] 完善讨论部分
- [ ] 准备投稿到TMC/IoTJ期刊

## 📋 文件位置

- **详细分析报告**: `reports/d2_analysis.html`
- **原始结果数据**: `results/`
- **验收脚本**: `scripts/validate_d2_acceptance.py`
- **Git管理指南**: `docs/Git_Management_Commands.md`
- **项目清单**: `PROJECT_MANIFEST.md`

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Markdown摘要已生成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成D2实验结果摘要')
    parser.add_argument('results_dir', help='结果目录路径')
    parser.add_argument('--format', choices=['markdown'], default='markdown', help='输出格式')
    parser.add_argument('--output', default='D2_Results_Summary.md', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 加载结果
    print("加载实验结果...")
    df = load_experiment_results(args.results_dir)
    
    if df is None or df.empty:
        print("错误: 未找到有效的实验结果")
        return
    
    print(f"成功加载 {len(df)} 条实验记录")
    
    # 生成摘要
    if args.format == 'markdown':
        create_markdown_summary(df, args.output)
    
    print("✅ D2结果摘要生成完成！")

if __name__ == "__main__":
    main()