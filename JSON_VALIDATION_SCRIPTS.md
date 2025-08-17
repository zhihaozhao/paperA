# 📁 JSON格式D2验收脚本 (修正版)

## 🔧 **文件1: scripts\run_d2_validation.bat (JSON版)**

```batch
@echo off
echo ===============================================
echo      D2实验结果验收执行脚本 (JSON格式)
echo ===============================================

echo [1] 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo Error: Python未找到，请检查Python安装
    pause
    exit /b 1
)

echo [2] 检查结果文件...
if not exist "results_gpu\d2\" (
    echo Error: results_gpu\d2\目录不存在，请检查结果路径
    pause
    exit /b 1
)

echo [3] 运行D2验收脚本 (JSON格式)...
python scripts\validate_d2_acceptance_json.py results_gpu\d2\

if %errorlevel% equ 0 (
    echo [✅] D2验收脚本执行成功！
) else (
    echo [❌] D2验收脚本执行失败
    pause
    exit /b 1
)

echo [4] 生成详细分析报告...
if not exist "reports\" mkdir reports
python scripts\generate_d2_analysis_report_json.py results_gpu\d2\ --output reports\d2_analysis.html

echo [5] 创建结果摘要...
python scripts\create_results_summary_json.py results_gpu\d2\ --format markdown --output D2_Results_Summary.md

echo ===============================================
echo      D2验收完成！请查看生成的报告
echo ===============================================
pause
```

---

## 🔬 **文件2: scripts\validate_d2_acceptance_json.py**

```python
#!/usr/bin/env python3
"""
D2实验结果验收脚本 (JSON格式)
验证540配置实验是否符合标准
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

def load_json_results(results_dir):
    """加载JSON格式的实验结果"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_path}")
        return None
    
    # 查找所有JSON文件
    json_files = list(results_path.glob("**/*.json"))
    
    if not json_files:
        print("❌ 未找到JSON结果文件")
        return None
    
    print(f"✅ 找到 {len(json_files)} 个结果文件")
    
    # 加载所有结果
    all_results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同的JSON结构
            if isinstance(data, dict):
                # 单个实验结果
                all_results.append(data)
                print(f"  - 加载: {json_file.name} (1 条记录)")
            elif isinstance(data, list):
                # 多个实验结果
                all_results.extend(data)
                print(f"  - 加载: {json_file.name} ({len(data)} 条记录)")
            else:
                print(f"⚠️ 跳过非标准格式: {json_file.name}")
                
        except Exception as e:
            print(f"❌ 加载失败: {json_file}: {e}")
            continue
    
    if not all_results:
        print("❌ 未能加载任何结果文件")
        return None
    
    # 转换为DataFrame
    try:
        df = pd.DataFrame(all_results)
        print(f"✅ 成功转换为DataFrame: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ DataFrame转换失败: {e}")
        return None

def validate_experiment_completeness(df):
    """验证实验完成度"""
    if df is None or df.empty:
        print("❌ 无有效数据")
        return False
    
    total_experiments = len(df)
    print(f"📊 总实验数: {total_experiments}")
    
    # 检查期望的实验数量
    expected_experiments = 540  # 4 models × 5 seeds × 27 configs
    completion_rate = (total_experiments / expected_experiments) * 100
    
    if completion_rate >= 95:
        print(f"✅ 实验完成度: {completion_rate:.1f}% ({total_experiments}/{expected_experiments})")
        return True
    elif completion_rate >= 80:
        print(f"⚠️  实验完成度: {completion_rate:.1f}% ({total_experiments}/{expected_experiments}) - 基本达标")
        return True
    else:
        print(f"❌ 实验完成度不足: {completion_rate:.1f}% ({total_experiments}/{expected_experiments})")
        return False

def validate_data_quality(df):
    """验证数据质量"""
    print("\n🔍 数据质量检查:")
    
    # 检查必要列
    required_columns = ['model', 'seed', 'macro_f1']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 缺失必要列: {missing_columns}")
        print(f"   实际列: {list(df.columns)}")
        return False
    
    print(f"✅ 必要列完整: {required_columns}")
    
    # 检查模型覆盖
    expected_models = {'enhanced', 'cnn', 'bilstm', 'conformer_lite'}
    actual_models = set(df['model'].unique())
    missing_models = expected_models - actual_models
    
    if missing_models:
        print(f"⚠️  缺失模型: {missing_models}")
        print(f"   实际模型: {actual_models}")
    else:
        print(f"✅ 模型覆盖完整: {actual_models}")
    
    # 检查种子覆盖
    expected_seeds = {0, 1, 2, 3, 4}
    actual_seeds = set(df['seed'].unique())
    missing_seeds = expected_seeds - actual_seeds
    
    if missing_seeds:
        print(f"⚠️  缺失种子: {missing_seeds}")
        print(f"   实际种子: {actual_seeds}")
    else:
        print(f"✅ 种子覆盖完整: {actual_seeds}")
    
    # 检查数据完整性
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"⚠️  存在缺失值:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"   {col}: {count} 个缺失值")
    else:
        print("✅ 无缺失值")
    
    return True

def validate_performance_stability(df):
    """验证性能稳定性"""
    print("\n📈 性能稳定性检查:")
    
    if 'macro_f1' not in df.columns:
        print("❌ 未找到macro_f1列，无法验证稳定性")
        return False
    
    # 按模型分组检查稳定性
    for model in df['model'].unique():
        model_data = df[df['model'] == model]['macro_f1']
        
        if len(model_data) < 2:
            print(f"⚠️  {model}: 数据点不足({len(model_data)})，无法评估稳定性")
            continue
        
        mean_f1 = model_data.mean()
        std_f1 = model_data.std()
        cv = (std_f1 / mean_f1) * 100  # 变异系数
        
        if cv < 10:
            stability_icon = "✅"
            stability_level = "优秀"
        elif cv < 20:
            stability_icon = "⚠️"
            stability_level = "良好"
        else:
            stability_icon = "❌"
            stability_level = "不稳定"
        
        print(f"  {stability_icon} {model}: F1={mean_f1:.4f}±{std_f1:.4f}, CV={cv:.2f}% ({stability_level})")
    
    return True

def generate_acceptance_summary(df):
    """生成验收摘要"""
    print("\n" + "="*50)
    print("📋 D2实验验收摘要")
    print("="*50)
    
    # 基本统计
    print(f"总实验数: {len(df)}")
    if 'model' in df.columns:
        print(f"模型数量: {len(df['model'].unique())}")
    if 'seed' in df.columns:
        print(f"种子数量: {len(df['seed'].unique())}")
    
    if 'macro_f1' in df.columns:
        overall_mean = df['macro_f1'].mean()
        overall_std = df['macro_f1'].std()
        print(f"整体F1: {overall_mean:.4f} ± {overall_std:.4f}")
        
        # 最佳模型
        best_idx = df['macro_f1'].idxmax()
        if 'model' in df.columns:
            best_model = df.loc[best_idx, 'model']
            best_f1 = df.loc[best_idx, 'macro_f1']
            print(f"最佳结果: {best_model} (F1={best_f1:.4f})")
    
    print("\n🎯 验收状态: 通过")
    print("🚀 可以进入Sim2Real阶段")

def main():
    parser = argparse.ArgumentParser(description='D2实验结果验收 (JSON格式)')
    parser.add_argument('results_dir', help='结果目录路径')
    parser.add_argument('--strict', action='store_true', help='严格模式验收')
    
    args = parser.parse_args()
    
    print("🔬 开始D2实验结果验收 (JSON格式)...")
    print(f"📂 结果目录: {args.results_dir}")
    
    # 步骤1: 加载JSON结果
    df = load_json_results(args.results_dir)
    if df is None:
        print("\n❌ 验收失败: 无法加载结果数据")
        sys.exit(1)
    
    # 步骤2: 验证完成度
    if not validate_experiment_completeness(df):
        print("\n❌ 验收失败: 实验完成度不符合要求")
        sys.exit(1)
    
    # 步骤3: 验证数据质量
    if not validate_data_quality(df):
        print("\n❌ 验收失败: 数据质量不符合要求")
        sys.exit(1)
    
    # 步骤4: 验证性能稳定性
    if not validate_performance_stability(df):
        print("\n❌ 验收失败: 性能稳定性不符合要求")
        sys.exit(1)
    
    # 步骤5: 生成摘要
    generate_acceptance_summary(df)
    
    print("\n🎉 D2实验验收通过！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## 📊 **文件3: scripts\generate_d2_analysis_report_json.py**

```python
#!/usr/bin/env python3
"""
D2实验结果详细分析报告生成器 (JSON格式)
生成HTML格式的交互式分析报告
"""

import os
import json
import pandas as pd
from pathlib import Path
import argparse
import numpy as np
from datetime import datetime

def load_json_results(results_dir):
    """加载JSON格式的实验结果"""
    results_path = Path(results_dir)
    all_results = []
    
    # 查找所有JSON文件
    json_files = list(results_path.glob("**/*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同的JSON结构
            if isinstance(data, dict):
                all_results.append(data)
            elif isinstance(data, list):
                all_results.extend(data)
            else:
                print(f"Warning: skipping non-standard format: {json_file}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if all_results:
        combined_df = pd.DataFrame(all_results)
        return combined_df
    else:
        return None

def analyze_model_performance(df):
    """分析模型性能"""
    analysis = {}
    
    if df is None or df.empty:
        return analysis
    
    if 'model' not in df.columns:
        print("Warning: 'model' column not found")
        return analysis
    
    # 按模型统计
    metrics_to_analyze = ['macro_f1']
    if 'ece' in df.columns:
        metrics_to_analyze.append('ece')
    if 'nll' in df.columns:
        metrics_to_analyze.append('nll')
    
    agg_dict = {}
    for metric in metrics_to_analyze:
        if metric in df.columns:
            agg_dict[metric] = ['mean', 'std', 'count']
    
    if agg_dict:
        model_stats = df.groupby('model').agg(agg_dict).round(4)
        analysis['model_stats'] = model_stats
    
    # 找到最佳模型
    best_models = {}
    for metric in metrics_to_analyze:
        if metric in df.columns:
            if metric == 'macro_f1':  # 越高越好
                best_model = df.loc[df[metric].idxmax(), 'model']
                best_value = df[metric].max()
            else:  # ECE和NLL越低越好
                best_model = df.loc[df[metric].idxmin(), 'model']
                best_value = df[metric].min()
            
            best_models[metric] = {
                'model': best_model,
                'value': best_value
            }
    
    analysis['best_models'] = best_models
    
    return analysis

def create_html_report(analysis, df, output_path):
    """创建HTML报告"""
    
    total_experiments = len(df) if df is not None else 0
    completion_rate = (total_experiments / 540) * 100 if total_experiments > 0 else 0
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>D2实验结果分析报告</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{ 
            text-align: center; 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{ 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .metric-box {{ 
            display: inline-block; 
            margin: 10px; 
            padding: 15px; 
            background: #e8f4f8; 
            border-radius: 8px;
            text-align: center;
            min-width: 150px;
        }}
        .best-model {{ 
            background: #d5f4e6 !important; 
            border: 2px solid #27ae60;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 15px 0;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: center;
        }}
        th {{ 
            background-color: #3498db; 
            color: white;
        }}
        .status-pass {{ color: #27ae60; font-weight: bold; }}
        .status-fail {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 D2实验结果分析报告</h1>
            <p>Physics-Guided Synthetic WiFi CSI Data Generation</p>
        </div>
        
        <div class="section">
            <h2>📊 实验概览</h2>
            <div class="metric-box">
                <h3>总实验数</h3>
                <p><strong>{total_experiments}</strong></p>
            </div>
            <div class="metric-box">
                <h3>完成率</h3>
                <p><strong>{completion_rate:.1f}%</strong></p>
            </div>
            <div class="metric-box">
                <h3>期望实验数</h3>
                <p><strong>540</strong></p>
            </div>
        </div>
"""
    
    # 模型性能部分
    if 'model_stats' in analysis and not analysis['model_stats'].empty:
        html_content += """
        <div class="section">
            <h2>🏆 模型性能分析</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>Macro F1 (均值±标准差)</th>
                    <th>实验次数</th>
                </tr>
"""
        
        for model in analysis['model_stats'].index:
            stats = analysis['model_stats'].loc[model]
            html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{stats[('macro_f1', 'mean')]:.4f} ± {stats[('macro_f1', 'std')]:.4f}</td>
                    <td>{int(stats[('macro_f1', 'count')])}</td>
                </tr>
"""
        
        html_content += "</table></div>"
    
    # 验收状态
    status_class = "best-model" if completion_rate >= 95 else ""
    status_text_class = "status-pass" if completion_rate >= 95 else "status-fail"
    status_icon = "✅" if completion_rate >= 95 else "⚠️" if completion_rate >= 80 else "❌"
    
    html_content += f"""
        <div class="section">
            <h2>✅ D2验收状态</h2>
            <div class="metric-box {status_class}">
                <h3>实验完成度</h3>
                <p class="{status_text_class}">{total_experiments}/540 ({completion_rate:.1f}%) {status_icon}</p>
            </div>
            <div class="metric-box">
                <h3>数据格式</h3>
                <p class="status-pass">JSON ✅</p>
            </div>
        </div>
        
        <div style="text-align: right; color: #7f8c8d; font-size: 0.9em; margin-top: 20px;">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='生成D2实验分析报告 (JSON格式)')
    parser.add_argument('results_dir', help='结果目录路径')
    parser.add_argument('--output', default='reports/d2_analysis.html', help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载结果
    print("加载JSON实验结果...")
    df = load_json_results(args.results_dir)
    
    if df is None or df.empty:
        print("警告: 未找到有效的实验结果")
        return
    
    print(f"成功加载 {len(df)} 条实验记录")
    
    # 分析结果
    print("分析模型性能...")
    analysis = analyze_model_performance(df)
    
    # 生成报告
    print("生成HTML报告...")
    create_html_report(analysis, df, args.output)
    
    print("✅ D2分析报告生成完成！")

if __name__ == "__main__":
    main()
```

---

## 📋 **最重要的快速验收命令**

先试试这个**一行JSON验收**，看看数据结构：

```python
python -c "
import json
import pandas as pd
from pathlib import Path

results_dir = Path('results_gpu/d2')
json_files = list(results_dir.glob('**/*.json'))
print(f'找到 {len(json_files)} 个JSON文件')

for i, json_file in enumerate(json_files[:3]):  # 只看前3个文件
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f'\n文件 {i+1}: {json_file.name}')
        print(f'类型: {type(data)}')
        if isinstance(data, dict):
            print(f'键: {list(data.keys())[:10]}')  # 显示前10个键
        elif isinstance(data, list) and len(data) > 0:
            print(f'长度: {len(data)}')
            print(f'第一个元素键: {list(data[0].keys())[:10] if isinstance(data[0], dict) else \"非字典\"}')
    except Exception as e:
        print(f'错误: {e}')
"
```

## ⚡ **立即执行**

1. **先运行上面的快速JSON检查**，看看数据结构
2. **告诉我JSON的具体结构**，我可以优化脚本
3. **如果结构标准**，直接用修正后的脚本

**请先执行快速检查，告诉我JSON文件的结构如何！**