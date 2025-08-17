# D2实验完整指南

## 📋 D2实验基本情况

### 🎯 实验定位
- **实验代号**: D2 (Day 2)
- **实验阶段**: P0.1 完成阶段
- **实验性质**: 扩展多种子/多难度 + 汇总分析
- **前置依赖**: D1实验成功完成 (四个模型基础验证)

### 🔬 实验目标
D2实验是WiFi CSI人体行为识别项目的关键验证节点，主要完成：

1. **多维度参数扫描**: 在多个随机种子和难度级别下验证模型稳定性
2. **统计分析**: 生成汇总CSV和可视化图表
3. **重叠度分析**: 验证类别特征重叠与分类误差的相关性
4. **模型对比**: 确认Enhanced模型相对于基线模型的优势

### 📊 实验规模
- **模型数量**: 4个 (Enhanced, CNN, BiLSTM, Conformer-Lite)
- **随机种子**: 5个 (seeds: 0-4)
- **网格参数**: 
  - Class Overlap: 3个值 [0.0, 0.4, 0.8]
  - Label Noise: 3个值 [0.0, 0.05, 0.1] 
  - Env Burst Rate: 3个值 [0.0, 0.1, 0.2]
- **总实验数**: 4 × 5 × 3 × 3 × 3 = **540个**单独实验
- **数据规模**: 每个实验默认样本数 (固定难度: hard)

## 🎯 D2验收标准详解

### ✅ 核心验收指标

#### 1. **分类性能指标**
```
- Macro F1 Score: 综合四类行为的F1分数
  • 期望范围: 0.85-0.95 (避免完美拟合)
  • 验收标准: 显示模型有提升空间

- Falling F1 Score: 跌倒检测专项F1
  • 严格标准: < 0.99 (必须未达完美)
  • 目的: 确保任务具有挑战性，为后续优化预留空间
```

#### 2. **误分类分析指标**
```
- Mutual Misclass Rate: 类间混淆度
  • 验收标准: > 0 (必须存在一定混淆)
  • 目的: 验证数据集难度合理，存在类间重叠

- Class Overlap Statistics: 特征重叠统计
  • 期望: overlap_stat 字段包含均值、标准差等统计量
```

#### 3. **可信度评估指标**
```
- ECE (Expected Calibration Error): 期望校准误差
  • Enhanced模型: 通过λ正则化显著改善
  • 验收阈值: 相比基线降低 > 5%

- Brier Score: 概率预测质量
  • 期望: Enhanced < Baseline models
  
- Temperature Scaling: 后校准效果
  • 验收: ece_cal < ece_raw (校准后改善)
```

#### 4. **统计显著性**
```
- 重叠度-误差回归分析:
  • 回归斜率: > 0 (正相关)
  • 统计显著性: p < 0.05
  • R² 系数: > 0.3 (合理解释方差)
  
- 多种子稳定性:
  • 每个(模型,难度)组合≥3个种子
  • 标准差/均值 < 0.15 (变异系数)
```

### 📈 产出文件验收清单

#### 必需产出文件
```
✅ results/synth/summary.csv
   - 包含所有540个实验的汇总指标
   - 字段: model, seed, class_overlap, label_noise_prob, env_burst_rate, macro_f1, falling_f1, ece, brier等

✅ plots/fig_synth_bars.pdf  
   - 四个模型在不同参数组合下的性能条形图
   - 误差棒显示多种子标准差

✅ plots/fig_overlap_scatter.pdf
   - 特征重叠度 vs 分类误差散点图
   - 包含回归线、R²、p值标注

✅ results/synth/metrics_by_model_difficulty.csv
   - 按模型和难度聚合的统计摘要
```

#### 可选产出文件
```
📊 plots/fig_calibration_comparison.pdf
   - ECE对比图 (原始 vs 温度校准后)

📊 tables/tab_statistical_tests.tex  
   - 配对t检验和Cohen's d效应量

📊 plots/fig_reliability_diagrams.pdf
   - 可靠性图表 (置信度 vs 准确度)
```

## 🚀 D2实验执行指南

### 🔧 环境准备

```bash
# 1. 确认环境和依赖
python -c "import torch, numpy, sklearn, matplotlib; print('✅ Dependencies OK')"

# 2. 确认GPU可用 (可选但推荐)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 清理旧缓存 (可选)
rm -rf cache/synth_data/*  # 或保留用于加速
```

### ⚡ 快速执行路径

#### 方案A: 使用预生成缓存 (推荐)
```bash
# 1. 预生成所有需要的数据集 (~10-15小时一次性投入)
python scripts/pregenerate_d2_datasets.py --spec scripts/d2_spec.json

# 2. 运行D2实验 (~3-4小时，从缓存加载)
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json --resume

# 3. 生成汇总分析
python scripts/analyze_d2_results.py --input results_gpu/d2 --output results/synth
```

#### 方案B: 逐步执行 (调试友好)
```bash
# 1. 小规模测试 (3个实验)
python scripts/pregenerate_d2_datasets.py --spec scripts/d2_spec_optimized.json

# 2. 验证结果质量
python scripts/check_d2_integrity.py --results results_gpu/d2_optimized

# 3. 满意后扩展到全规模
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json --resume
```

### 📋 实验配置文件

#### 标准D2配置 (`scripts/d2_spec.json`)
```json
{
  "models": ["enhanced", "cnn", "bilstm", "conformer_lite"],
  "seeds": [0, 1, 2, 3, 4],
  "fixed": {
    "difficulty": "hard",
    "epochs": 100,
    "batch": 768,
    "amp": true,
    "save_ckpt": "final",
    "val_every": 3,
    "num_workers_override": 0
  },
  "grid": {
    "class_overlap": [0.0, 0.4, 0.8],
    "label_noise_prob": [0.0, 0.05, 0.1],
    "env_burst_rate": [0.0, 0.1, 0.2]
  },
  "output_dir": "results_gpu/d2"
}
```

### 🔍 实时监控与验收

#### 监控关键指标
```bash
# 1. 监控实验进度
tail -f logs/d2_experiment.log

# 2. 实时查看结果摘要
python -c "
import pandas as pd
df = pd.read_csv('results/synth/summary.csv')
print(f'Progress: {len(df)}/540 experiments')
print(f'Falling F1 range: {df.falling_f1.min():.3f}-{df.falling_f1.max():.3f}')
print(f'Mutual misclass > 0: {(df.mutual_misclass > 0).sum()}/{len(df)}')
print(f'Parameter combinations completed: {df.groupby([\"class_overlap\", \"label_noise_prob\", \"env_burst_rate\"]).ngroups}/27')
"

# 3. 验收关键指标
python scripts/validate_d2_acceptance.py
```

#### 预期性能基准
```
✅ 正常范围指标:
- Macro F1: 0.88 ± 0.05
- Falling F1: 0.92 ± 0.04 (< 0.99)
- ECE (Enhanced): 0.08 ± 0.02
- ECE (Baselines): 0.12 ± 0.03
- Mutual Misclass: 0.05 ± 0.02 (> 0)

⚠️ 异常情况处理:
- 若 Falling F1 = 1.00: 增加难度或噪声
- 若 Mutual Misclass = 0: 检查类重叠设置
- 若 ECE 无改善: 检查λ正则化参数
```

## 🛠️ 故障排除与优化

### 常见问题与解决方案

#### 1. **内存/显存不足**
```bash
# 减少批量大小和样本数
python scripts/run_sweep_from_json.py --spec scripts/d2_spec_small.json
# 或分批执行
for model in enhanced lstm tcn txf; do
    python scripts/run_single_model.py --model $model --spec scripts/d2_spec.json
done
```

#### 2. **数据生成太慢**
```bash
# 启用缓存机制 (已实现)
export CACHE_DIR="cache/synth_data"
# 或使用SSD存储缓存
export CACHE_DIR="/fast_ssd/paperA_cache"
```

#### 3. **实验中断恢复**
```bash
# 使用--resume标志跳过已完成的实验
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json --resume --max 50
```

### 性能优化建议

#### 硬件配置推荐
- **CPU**: 8+ cores (并行数据生成)
- **GPU**: 8GB+ VRAM (批量训练)  
- **RAM**: 32GB+ (数据缓存)
- **存储**: 100GB+ 可用空间

#### 软件优化
```python
# 在train_eval.py中启用的优化
--num_workers_override 4        # 数据加载并行
--pin_memory True              # GPU内存优化  
--amp True                     # 混合精度训练
--cache_dir /fast_storage/cache # SSD缓存
```

## 📊 D2实验后续流程

### ✅ D2验收通过后
1. **归档实验结果**: 保存到`results/d2_final/`
2. **更新项目状态**: P0.1 → P0.2 (进入D3阶段)
3. **准备D3实验**: λ扫描与容量对齐
4. **文档更新**: 更新项目README和进度报告

### ❌ D2验收失败处理
1. **分析失败原因**: 检查指标是否达标
2. **参数调优**: 修改数据生成参数或模型配置
3. **重新实验**: 针对性重跑失败的部分
4. **寻求支持**: 联系技术团队讨论解决方案

## 📝 附录

### 重要文件清单
```
D2实验核心文件:
├── scripts/
│   ├── d2_spec.json                    # 实验配置
│   ├── pregenerate_d2_datasets.py      # 数据预生成
│   ├── run_sweep_from_json.py          # 批量实验执行
│   └── validate_d2_acceptance.py       # 验收检查
├── src/
│   ├── train_eval.py                   # 训练评估主程序
│   ├── data_synth.py                   # 合成数据生成(含缓存)
│   └── models.py                       # 模型定义
└── results/
    ├── synth/summary.csv               # 汇总结果
    └── plots/fig_*.pdf                 # 可视化图表
```

### 联系与支持
- **技术问题**: 检查 `logs/` 目录错误日志
- **性能问题**: 使用 `scripts/benchmark_system.py` 测试硬件
- **结果异常**: 运行 `scripts/diagnose_d2_results.py` 诊断

---

**最后更新**: 2025-08-17  
**文档版本**: v1.0  
**兼容实验版本**: D2 Phase P0.1