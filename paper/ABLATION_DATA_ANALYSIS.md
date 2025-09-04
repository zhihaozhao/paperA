# 📊 Ablation Study数据分析报告

## ✅ 数据验证结果

**`ablation_noise_env_claude4.pdf`图片包含真实实验数据！**

### 📈 实验数据统计

#### 数据覆盖
- **总实验文件数**: 每个模型45个文件
- **实验条件**: 9种组合 (3个class overlap × 3个environment burst)
- **每个条件重复次数**: 5次（seeds 0-4）
- **固定label noise**: 0.05

#### 性能结果（Macro F1）

| Model | Mean F1 | Min F1 | Max F1 | 最差条件 |
|-------|---------|--------|--------|----------|
| **PASE-Net** | 0.950 | 0.684 | 1.000 | Class=0.8, Env=0.0: 0.706 |
| **CNN** | 0.947 | 0.707 | 1.000 | Class=0.8, Env=0.0: 0.713 |
| **BiLSTM** | 0.921 | 0.699 | 1.000 | Class=0.8, Env=0.0: 0.712 |

### 🔍 详细数据分析

#### PASE-Net (Enhanced)性能矩阵
```
                Environment Burst Rate
                0.0     0.1     0.2
Class    0.0   1.000   1.000   1.000
Overlap  0.4   0.928   0.985   0.993
         0.8   0.706   0.959   0.979
```

#### 关键发现
1. **所有模型在无噪声条件下达到100% F1** (Class=0.0, Env=0.0/0.1/0.2)
2. **高class overlap (0.8) + 无环境噪声 (0.0)** 是最具挑战的条件
   - PASE-Net: 70.6%
   - CNN: 71.3%
   - BiLSTM: 71.2%
3. **环境噪声反而提升了性能** - 这可能是因为环境噪声起到了正则化作用

### ⚠️ 数据异常分析

#### 异常现象
- 当class overlap=0.8时，增加环境噪声反而提升性能
  - Env=0.0: ~71%
  - Env=0.1: ~90-96%
  - Env=0.2: ~95-98%

#### 可能原因
1. **正则化效应**: 环境噪声可能防止过拟合
2. **训练策略**: 模型可能在有噪声环境下训练，对噪声有适应性
3. **数据特性**: 合成数据的特定属性

### 📝 论文描述修正建议

原文描述：
> "PASE-Net maintains >85% macro-F1 under combined stress conditions"

**实际数据显示**：
- PASE-Net在最差条件下只有70.6% F1
- 但在大多数条件下确实>85%（9个条件中7个）

建议修改为：
> "PASE-Net achieves >95% macro-F1 in 7 out of 9 stress conditions, with performance degrading to 70.6% only under extreme class overlap (0.8) without environmental noise"

### 🎯 图片状态

- **文件大小**: 52KB（不是空的）
- **数据来源**: `results_gpu/d2/`真实实验数据
- **可视化类型**: 热力图展示不同噪声条件下的性能
- **用途**: 补充材料（已被注释）

### ✅ 结论

1. **数据是真实的** - 来自135个实验文件（3模型×45文件）
2. **图片有效** - 包含完整的实验结果可视化
3. **适合作为补充材料** - 展示了模型在不同噪声条件下的鲁棒性

### 📊 数据文件示例
```
results_gpu/d2/paperA_enhanced_hard_s0_cla0p8_env0p0_lab0p05.json
                     ↑模型    ↑难度 ↑种子 ↑class ↑env  ↑label
                                        overlap burst  noise
```

---

**验证完成**: ablation_noise_env_claude4.pdf包含真实有效的实验数据！