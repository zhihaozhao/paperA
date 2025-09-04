# 📊 图片和脚本修改报告

## 🔄 修改的图片和脚本总览

### ⚠️ 重要发现
根据代码分析，**manuscript/plots/中的脚本大部分仍使用硬编码数据**，只有部分进行了修改以使用真实数据。

---

## 📝 详细修改情况

### 1. Figure 2: Physics Modeling (`scr2_physics_modeling.py`)
**修改状态**: ✅ **部分修改**

#### 修改内容：
- **第125-161行**: 添加了从真实数据加载SRV结果的代码
- **数据源**: `/workspace/paper/scripts/extracted_data/srv_performance.json`
- **Fallback**: 如果找不到数据文件，使用基于真实平均值的硬编码数据

```python
# 修改后的代码片段
data_file = Path('/workspace/paper/scripts/extracted_data/srv_performance.json')
if data_file.exists():
    with open(data_file, 'r') as f:
        srv_data = json.load(f)
    # 使用真实数据构建性能矩阵
else:
    # 使用基于真实实验平均值的fallback
    performance_matrix = np.array([
        [0.946, 0.940, 0.930, 0.920, 0.900],  # CNN (real avg: 94.6%)
        [0.921, 0.910, 0.900, 0.880, 0.860],  # BiLSTM (real avg: 92.1%)
        [0.930, 0.920, 0.910, 0.890, 0.870],  # Conformer
        [0.949, 0.940, 0.930, 0.920, 0.910]   # PASE-Net (real avg: 94.9%)
    ])
```

---

### 2. Figure 3: Cross-Domain (`scr3_cross_domain.py`)
**修改状态**: ❌ **未修改** (仍使用硬编码)

#### 当前状态：
- 仍然使用硬编码的性能数据
- 有注释提到"Load real cross-domain data"但实际未实现
- 需要使用`supplementary/scripts/figure_generation/scr3_cross_domain_FINAL.py`

---

### 3. Figure 4: Calibration (`scr4_calibration.py`)
**修改状态**: ❌ **未修改** (仍使用模拟数据)

#### 当前状态：
- 使用"realistic simulation"而非真实数据
- 需要使用`supplementary/scripts/figure_generation/scr4_calibration_REAL.py`

---

### 4. Figure 5: Label Efficiency (`scr5_label_efficiency.py`)
**修改状态**: ❌ **未修改** (仍使用硬编码)

#### 当前状态：
- 使用硬编码的Sim2Real数据
- 需要使用`supplementary/scripts/figure_generation/scr5_label_efficiency_FINAL.py`

---

### 5. Figure 6: Interpretability (`scr6_interpretability.py`)
**修改状态**: ❌ **未修改** (使用模拟数据)

#### 当前状态：
- 使用"Simulate realistic SE attention patterns"
- 这个图本质上是可视化，可能不需要真实实验数据

---

## 📂 Supplementary中的真实数据脚本

### ✅ 已创建的真实数据版本脚本

位置：`paper/paper2_pase_net/supplementary/scripts/figure_generation/`

| 脚本名称 | 用途 | 数据源 |
|---------|------|--------|
| `scr2_srv_REAL.py` | SRV性能图 | `extracted_data/srv_performance.json` |
| `scr3_cross_domain_REAL.py` | 跨域性能 | `extracted_data/cross_domain_performance.json` |
| `scr3_cross_domain_FINAL.py` | 跨域性能(完整版) | 同上 |
| `scr4_calibration_REAL.py` | 校准分析 | `extracted_data/calibration_metrics.json` |
| `scr5_label_efficiency_REAL.py` | 标签效率 | `extracted_data/label_efficiency.json` |
| `scr5_label_efficiency_FINAL.py` | 标签效率(完整版) | 同上 |
| `scr6_fall_detection_FINAL.py` | 跌倒检测 | `extracted_data/fall_detection_performance.json` |
| `scr6_fall_types_REAL.py` | 跌倒类型分析 | 同上 |

---

## 🔧 需要的修改

### 紧急修改建议

1. **替换manuscript/plots/中的脚本**
   ```bash
   # 用真实数据版本替换当前脚本
   cp supplementary/scripts/figure_generation/scr3_cross_domain_FINAL.py \
      manuscript/plots/scr3_cross_domain.py
   
   cp supplementary/scripts/figure_generation/scr4_calibration_REAL.py \
      manuscript/plots/scr4_calibration.py
   
   cp supplementary/scripts/figure_generation/scr5_label_efficiency_FINAL.py \
      manuscript/plots/scr5_label_efficiency.py
   ```

2. **重新生成图片**
   ```bash
   cd manuscript/plots
   python3 scr2_physics_modeling.py  # 已部分修改
   python3 scr3_cross_domain.py      # 需要替换
   python3 scr4_calibration.py       # 需要替换
   python3 scr5_label_efficiency.py  # 需要替换
   ```

---

## 📊 数据真实性状态总结

| 图片 | 文件名 | 脚本状态 | 数据状态 | 行动建议 |
|------|--------|---------|---------|---------|
| Fig 1 | `fig1_system_architecture.pdf` | ✅ | 架构图(无数据) | 无需修改 |
| Fig 2 | `fig2_physics_modeling_new.pdf` | ⚠️ | 部分真实数据 | 检查数据文件存在性 |
| Fig 3 | `fig3_cross_domain.pdf` | ❌ | 硬编码数据 | **需要替换脚本** |
| Fig 4 | `fig4_calibration.pdf` | ❌ | 模拟数据 | **需要替换脚本** |
| Fig 5 | `fig5_label_efficiency.pdf` | ❌ | 硬编码数据 | **需要替换脚本** |
| Fig 6 | `fig6_interpretability.pdf` | ⚠️ | 可视化(可接受) | 可选更新 |

---

## ⚠️ 关键问题

### 数据文件缺失
manuscript/plots/脚本引用的数据文件路径：
- `/workspace/paper/scripts/extracted_data/srv_performance.json`

但实际数据文件在：
- `/workspace/paper/paper2_pase_net/supplementary/data/processed/`

需要：
1. 创建符号链接，或
2. 修改脚本中的路径，或
3. 复制数据文件到期望位置

---

## 🎯 推荐行动

### Option 1: 快速修复（推荐）
使用supplementary中已经准备好的真实数据脚本：
```bash
cd paper/paper2_pase_net/supplementary/scripts/figure_generation
python3 generate_all_figures.py
# 然后复制生成的图片到manuscript/plots/
```

### Option 2: 修复现有脚本
1. 更新manuscript/plots/中的脚本路径
2. 确保数据文件可访问
3. 重新生成所有图片

### Option 3: 保持现状风险
- Figure 3, 4, 5 使用的是硬编码/模拟数据
- 可能被审稿人质疑数据真实性
- **不推荐用于正式投稿**

---

**建议：在投稿前必须确保所有图片使用真实实验数据！**