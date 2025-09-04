# 📋 硬编码数据替换方案

## 一、当前硬编码数据清单

### 1. Figure 2(c) - SRV Performance Matrix
**当前状态**：硬编码
```python
performance_matrix = np.array([
    [0.89, 0.85, 0.80, 0.75, 0.70],  # CNN
    [0.91, 0.87, 0.83, 0.78, 0.73],  # BiLSTM  
    [0.93, 0.89, 0.85, 0.80, 0.75],  # Conformer
    [0.97, 0.95, 0.93, 0.90, 0.87]   # PASE-Net
])
```

**真实数据源**：`results_gpu/d2/` (540个文件)
**真实性能**：
- CNN: 94-100% (不同噪声水平)
- BiLSTM: 82-96%
- Conformer_lite: 96-100%
- Enhanced: 96-97%

### 2. Figure 3 - Cross-Domain Performance
**当前状态**：已使用真实数据 ✅
**数据源**：`results_gpu/d3/loso/` 和 `results_gpu/d3/loro/`
**真实性能**：
- PASE-Net: LOSO 83.0%, LORO 83.0%
- CNN: LOSO 84.2%, LORO 79.6%
- BiLSTM: LOSO 80.3%, LORO 78.9%

### 3. Figure 4 - Calibration
**当前状态**：已使用真实数据 ✅
**数据源**：`results_gpu/d6/`
**真实ECE**：
- Enhanced: 0.093 → 0.001
- CNN: 0.119 → 0.001

### 4. Figure 5 - Label Efficiency
**当前状态**：硬编码
```python
label_percentages = [5, 10, 20, 50, 100]
baseline_accuracy = [45, 58, 68, 78, 85]
sim2real_accuracy = [62, 71, 78, 84, 89]
pase_accuracy = [71, 79, 85, 91, 95]
```

**真实数据源**：`results_gpu/d4/sim2real/` (57个文件)
**真实性能**：
- 1%: Zero-shot 15.4%, Fine-tuned 32.9%
- 5%: Zero-shot 15.3%, Fine-tuned 22.5%
- 10%: Zero-shot 15.7%, Fine-tuned 55.6%
- 20%: Zero-shot 15.1%, Fine-tuned 82.4%
- 100%: Zero-shot 8.3%, Fine-tuned 83.3%

### 5. Figure 6 - Interpretability
**当前状态**：完全模拟（无真实注意力权重）
**替代方案**：使用Fall类型分析（真实数据）
**数据源**：`results_gpu/d3/`
**真实Fall性能**：
- Epileptic Fall: 99.5%
- Elderly Fall: 99.5%
- Fall Can't Get Up: 99.5%

### 6. Table 1 - Main Results
**当前状态**：部分硬编码
**需要更新为**：
```
Model     | LOSO Real | LORO Real | ECE Raw | ECE Cal
----------|-----------|-----------|---------|--------
PASE-Net  | 83.0%     | 83.0%     | 0.093   | 0.001
CNN       | 84.2%     | 79.6%     | 0.119   | 0.001
BiLSTM    | 80.3%     | 78.9%     | -       | -
```

## 二、修改方案

### 阶段1：数据提取脚本
创建统一的数据提取脚本，从真实实验结果中提取所有需要的数据：

```python
# extract_all_real_data.py
1. 提取SRV数据 (d2/)
2. 提取LOSO/LORO数据 (d3/)
3. 提取Sim2Real数据 (d4/)
4. 提取校准数据 (d6/)
5. 提取Fall类型数据
6. 生成数据摘要JSON文件
```

### 阶段2：图片脚本修改

#### 2.1 Figure 2(c) - scr2_physics_modeling.py
```python
# 替换硬编码矩阵
performance_matrix = load_real_srv_data()  # 从results_gpu/d2/
```

#### 2.2 Figure 5 - scr5_label_efficiency.py
```python
# 替换硬编码数组
label_data = load_real_sim2real_data()  # 从results_gpu/d4/
```

#### 2.3 Figure 6 - 新建scr6_fall_analysis.py
```python
# 全新图片展示Fall类型性能
fall_performance = load_fall_type_data()  # 从results_gpu/d3/
```

### 阶段3：LaTeX文档更新

#### 3.1 更新Table 1
- 使用真实LOSO/LORO数据
- 使用真实校准数据

#### 3.2 更新文本描述
- Results section: 匹配真实数据
- Abstract: 使用真实83%性能
- Figure captions: 标注"Real experimental data"

#### 3.3 添加数据说明
```latex
\subsection{Experimental Data}
We conduct experiments on two types of data:
\begin{itemize}
\item Real-world WiFi CSI from WiFi-CSI-Sensing-Benchmark
\item Physics-based synthetic data for controlled ablations
\end{itemize}
```

### 阶段4：验证和提交

1. 运行所有修改后的脚本
2. 检查生成的图片
3. 验证数值一致性
4. Git提交到feat/enhanced-model-and-sweep

## 三、具体修改列表

| 文件 | 修改内容 | 数据源 |
|------|----------|--------|
| scr2_physics_modeling.py | 替换performance_matrix | results_gpu/d2/ |
| scr5_label_efficiency.py | 替换label arrays | results_gpu/d4/sim2real/ |
| scr6_interpretability.py | 替换为fall_analysis.py | results_gpu/d3/ |
| enhanced_claude_v1.tex | 更新Table 1数值 | results_gpu/d3/, d6/ |
| enhanced_claude_v1.tex | 更新Results文本 | 所有真实数据 |
| enhanced_claude_v1.tex | 更新Figure captions | 添加"Real data"标注 |

## 四、预期结果

### 修改前问题：
- 硬编码数据无法验证
- 性能数值不一致
- 缺乏真实实验支撑

### 修改后优势：
- ✅ 所有数据可追溯到实验结果
- ✅ 性能数值真实可信
- ✅ 满足学术诚信要求
- ✅ 结果可重现

## 五、时间估计

- 数据提取脚本：30分钟
- 图片脚本修改：45分钟
- LaTeX更新：30分钟
- 验证和调试：30分钟
- Git提交：15分钟

**总计：约2.5小时**

## 六、风险和注意事项

### 风险1：真实性能可能与论文声称不同
**缓解**：诚实报告，解释差异原因

### 风险2：部分实验数据缺失
**缓解**：明确标注为future work

### 风险3：高合成性能（>95%）
**缓解**：解释为合成数据特性，强调真实数据的83%

## 七、执行确认清单

- [ ] 确认使用results_gpu/d2/的SRV数据
- [ ] 确认使用results_gpu/d3/的LOSO/LORO数据
- [ ] 确认使用results_gpu/d4/的Sim2Real数据
- [ ] 确认使用results_gpu/d6/的校准数据
- [ ] 确认使用Fall类型分析替代注意力可视化
- [ ] 确认更新所有相关文本描述
- [ ] 确认添加数据来源说明
- [ ] 确认提交到正确的Git分支

---

**请确认以上方案，确认后将立即执行所有修改。**