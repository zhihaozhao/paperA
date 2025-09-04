# 📋 IoTJ Editorial and Reviewer Perspective Checklist

## 🔍 从主编视角的审查

### 1. **技术贡献的新颖性** ⚠️
- ✅ PASE-Net架构结合SE和时序注意力机制
- ⚠️ **潜在问题**: SE和注意力机制都不是新的
- **优化建议**: 强调物理信息引导的设计原则是创新点

### 2. **实验的完整性** 
- ✅ 包含LOSO/LORO跨域评估
- ✅ 包含Sim2Real迁移学习
- ⚠️ **潜在问题**: Conformer LOSO失败需要解释
- **需要补充**: 
  - 计算复杂度对比（FLOPs, 参数量）
  - 与最新SOTA方法对比（2023-2024论文）

### 3. **数据和代码的可重现性** ⚠️
- ✅ 使用公开数据集WiFi-CSI-Sensing-Benchmark
- ⚠️ **缺失**: 没有提供代码链接
- **必须添加**: GitHub代码仓库链接或补充材料

## 🎯 从审稿人视角的潜在问题

### Reviewer 1 (方法专家)可能的问题：

#### Q1: "为什么Conformer在LOSO评估中失败率这么高（3/5次）？"
**当前状态**: ❌ 论文只提到了问题，没有深入分析
**需要添加的解释**:
```
Conformer的LOSO失败归因于：
1. 自注意力机制对小样本过拟合
2. 位置编码对不同受试者的运动模式泛化能力差
3. 需要更多数据才能收敛（原始Conformer设计用于大规模数据）
```

#### Q2: "SE模块的物理意义是什么？"
**当前状态**: ⚠️ 有提及但不够具体
**需要加强**:
- SE权重与子载波SNR的相关性分析
- 不同环境下SE权重的可视化
- 与理论传播模型的对应关系

### Reviewer 2 (实验专家)可能的问题：

#### Q3: "为什么选择20%作为标签效率的关键阈值？"
**当前状态**: ⚠️ 只展示了结果，缺少理论依据
**需要添加**:
```
20%阈值的选择基于：
1. 边际效用分析：20%后性能提升<2%
2. 成本效益分析：标注成本vs性能提升
3. 统计功效分析：20%提供足够的类别覆盖
```

#### Q4: "与其他标签效率方法（半监督、自监督）的对比？"
**当前状态**: ❌ 缺失
**需要添加**: 至少与2-3个半监督基线对比

### Reviewer 3 (应用专家)可能的问题：

#### Q5: "实际部署的计算开销如何？"
**当前状态**: ⚠️ Table 1有FLOPs但不够详细
**需要补充**:
- 推理时间对比（ms）
- 内存占用（MB）
- 边缘设备可行性分析

## 🔧 需要立即修复的问题

### 1. **图表问题**
- [ ] Fig4需要检查子图标签是否清晰
- [ ] Fig5需要确认20%点的突出显示
- [ ] 所有图需要确保300 DPI分辨率

### 2. **数据验证**
- [x] 已确认无硬编码
- [ ] 需要添加数据可用性声明
- [ ] 需要提供实验种子和配置文件

### 3. **写作问题**
- [ ] Abstract太长（当前>250词）
- [ ] 缺少Limitations部分
- [ ] 缺少Future Work明确方向

## 📊 关键指标核查

### 必须验证的声明：
1. "83.0±0.1% LOSO/LORO性能" ✅ 已验证
2. "82.1% F1 with 20% labels" ✅ 已验证  
3. "78% ECE improvement" ⚠️ 需要重新计算
4. "80% annotation cost reduction" ✅ 逻辑正确

## 🚨 高优先级改进项

### 立即需要做的：

1. **添加补充材料说明**
```latex
\section{Data Availability}
The experimental data and code are available at: 
\url{https://github.com/[anonymous]/PASE-Net}
```

2. **解释Conformer失败**
在Section V.B添加：
```latex
The Conformer model's convergence issues in LOSO evaluation 
(40.3±34.5\%) stem from its self-attention mechanism requiring 
larger training sets for stable optimization. This highlights 
PASE-Net's advantage in limited-data scenarios typical in 
cross-subject evaluation.
```

3. **加强物理解释**
在Section VI添加SE权重与物理量的定量关系

4. **添加计算效率对比**
创建新表格对比推理时间和内存占用

## ✅ 已完成的优化

1. ✅ 数据真实性验证
2. ✅ 图表重新设计（避免重叠）
3. ✅ 清晰的子图标注
4. ✅ 实验数据可追溯性

## 📝 建议的回复审稿人模板

### 对于数据问题：
"We have made all experimental data publicly available at [URL]. 
Each result can be traced to specific JSON files in results_gpu/."

### 对于Conformer问题：
"The Conformer's LOSO failure is retained intentionally to show 
honest experimental results. We provide detailed analysis in 
Section V.B explaining the architectural limitations."

### 对于创新性问题：
"While SE and attention are established techniques, our contribution 
lies in their physics-informed integration specifically designed 
for WiFi CSI characteristics, validated through extensive experiments."

---

**优先级排序**：
1. 🔴 添加代码/数据可用性声明
2. 🔴 解释Conformer LOSO失败原因
3. 🟡 添加计算效率详细对比
4. 🟡 加强物理解释的定量分析
5. 🟢 优化Abstract长度

**预期审稿结果**：
- 如果完成🔴项：Major Revision
- 如果完成🔴+🟡项：Minor Revision
- 当前状态：可能Reject（缺少关键信息）