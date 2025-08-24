# Figure 4: Vision Model Performance Meta-Analysis 数据采集模板

## 📋 采集目标
为Figure 4生成真实的视觉模型性能元分析图表，需要从109篇相关论文中提取真实的性能数据。

## 🎯 图表设计 (2x2布局)

### (a) Algorithm Family Performance Distribution
**数据需求**: 不同算法家族的性能分布对比
- **算法家族**: R-CNN系列, YOLO系列, SSD系列, 其他CNN
- **性能指标**: 准确率(%), mAP, F1-score
- **样本量**: 每个家族至少10篇论文

### (b) Recent Model Achievements and Temporal Evolution  
**数据需求**: 2015-2024年模型性能时间演进
- **时间轴**: 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
- **性能指标**: 最高准确率, 平均准确率
- **技术里程碑**: 重要模型发布时间点

### (c) Real-time Processing Capability Analysis
**数据需求**: 实时处理能力vs准确率散点图
- **X轴**: 处理速度 (FPS 或 ms/frame)
- **Y轴**: 准确率 (%)
- **数据点**: 每篇论文的性能数据
- **标注**: 算法名称

### (d) Environmental Robustness Comparison
**数据需求**: 不同环境条件下的性能对比
- **环境类型**: 温室, 果园, 田间, 实验室
- **性能指标**: 准确率, 鲁棒性评分
- **样本分布**: 每种环境至少15篇论文

---

## 📊 数据采集表格

### 基础信息采集
| 论文ID | 标题 | 年份 | 期刊/会议 | 算法类型 | 数据集 | 环境类型 |
|--------|------|------|----------|----------|--------|----------|
| tang2020recognition | Recognition and localization methods for vision-based fruit picking robots | 2020 | Frontiers in Plant Science | Review | - | Mixed |
| mavridou2019machine | Machine vision systems in precision agriculture for crop farming | 2019 | Journal of Imaging | Review | - | Mixed |
| Wang2019 | Faster R-CNN for multi-class fruit detection using a robotic vision system | 2019 | - | Faster R-CNN | Custom | Orchard |
| jia2020detection | Detection and segmentation of overlapped fruits based on optimized mask R-CNN | 2020 | - | Mask R-CNN | Custom | Greenhouse |
| fu2020faster | Faster R-CNN-based apple detection in dense-foliage fruiting-wall trees | 2020 | - | Faster R-CNN | Custom | Orchard |
| ... | ... | ... | ... | ... | ... | ... |

### 性能数据采集
| 论文ID | 准确率(%) | 精确率(%) | 召回率(%) | F1-Score | mAP | 处理速度(FPS) | 处理时间(ms) | 数据集大小 | 测试条件 |
|--------|-----------|-----------|-----------|----------|-----|---------------|-------------|----------|----------|
| tang2020recognition | - | - | - | - | - | - | - | - | Review Paper |
| mavridou2019machine | - | - | - | - | - | - | - | - | Review Paper |
| Wang2019 | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** |
| jia2020detection | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** |
| fu2020faster | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** | **需要人工填写** |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 🔍 数据采集指南

### 1. 性能指标定义
- **准确率 (Accuracy)**: 正确分类的样本占总样本的比例
- **精确率 (Precision)**: TP/(TP+FP)，预测为正的样本中实际为正的比例  
- **召回率 (Recall)**: TP/(TP+FN)，实际为正的样本中预测为正的比例
- **F1-Score**: 2×(Precision×Recall)/(Precision+Recall)
- **mAP (mean Average Precision)**: 多类别检测的平均精确率
- **处理速度**: 每秒处理帧数(FPS)或每帧处理时间(ms)

### 2. 数据提取优先级
1. **实验结果表格**: 直接从论文的Results/Experiments部分提取
2. **性能对比图**: 从图表中读取数值
3. **文字描述**: 从结论中提取关键性能数据
4. **Abstract**: 如果正文没有具体数值，从摘要中提取

### 3. 数据质量要求
- ✅ **真实性**: 必须来自原文，不能估算或猜测
- ✅ **完整性**: 尽量填写所有相关指标
- ✅ **准确性**: 数值必须与原文完全一致
- ✅ **可追溯**: 记录数据来源页码或章节

### 4. 特殊情况处理
- **Review论文**: 标记为"Review Paper"，不提取性能数据
- **缺失数据**: 标记为"Not Reported"或"N/A"
- **范围数据**: 记录为"85-92%"的格式
- **多个实验**: 选择最佳性能或平均性能

---

## 📝 采集流程

### Step 1: 论文筛选
从109篇候选论文中选择包含实验结果的论文（排除纯综述）

### Step 2: 数据提取  
逐篇阅读，提取上述表格中的所有指标

### Step 3: 数据验证
检查数据的合理性和一致性

### Step 4: 分类汇总
按算法家族、年份、环境类型进行分类

### Step 5: 生成真实图表
基于真实数据重新生成Figure 4

---

## ⚠️ 重要提醒

1. **绝对不能编造数据** - 所有数据必须来自原文
2. **记录数据来源** - 标注页码、表格编号、图表编号  
3. **保持数据完整性** - 不要遗漏重要指标
4. **注意单位统一** - 确保所有数据使用相同单位
5. **验证数据合理性** - 检查是否在合理范围内

---

## 📁 输出格式

完成采集后，请将数据保存为：
- `FIGURE4_REAL_DATA.json` - 结构化数据
- `FIGURE4_REAL_DATA.csv` - 表格数据  
- `FIGURE4_DATA_SOURCES.md` - 数据来源文档

---

*模板创建时间: 2024*  
*状态: 等待人工数据采集*