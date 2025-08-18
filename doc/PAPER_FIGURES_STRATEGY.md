# 📊 PaperA 论文图表策略

**目标期刊**: IEEE IoTJ/TMC/IMWUT  
**论文主题**: Physics-Guided Synthetic WiFi CSI for Trustworthy HAR

---

## 🎯 **Top期刊图表标准** (基于IEEE/ACM要求)

### **技术规范**:
- **分辨率**: 彩图300 DPI, 灰度图600 DPI, 线图1200 DPI
- **尺寸**: 单栏8.3cm, 双栏17.1cm, 高度≤23.3cm
- **字体**: Helvetica/Arial, 8-12pt, 最小4.5pt
- **线条**: 最小0.5pt宽度
- **格式**: PDF/EPS (矢量), TIFF (位图)
- **颜色**: RGB模式, 黑白打印兼容

### **内容要求**:
- **自明性**: 图表独立可理解
- **一致性**: 统一风格、字体、配色
- **清晰性**: 避免过度拥挤，重点突出
- **专业性**: 高质量、精确、无冗余

---

## 📋 **基于论文结构的图表规划**

### **Section 3: Methods** (2-3个图)

#### **Figure 1: Physics-Guided Synthetic Data Generation Framework** 🏗️
```
类型: 系统架构图 (双栏)
内容: 
├── 物理建模模块 (多路径、人体交互、环境变化)
├── 合成数据生成器
├── Enhanced模型架构 (CNN+SE+Attention)  
└── Sim2Real迁移流程

设计要点:
- 清晰的数据流向箭头
- 模块间的依赖关系
- 物理公式的可视化表示
- 颜色编码不同的处理阶段
```

#### **Figure 2: Enhanced Model Architecture** 🧠
```
类型: 网络结构图 (单栏)
内容:
├── Input: CSI Tensor (T×F×N)
├── CNN Feature Extraction
├── SE Module (Squeeze & Excitation)
├── Temporal Attention Mechanism  
└── Output: Activity Classification

设计要点:
- 清晰的层级结构
- 特征维度标注
- SE和Attention模块的详细展示
- 与传统CNN的对比
```

### **Section 5: Results** (4-6个核心图)

#### **Figure 3: D3 Cross-Domain Generalization Performance** 📈
```
类型: 分组柱状图 (双栏)
数据源: results/metrics/summary_d3.csv
内容:
├── LOSO vs LORO协议对比
├── 4个模型 (Enhanced, CNN, BiLSTM, Conformer)
├── Macro F1性能 + 误差棒
└── 突出Enhanced模型的优势

关键数据:
- Enhanced LOSO: 83.0±0.1% F1
- Enhanced LORO: 83.0±0.1% F1  
- 与基线模型的显著性差异
```

#### **Figure 4: D4 Sim2Real Label Efficiency Curve** 🎯
```
类型: 多线图 + 标注 (双栏)
数据源: results/metrics/summary_d4.csv
内容:
├── X轴: 标签比例 (1%, 5%, 10%, 20%, 100%)
├── Y轴: Macro F1性能
├── 多条线: Enhanced, CNN, BiLSTM基线
├── 重点标注: 82.1% F1 @ 20%标签
└── 目标线: 80% 性能阈值

设计亮点:
- 标签效率的清晰趋势
- 成本效益的visual evidence
- 与其他方法的gap analysis
```

#### **Figure 5: Model Calibration and Trustworthiness** 🎨
```
类型: 多子图组合 (双栏)
子图A: Reliability Diagram (对角线 + 实际曲线)
子图B: ECE对比柱状图  
子图C: 置信度分布直方图
子图D: Temperature Scaling效果

数据重点:
- Enhanced模型ECE < 0.01 (excellent calibration)
- 校准前后的对比
- 模型可信度的visual evidence
```

#### **Figure 6: Transfer Learning Effectiveness** 🔄
```
类型: 热力图 + 性能矩阵 (单栏)
内容:
├── 不同迁移方法的效果矩阵
├── Zero-shot vs Linear Probe vs Fine-tune
├── 标签比例 vs 性能的热力图
└── 最优配置的高亮显示

设计目标:
- 直观显示迁移学习的有效性
- 帮助选择最佳迁移策略
- 支持实际部署决策
```

### **Section 4/5: 可解释性分析** (如果加入CHAP)

#### **Figure 7: Model Interpretability Analysis** 🔍
```
类型: 可视化组合图 (双栏)
可能包含:
├── Attention权重热力图
├── Feature重要性分析
├── 激活图可视化
└── 决策边界分析

前提: 需要明确CHAP的具体方法
```

---

## 🤔 **关于CHAP可解释性方法**

**我需要你的确认**: CHAP具体指什么方法？可能的候选：

1. **Channel Attention Pooling** - 通道注意力机制
2. **Class Activation Heatmap** - 类激活热图 (类似CAM/Grad-CAM)  
3. **Contextual Hierarchical Attention Pooling** - 分层注意力
4. **其他缩写** - 请具体说明

### **可解释性在WiFi CSI HAR中的价值**:
- **信号分析**: 哪些频率/时间特征最重要
- **活动区分**: 不同活动的判别特征
- **故障诊断**: 模型错误的原因分析
- **部署指导**: 传感器放置优化

### **常见可解释性方法适用性**:
```
✅ 适合WiFi CSI的方法:
├── Attention Visualization (时间/频率注意力)
├── Feature Importance (SHAP/LIME)  
├── Grad-CAM (空间-时间激活)
└── Saliency Maps (信号重要性)

⚠️ 需要适配的方法:
├── 传统图像CAM → CSI时频域CAM
├── 文本注意力 → 信号时序注意力
└── 通用SHAP → CSI特定解释
```

---

## 🎨 **图表设计最佳实践**

### **1. IEEE期刊特定要求**:
- **配色**: 色盲友好 (ColorBrewer推荐)
- **标注**: 清晰的轴标签和单位
- **图例**: 简洁明了，避免重复
- **质量**: 300 DPI矢量图，支持缩放

### **2. 数据驱动的视觉层次**:
```
优先级1: 核心贡献图表
├── D4标签效率曲线 (82.1% @ 20%标签)
└── D3跨域一致性 (83% F1)

优先级2: 方法论说明图表  
├── Enhanced模型架构
└── 物理建模框架

优先级3: 支撑分析图表
├── 校准和可信度分析
└── 计算效率对比
```

### **3. 故事线设计**:
```
图表叙事逻辑:
Problem → Method → Validation → Impact

Fig1(Method) → Fig2(Architecture) → Fig3-4(Results) → Fig5(Trustworthy) → Fig6(Transfer)
```

---

## 🚀 **立即可实施的图表生成计划**

### **Phase 1: 核心实验图表** (优先级最高)
```bash
# 已准备的脚本
python3 scripts/generate_paper_figures.py --d3_csv results/metrics/summary_d3.csv --d4_csv results/metrics/summary_d4.csv

输出:
├── figure1_d3_cross_domain.pdf    # D3跨域性能对比
├── figure2_d4_label_efficiency.pdf # D4标签效率曲线  
└── figure3_model_comparison.pdf   # 模型对比分析
```

### **Phase 2: 方法论图表** (需要设计)
- Enhanced模型架构图 (手绘 → 数字化)
- Physics-guided生成框架图
- Sim2Real迁移流程图

### **Phase 3: 可解释性图表** (取决于CHAP方法)
- Attention可视化
- Feature重要性分析
- 模型决策解释

---

## 💡 **专业建议总结**

### **1. 图表驱动的撰写策略** ✅
- **先图后文**: 生成关键图表，基于视觉结果撰写文字
- **数据完整性**: 你的D3/D4实验数据完全支持高质量图表
- **故事连贯性**: 图表间的逻辑递进关系

### **2. IEEE期刊投稿优势**:
- **系统性实验**: D1-D4完整的实验协议
- **定量分析**: 82.1% F1 @ 20%标签的具体数字
- **可信评估**: ECE、校准分析等trustworthy AI要求
- **实际价值**: 80%成本降低的clear benefit

### **3. 关于CHAP的建议**:
**请确认CHAP具体指什么方法？**基于你的回答，我可以：
- 分析其在WiFi CSI HAR中的适用性  
- 设计相应的可解释性图表
- 评估是否值得在top期刊中包含

### **4. 下一步行动**:
1. **立即生成**: D3/D4核心实验图表
2. **澄清CHAP**: 确定具体的可解释性方法
3. **设计架构图**: Enhanced模型和生成框架
4. **开始Results章节**: 基于图表撰写分析

---

## 🎯 **你现在可以决定**:

1. **CHAP方法确认** - 具体指什么可解释性技术？
2. **图表优先级** - 是否先专注于D3/D4核心结果？  
3. **期刊选择** - IoTJ vs TMC，影响图表风格选择

**💡 我的推荐**: 先生成D3/D4核心图表建立Results章节骨架，然后根据CHAP方法决定是否增加可解释性分析。

你想先从哪个图表开始？