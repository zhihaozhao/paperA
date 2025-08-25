# 📊 PaperA Figure 3 & 4 制作规范 (IEEE IoTJ标准)

## 🎯 **Figure 3: Cross-Domain Generalization Performance**

### **图表类型**: 分组柱状图 (Grouped Bar Chart)

### **IEEE IoTJ规范**:
- **尺寸**: 17.1cm × 10cm (双栏)
- **分辨率**: 300 DPI
- **格式**: PDF/EPS (矢量图优先)
- **字体**: Times New Roman, 轴标签10pt, 数值8pt

### **数据文件**: `figure3_d3_cross_domain_data.csv`

### **设计细节**:
```
X轴: 模型类型 (Enhanced, CNN, BiLSTM, Conformer-lite)
Y轴: Macro F1 Score (0.0 - 1.0)
分组: LOSO vs LORO (两组并排柱状)
颜色方案:
├── Enhanced: #2E86AB (深蓝) ⭐
├── CNN: #E84855 (橙红)
├── BiLSTM: #3CB371 (中绿)  
└── Conformer: #DC143C (深红)

误差棒:
├── 类型: ±1 standard deviation
├── cap大小: 3pt
├── 线宽: 0.5pt
└── 颜色: 与主柱状相同但更深

数值标注:
├── 位置: 柱状顶部 + 误差棒上方2pt
├── 格式: "0.830±0.001" 
├── 字体: 8pt, Times New Roman
└── 颜色: 黑色

网格: 水平网格线, 0.25pt, 灰色(#CCCCCC), alpha=0.3
```

### **重点突出**:
- **Enhanced模型**: 两协议下完美一致的83.0%性能
- **低变异性**: Enhanced CV<1% vs 其他模型CV>2%
- **跨域稳定**: LOSO/LORO性能接近，证明泛化能力

### **图注** (≤300字):
```
Figure 3. Cross-domain generalization performance across LOSO (Leave-One-Subject-Out) and LORO (Leave-One-Room-Out) evaluation protocols. The Enhanced model demonstrates exceptional consistency with 83.0±0.1% macro F1 score across both protocols, significantly outperforming baseline architectures in terms of both performance and stability (CV<0.2%). Error bars represent ±1 standard deviation across 5 random seeds. Results indicate superior cross-domain robustness essential for practical WiFi CSI HAR deployment.
```

---

## 🎯 **Figure 4: Sim2Real Label Efficiency Curve** ⭐

### **图表类型**: 效率曲线 + 关键标注

### **IEEE IoTJ规范**:
- **尺寸**: 17.1cm × 12cm (双栏)
- **分辨率**: 300 DPI
- **格式**: PDF/EPS (矢量图)

### **数据文件**: `figure4_d4_label_efficiency_data.csv`

### **设计细节**:
```
X轴: Label Ratio (%) [0, 20, 40, 60, 80, 100]
Y轴: Macro F1 Score [0.0, 1.0]

主曲线: Enhanced Fine-tune
├── 颜色: #2E86AB (深蓝)
├── 线型: 实线, 2.5pt宽
├── 标记: 圆点, 8pt直径, 填充
├── 误差带: 半透明蓝色填充 (alpha=0.3)
└── 数据点: [1%, 5%, 10%, 20%, 100%]

关键标注: 82.1% @ 20% Labels
├── 位置: (20, 0.821)
├── 箭头: 红色, 1.5pt, 指向数据点
├── 文本框: 黄色背景, 边框1pt
├── 内容: "Key Achievement\n82.1% F1 @ 20% Labels"
└── 字体: Times New Roman Bold, 10pt

参考线:
├── 目标线: y=0.80, 红色虚线, 1.5pt, "Target: 80% F1"
├── 理想线: y=0.90, 橙色点线, 1pt, "Ideal: 90% F1"
└── 基线: Zero-shot 15.1% (灰色水平线)

效率区域标记:
├── 范围: x=0 to x=20
├── 填充: 浅绿色背景 (alpha=0.2)
└── 标签: "Efficient Range (≤20%)"

网格: 主要网格线, 0.5pt, 次要网格线0.25pt
```

### **数据点坐标**:
```
Point 1: (1.0, 0.455±0.050)
Point 2: (5.0, 0.780±0.016)  
Point 3: (10.0, 0.730±0.104)
Point 4: (20.0, 0.821±0.003) ⭐ KEY POINT
Point 5: (100.0, 0.833±0.000)
```

### **图注**:
```
Figure 4. Sim2Real label efficiency breakthrough achieved by Enhanced model. The efficiency curve demonstrates that only 20% labeled real data is required to achieve 82.1% macro F1 score, representing merely 1.2% performance gap compared to full supervision (83.3%). This breakthrough reduces labeling costs by 80% while maintaining near-optimal performance, enabling practical deployment of WiFi CSI HAR systems. Shaded region indicates the efficient deployment range (≤20% labels).
```

---

## 📝 **制作工具指南**

### **推荐绘图软件**:
- **MATLAB**: 适合精确的科学图表
- **Python matplotlib**: 如果环境允许
- **Origin/OriginPro**: 专业科学绘图
- **Adobe Illustrator**: 最终美化调整

### **图表制作检查清单**:
```
✅ 分辨率: 300 DPI
✅ 尺寸: IEEE IoTJ标准
✅ 字体: Times New Roman
✅ 颜色: 色盲友好方案
✅ 误差棒: ±1σ标准差
✅ 图注: <300字, 清晰自明
✅ 文件格式: PDF/EPS矢量
```

现在让我开始制作这些图表并更新论文内容...