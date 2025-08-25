
# 📊 PaperA Figure 3 & 4 手工绘图详细指南

## 🎯 Figure 3: D3 Cross-Domain Generalization Performance

### 图表类型: 分组柱状图 (Grouped Bar Chart)
### IEEE IoTJ规范: 17.1cm × 10cm, 300 DPI

### 精确数据坐标:
```
X轴位置: 1, 2, 3, 4 (Enhanced, CNN, BiLSTM, Conformer-lite)
Y轴范围: 0.0 - 1.0 (Macro F1 Score)

LOSO组 (左侧柱状):
├── Enhanced: 0.830 ± 0.001 (高度83.0%, 误差棒±0.1%)
├── CNN: 0.842 ± 0.025 (高度84.2%, 误差棒±2.5%)  
├── BiLSTM: 0.803 ± 0.022 (高度80.3%, 误差棒±2.2%)
└── Conformer: 0.403 ± 0.386 (高度40.3%, 误差棒±38.6%)

LORO组 (右侧柱状):
├── Enhanced: 0.830 ± 0.001 (高度83.0%, 误差棒±0.1%) ⭐
├── CNN: 0.796 ± 0.097 (高度79.6%, 误差棒±9.7%)
├── BiLSTM: 0.789 ± 0.044 (高度78.9%, 误差棒±4.4%)
└── Conformer: 0.841 ± 0.040 (高度84.1%, 误差棒±4.0%)
```

### 颜色方案 (色盲友好):
- Enhanced: #2E86AB (深蓝色) - 突出主模型
- CNN: #E84855 (橙红色)
- BiLSTM: #3CB371 (中绿色)  
- Conformer: #DC143C (深红色)

### 设计要点:
- 柱状宽度: 0.35相对单位
- 间距: 0.1相对单位
- 误差棒: cap大小3pt, 线宽0.5pt
- 突出Enhanced: 边框加粗1.5pt黑色
- 网格: 水平网格线, 0.25pt灰色, alpha=0.3

### 标注文字:
- 标题: "Cross-Domain Generalization Performance" (12pt, 粗体)
- X轴: "Model Architecture" (10pt)
- Y轴: "Macro F1 Score" (10pt)
- 图例: "LOSO" / "LORO" (9pt)
- 数值标签: 每个柱状顶部标注 "0.830±0.001" (8pt)

## 🎯 Figure 4: D4 Sim2Real Label Efficiency Curve

### 图表类型: 效率曲线 + 关键标注
### IEEE IoTJ规范: 17.1cm × 12cm, 300 DPI

### 精确数据坐标:
```
曲线数据点:
Point 1: (1.0%, 0.455) ± 0.050
Point 2: (5.0%, 0.780) ± 0.016  
Point 3: (10.0%, 0.730) ± 0.104
Point 4: (20.0%, 0.821) ± 0.003  ⭐ KEY POINT
Point 5: (100.0%, 0.833) ± 0.000

参考线:
- 目标线: y=0.80 (红色虚线, 1.5pt)
- 理想线: y=0.90 (橙色点线, 1pt)
- Zero-shot基线: y=0.151 (灰色实线, 1pt)
```

### 设计元素:
- 主曲线: #2E86AB (深蓝), 2.5pt线宽, 圆点标记8pt
- 误差带: 半透明蓝色填充 (alpha=0.3)
- 效率区域: 0-20%标签背景浅绿色 (alpha=0.2)

### 关键标注 (20%, 0.821坐标):
```
标注框位置: (35%, 0.87)
箭头: 红色实线, 1.5pt, 指向(20%, 0.821)
文本框: 
├── 背景: 浅黄色 (#FFFACD)
├── 边框: 红色 (#FF6B6B), 1pt
├── 文字: "Key Achievement:
82.1% F1 @ 20% Labels"
└── 字体: Times New Roman Bold, 10pt
```

### 标注文字:
- 标题: "Sim2Real Label Efficiency Breakthrough" (12pt, 粗体)
- X轴: "Label Ratio (%)" (10pt)
- Y轴: "Macro F1 Score" (10pt)
- 数据点标签: 每点上方标注F1值 (8pt, 蓝色)

## 🎨 绘图软件操作指南

### MATLAB操作步骤:
1. 运行 plot_method4_matlab.m
2. 自动生成IEEE IoTJ规范的PDF
3. 检查输出: figure3_*_matlab.pdf, figure4_*_matlab.pdf

### Origin操作步骤:
1. 导入 figure*_origin_data.txt
2. 运行 figures_origin_pro.ogs 脚本
3. 手动调整标注和颜色

### Adobe Illustrator操作步骤:
1. 导入生成的SVG或PDF文件
2. 转换为矢量路径
3. 调整字体为Times New Roman
4. 确保线条宽度符合IEEE标准
5. 导出为300 DPI PDF

### Excel/LibreOffice操作步骤:
1. 按照 excel_plotting_guide.txt 操作
2. 使用提供的颜色代码
3. 手动添加误差棒和标注

## 🔍 数据验证检查清单

### Figure 3验证:
- [ ] Enhanced LOSO = Enhanced LORO = 83.0%
- [ ] Enhanced误差棒最小 (±0.001)
- [ ] CNN在LOSO中最高但变异大
- [ ] Conformer在LOSO中不稳定 (CV>90%)

### Figure 4验证:
- [ ] 20%标签点 = 82.1% F1 (关键成果)
- [ ] 曲线趋势: 1%→5%大幅提升, 20%→100%平缓
- [ ] 目标线80%被20%点超越
- [ ] 误差棒在20%点最小 (±0.003)

## 📋 IEEE IoTJ投稿检查:
- [ ] 分辨率: 300 DPI ✓
- [ ] 尺寸: 符合期刊要求 ✓
- [ ] 字体: Times New Roman ✓
- [ ] 颜色: 色盲友好方案 ✓
- [ ] 图注: <300字清晰描述 ✓
- [ ] 文件格式: PDF/EPS矢量 ✓
