# 📊 IEEE IoTJ Figure 3 & 4 绘图说明文档

## 🎯 概述
本目录包含为IEEE IoTJ期刊论文生成Figure 3和Figure 4的完整解决方案，支持多种绘图软件和工作流程。

## 📁 文件结构
```
plots/
├── README_plotting_instructions.md          # 本说明文档
├── DETAILED_PLOTTING_GUIDE.md              # 详细绘图指南（用户提供）
├── figure3_cross_domain_data.csv           # Figure 3数据文件
├── figure4_sim2real_data.csv               # Figure 4数据文件
├── plot_method4_matlab.m                   # MATLAB主绘图脚本
├── generate_figures_python.py              # Python备用脚本
├── figure3_origin_data.txt                 # Origin Pro数据格式
├── figure4_origin_data.txt                 # Origin Pro数据格式  
├── figures_origin_pro.ogs                  # Origin Pro脚本
└── excel_plotting_guide.txt                # Excel/LibreOffice操作指南
```

## 🚀 快速开始

### 方法1: MATLAB (推荐)
```matlab
cd plots
run('plot_method4_matlab.m')
```
**输出**: `figure3_cross_domain_matlab.pdf`, `figure4_sim2real_matlab.pdf`

### 方法2: Python (备用)  
```bash
cd plots
python generate_figures_python.py
```
**输出**: `figure3_cross_domain_python.pdf`, `figure4_sim2real_python.pdf`

### 方法3: Origin Pro
1. 打开Origin Pro
2. 运行脚本: `figures_origin_pro.ogs`  
3. 导入数据: `figure3_origin_data.txt`, `figure4_origin_data.txt`

### 方法4: Excel/LibreOffice
参考 `excel_plotting_guide.txt` 详细步骤操作

## 📋 图表规范 - IEEE IoTJ标准

### Figure 3: Cross-Domain Generalization Performance
- **类型**: 分组柱状图 (Grouped Bar Chart)
- **尺寸**: 17.1cm × 10cm, 300 DPI
- **数据**: LOSO/LORO跨域泛化性能对比
- **关键点**: Enhanced模型LOSO=LORO=83.0%，误差最小

### Figure 4: Sim2Real Label Efficiency Curve  
- **类型**: 效率曲线 + 关键标注
- **尺寸**: 17.1cm × 12cm, 300 DPI
- **数据**: 标签效率分析，1%-100%标签比例
- **关键点**: 20%标签达到82.1% F1，超越80%目标

## 🎨 设计要素

### 颜色方案 (色盲友好)
- Enhanced: `#2E86AB` (深蓝色) - 主模型
- CNN: `#E84855` (橙红色)
- BiLSTM: `#3CB371` (中绿色)  
- Conformer: `#DC143C` (深红色)

### 字体规范
- **主体**: Times New Roman
- **标题**: 12pt, 加粗
- **坐标轴标签**: 10pt, 常规
- **数据标签**: 8pt
- **图例**: 9pt

### 误差棒设置
- **帽子大小**: 3pt
- **线宽**: 0.5pt
- **颜色**: 黑色

## ✅ 质量检查清单

### Figure 3验证项目:
- [ ] Enhanced LOSO = Enhanced LORO = 83.0%
- [ ] Enhanced误差棒最小 (±0.001)
- [ ] CNN在LOSO中最高但变异大  
- [ ] Conformer在LOSO中不稳定 (CV>90%)
- [ ] 颜色方案色盲友好
- [ ] Enhanced模型边框加粗突出

### Figure 4验证项目:
- [ ] 20%标签点 = 82.1% F1 (关键成果)
- [ ] 曲线趋势: 1%→5%大幅提升, 20%→100%平缓
- [ ] 目标线80%被20%点超越
- [ ] 误差棒在20%点最小 (±0.003)
- [ ] 关键标注框清晰可见
- [ ] 效率区域背景适当

### IEEE IoTJ规范检查:
- [ ] 分辨率: 300 DPI
- [ ] 尺寸符合期刊要求
- [ ] 字体: Times New Roman  
- [ ] 文件格式: PDF/EPS矢量
- [ ] 线条宽度符合标准
- [ ] 颜色对比度充足

## 🔧 故障排除

### 常见问题:
1. **MATLAB字体问题**: 确保系统安装Times New Roman字体
2. **Python依赖缺失**: `pip install matplotlib pandas seaborn numpy`
3. **Origin脚本错误**: 检查数据文件路径和格式
4. **输出模糊**: 验证DPI设置为300
5. **颜色不准**: 使用精确的十六进制色码

### 兼容性:
- **MATLAB**: R2018b或更高版本
- **Python**: 3.7+, matplotlib 3.0+
- **Origin Pro**: 2018或更高版本
- **Excel**: 2016或更高版本，LibreOffice 6.0+

## 📞 技术支持

如遇到问题，请检查:
1. 数据文件完整性和格式
2. 软件版本兼容性
3. 系统字体安装状态
4. 输出目录权限

---
**生成日期**: 2025年1月
**适用期刊**: IEEE IoTJ, IEEE TMC, IMWUT等顶级期刊
**论文主题**: WiFi CSI Fall Detection, Trustworthy Evaluation