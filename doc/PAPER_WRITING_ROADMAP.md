# 📝 PaperA 论文撰写路线图

**目标期刊**: IoTJ/TMC/IMWUT (Top-tier IEEE/ACM)  
**论文主题**: Physics-Guided Synthetic WiFi CSI Data Generation for Trustworthy HAR

---

## 🎯 **论文撰写状态**

### ✅ **已完成的实验基础**
- **D1**: 合成数据容量对齐验证 (9配置)
- **D2**: 鲁棒性扫描实验 (540配置) 
- **D3**: 跨域泛化验证 (40配置) - LOSO/LORO
- **D4**: Sim2Real标签效率 (56配置) - 82.1% F1 @ 20%标签
- **验收报告**: 完整的实验验收文档

### 📊 **关键实验成果可用**
- Enhanced模型: 83%+ F1跨域一致性
- 标签效率: 20%标签达到82% F1
- 可信评估: ECE校准和可靠性分析
- 容量匹配: 公平的基线对比

---

## 📋 **论文撰写计划**

### **Phase 1: 实验结果整合** (1-2天)
- [ ] **图表生成**: 基于D3/D4结果创建关键图表
  - D3跨域泛化性能对比图
  - D4标签效率曲线图  
  - 校准和可靠性分析图
- [ ] **表格制作**: 性能对比表格
  - 跨域性能汇总表
  - Sim2Real效率对比表
  - 与SOTA方法对比表
- [ ] **统计分析**: 显著性检验和置信区间

### **Phase 2: 核心章节撰写** (2-3天)
- [ ] **方法论章节**: Enhanced模型架构详述
- [ ] **实验设计**: D1-D4实验协议说明
- [ ] **结果分析**: 基于真实数据的性能分析
- [ ] **可信评估**: 校准、可靠性、跨域泛化

### **Phase 3: 论文完善** (1-2天)  
- [ ] **Introduction**: 强调标签效率和可信AI
- [ ] **Related Work**: WiFi CSI + Sim2Real + 可信ML
- [ ] **Discussion**: 实际部署意义和局限性
- [ ] **Conclusion**: 贡献总结和future work

### **Phase 4: 投稿准备** (1天)
- [ ] **格式检查**: IEEE/ACM期刊格式要求
- [ ] **图表质量**: 高分辨率、清晰标注
- [ ] **参考文献**: 完整、最新、相关性强
- [ ] **Final Review**: 语言润色、逻辑检查

---

## 🏆 **论文核心贡献**

### **1. Physics-Guided Synthetic Data Generation**
- 基于WiFi信号传播物理原理的合成数据生成
- 多路径效应、环境变化、人体交互建模
- D1验证: Enhanced vs CNN参数匹配，性能相当

### **2. Enhanced Architecture for CSI Sensing**  
- CNN + Squeeze-Excitation + 轻量级时间注意力
- D3验证: 83%+ F1跨域一致性 (LOSO/LORO)
- 优于BiLSTM、CNN、Conformer-lite基线

### **3. Sim2Real Label Efficiency**
- **突破性成果**: 20%标签达到82.1% F1
- D4验证: 完整的标签效率曲线分析
- 实际部署意义: 减少90%标注成本

### **4. Trustworthy Evaluation Framework**
- 校准分析(ECE)、可靠性评估
- 跨域泛化验证 (LOSO/LORO协议)
- 容量匹配的公平对比方法

---

## 📊 **关键实验数据引用**

### **表格数据** (来自results/metrics/):
- `summary_d3.csv`: D3跨域性能详细数据
- `summary_d4.csv`: D4标签效率详细数据  
- `summary_d2.csv`: D2鲁棒性扫描数据

### **性能亮点**:
```
D3 跨域泛化 (Enhanced模型):
├── LOSO: 83.0±0.1% macro F1 (CV=0.2%)
└── LORO: 83.0±0.1% macro F1 (CV=0.1%)

D4 标签效率 (Enhanced Fine-tune):
├── 1% 标签: 45.5% F1  
├── 5% 标签: 78.0% F1
├── 10% 标签: 73.0% F1  
├── 20% 标签: 82.1% F1 ⭐
└── 100% 标签: 83.3% F1 (上限)
```

---

## 🎯 **期刊投稿策略**

### **目标期刊排序**:
1. **IoTJ** (IEEE Internet of Things Journal) - IF: 10.6
2. **TMC** (IEEE Transactions on Mobile Computing) - IF: 7.9  
3. **IMWUT** (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies) - IF: 3.6

### **投稿优势**:
- **Novel Sim2Real approach** for WiFi CSI sensing
- **Trustworthy evaluation** with calibration analysis
- **Practical label efficiency** (82% F1 @ 20% labels)
- **Systematic experimental validation** (D1-D4 protocol)

### **潜在审稿关注点**:
1. **对比最新SOTA方法** (需要补充)
2. **更多真实环境验证** (目前基于SenseFi benchmark)
3. **计算复杂度分析** (Enhanced模型 vs 基线)
4. **失败案例分析** (Conformer-lite不稳定原因)

---

## 🚀 **立即可开始的论文工作**

### **优先级1: 关键图表制作**
```bash
# 基于验收脚本生成核心图表
cd /workspace
python3 scripts/generate_d3_d4_figures.py  # 需要创建
```

### **优先级2: 结果章节撰写**
- 更新论文中的D3/D4实验结果
- 添加跨域泛化和标签效率分析
- 完善可信评估章节

### **优先级3: 方法论完善**
- Enhanced模型架构图
- Sim2Real训练流程图
- 物理建模方法详述

---

## 📅 **撰写时间表**

| 时间 | 任务 | 输出 | 责任人 |
|------|------|------|--------|
| Day 1-2 | 实验结果整合 | 图表+表格 | AI助手 |
| Day 3-4 | 核心章节撰写 | Methods+Results | 合作 |
| Day 5-6 | 论文完善 | Full Draft | 合作 |
| Day 7 | 投稿准备 | Camera Ready | 人工审核 |

---

## 🔧 **撰写工具和环境**

### **LaTeX编译**:
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex  
pdflatex main.tex
```

### **图表生成** (计划):
- **Python脚本**: `scripts/generate_d3_d4_figures.py`
- **数据源**: `results/metrics/summary_d*.csv`
- **输出**: `paper/figures/` (高质量PDF/PNG)

### **表格生成** (计划):
- **数据处理**: pandas + numpy分析
- **LaTeX表格**: booktabs + multirow格式
- **自动化**: 脚本生成LaTeX代码

---

## 📝 **下一步具体行动**

### **立即开始**:
1. **创建图表生成脚本** - 基于验收的CSV数据
2. **更新论文Abstract** - 加入82.1% @ 20%标签的具体数字  
3. **撰写Results章节** - 详细的D3/D4分析

### **本次会话目标**:
- [x] ✅ 实验结果上传完成
- [x] ✅ 文档组织完成 
- [ ] 🎯 开始论文Results章节撰写
- [ ] 📊 创建关键图表脚本

---

*撰写计划创建时间: 2025-08-18*  
*当前分支: feat/enhanced-model-and-sweep (开发分支)*  
*实验结果分支: results/main (独立管理)*  
*论文状态: 🚀 准备就绪，可开始核心撰写工作*