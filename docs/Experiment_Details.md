# 实验详细内容 (D1-D6)

## **D1: 合成数据InD验证 (容量匹配)**

### **实验目标**
验证合成数据生成器的基本有效性，确保模型容量匹配

### **实验内容**
- **模型**: enhanced, cnn, bilstm, conformer_lite (容量匹配±10%)
- **配置**: 10个随机种子 (0-9)
- **难度**: 中等难度
- **指标**: Macro-F1, 参数数量, 计算复杂度

### **验收标准**
- 所有模型容量差异≤10%
- Enhanced模型在合成数据上表现最佳
- 生成结果可重现且稳定

### **输出**
- `results/metrics/summary_cpu.csv`
- 容量匹配验证报告

---

## **D2: 合成数据鲁棒性验证 (540配置)**

### **实验目标**
通过系统性的参数变化验证合成数据生成器的鲁棒性和可控性

### **实验内容**
- **配置总数**: 540种配置
- **变化参数**: 噪声水平、类别重叠度、信道衰落、谐波干扰
- **难度等级**: 低、中、高三个等级
- **随机种子**: 每种配置8个独立随机种子
- **模型对比**: Enhanced vs CNN vs BiLSTM vs Conformer-lite

### **关键参数设置**
```python
D2_Parameters = {
    "class_overlap": [0.3, 0.5, 0.7, 0.8, 0.9],
    "label_noise_prob": [0.0, 0.05, 0.1, 0.15, 0.2],
    "env_burst_rate": [0.0, 0.1, 0.2, 0.3, 0.4],
    "gain_drift_std": [0.0, 0.3, 0.6, 0.9, 1.2],
    "sc_corr_rho": [0.0, 0.3, 0.5, 0.7, 0.9]
}
```

### **验收标准**
- **覆盖率**: 每个配置×≥3种子完成
- **一致性**: Enhanced模型平均性能≥基线5% macro_f1
- **校准**: 带λ的模型ECE比不带λ低≥5%
- **报告**: 生成消融条形图和可靠性图

### **输出**
- `results_gpu/d2/*.json` (540个结果文件)
- `results/metrics/summary_d2.csv`
- 消融分析图表

---

## **D3: 跨域泛化评估 (LOSO/LORO)**

### **实验目标**
验证模型在真实数据上的跨受试者和跨环境泛化能力

### **实验内容**
- **数据集**: benchmarks/WiFi-CSI-Sensing-Benchmark-main
- **协议**: LOSO (Leave-One-Subject-Out) & LORO (Leave-One-Room-Out)
- **模型**: enhanced, cnn, bilstm, conformer_lite
- **种子**: 0, 1, 2, 3, 4
- **配置总数**: ~200-400个实验

### **实验设计**
```python
D3_Protocols = {
    "LOSO": "每个受试者作为测试集，其他受试者作为训练集",
    "LORO": "每个房间作为测试集，其他房间作为训练集"
}
```

### **验收标准**
- **性能**: Falling F1 ≥ 0.75 (跨域平均), Macro F1 ≥ 0.80
- **校准**: 温度缩放后ECE ≤ 0.15
- **鲁棒性**: Enhanced模型比基线≥5% Falling F1
- **覆盖率**: ≥90%的LOSO/LORO折完成
- **统计**: 所有主要指标的Bootstrap 95% CI

### **输出**
- `results/d3/loso/*.json`, `results/d3/loro/*.json`
- `results/metrics/summary_d3.csv`
- 跨域性能分析报告

---

## **D4: Sim2Real标签效率评估 (56配置)**

### **实验目标**
评估合成到真实数据的迁移学习效率，量化标签需求

### **实验内容**
- **源数据**: D2预训练模型 (enhanced, cnn, bilstm, conformer_lite)
- **目标数据**: 真实WiFi CSI基准数据
- **标签比例**: [1%, 5%, 10%, 15%, 20%, 50%, 100%]
- **迁移方法**: Zero-shot, Fine-tuning, Linear Probing, Temperature Scaling
- **配置总数**: 4模型 × 7比例 × 4方法 × 5种子 = 560配置

### **实验设计**
```python
D4_Transfer_Methods = {
    "zero_shot": "直接评估合成训练模型在真实数据上的表现",
    "linear_probe": "冻结骨干网络，只训练分类器",
    "fine_tune": "端到端微调，低学习率",
    "temp_scale": "仅校准适应，小真实数据子集"
}
```

### **验收标准**
- **标签效率**: 10-20%真实标签达到≥90%全监督性能
- **Zero-shot基线**: Falling F1 ≥ 0.60, Macro F1 ≥ 0.70
- **迁移增益**: Fine-tuning比zero-shot提升≥15% Falling F1
- **校准**: Sim2Real ECE差距≤0.10 (温度缩放后)
- **覆盖率**: 所有标签效率点≥3个成功种子

### **输出**
- `results/d4/sim2real/*.json`
- `results/metrics/summary_d4.csv`
- 标签效率曲线和迁移分析报告

---

## **D5: 消融研究与机制分析**

### **实验目标**
识别Enhanced模型组件和训练策略对性能和校准的贡献

### **实验内容**
- **架构变体**: Enhanced (完整), Enhanced−SE, Enhanced−Attn, Enhanced−SE−Attn
- **训练策略**: 带/不带logit-L2正则化, 输入归一化对比
- **真实数据验证**: D4比例1%/5%的linear_probe vs fine_tune对比

### **消融变体**
```python
D5_Variants = {
    "Enhanced (full)": "完整模型",
    "Enhanced−SE": "移除Squeeze-Excitation模块",
    "Enhanced−Attn": "移除时序注意力机制",
    "Enhanced−SE−Attn": "移除SE和注意力 (≈容量匹配CNN)",
    "Enhanced−L2": "移除logit L2正则化",
    "Enhanced+CenterNorm": "使用中心化归一化"
}
```

### **验收标准**
- **覆盖率**: 每个消融变体×≥3种子完成
- **一致性**: Enhanced (完整)平均性能≥简化变体5% macro_f1
- **校准**: 带λ的模型ECE比不带λ低≥5%
- **真实验证**: D4比例1%/5%上fine_tune比linear_probe≥5% macro_f1
- **报告**: 消融条形图、校准图、可靠性图

### **输出**
- `results/ablation/*.json`
- `plots/ablation/*.pdf`
- `docs/D5_Acceptance_Report.md`

---

## **D6: 鲁棒性、效率与总结**

### **实验目标**
总结跨域鲁棒性(D3)和Sim2Real效率(D4)，形成发表就绪的证据

### **实验内容**
- **鲁棒性**: LOSO/LORO快速集 (Enhanced + 一个基线)
- **效率**: 参数、FLOPs、延迟 (CPU/GPU) 对比
- **可信度**: 关键比例(1%, 5%, 100%)的校准总结和可靠性图

### **最小运行**
```python
D6_Minimal_Runs = {
    "D3": "Enhanced在采样折上(≥50%覆盖率) × seeds=[0,1]",
    "D4": "使用最终标签效率运行 + 额外种子用于CI",
    "Efficiency": "Enhanced vs 基线 (±10%容量匹配)"
}
```

### **验收标准**
- **鲁棒性**: LOSO/LORO聚合macro_f1 ≥ 计划阈值; Enhanced ≥ 基线5%
- **标签效率**: 10-20%标签达到≥90%全监督性能 (macro_f1)
- **校准**: 关键比例校准后ECE ≤ 0.15
- **效率**: Enhanced vs 基线参数/FLOPs/延迟报告，明确优势或对等
- **报告完整性**: 所有图表/表格生成; 文档更新

### **输出**
- `results/metrics/*.csv` 和 `plots/*.pdf`
- `docs/D6_Final_Summary.md`

