# 📊 当前实验设计与文献基线对比分析

## 🔍 当前实验设计概览

基于实际代码分析（`scripts/d2_spec.json`, `src/models.py`, `src/train_eval.py`, `src/data_synth.py`）：

### 🏗️ **当前实验架构**

#### 1. 模型配置 (D2实验)
```json
{
  "models": ["enhanced", "cnn", "bilstm", "conformer_lite"],
  "seeds": [0, 1, 2, 3, 4],
  "fixed": {
    "difficulty": "hard",
    "epochs": 100,
    "batch": 768,
    "amp": true
  },
  "grid": {
    "class_overlap": [0.0, 0.4, 0.8],
    "label_noise_prob": [0.0, 0.05, 0.1],
    "env_burst_rate": [0.0, 0.1, 0.2]
  }
}
```

#### 2. 模型实现分析
| 模型名 | 实际架构 | 创新点 | 参数规模 |
|-------|----------|--------|----------|
| **enhanced** | EnhancedNet: CNN+SE+TemporalSelfAttention | 挤压激励+时序注意力 | 中等 (~320通道) |
| **cnn** | SimpleCNN: 2层CNN+MaxPool | 基础卷积网络 | 小 (~48隐藏) |
| **bilstm** | BiLSTM: 双向LSTM+FC | 时序建模 | 中等 (~256隐藏) |
| **conformer_lite** | ConformerLite: Conv+Attention | 类Transformer架构 | 中等 (~192维度) |

#### 3. 合成数据生成器特点
- **物理可控参数**: class_overlap, label_noise_prob, env_burst_rate, gain_drift_std
- **语义类别**: 8类CSI跌倒检测场景 ("Normal Walking", "Epileptic Fall"等)
- **难度因子**: 可控的类间重叠和环境干扰
- **缓存机制**: 多层缓存系统优化数据生成

#### 4. 评估指标
- **准确率**: macro_f1, falling_f1
- **校准指标**: ECE, Brier score, NLL
- **鲁棒性**: mutual_misclass (类间混淆)
- **可信度**: 温度标定, 可靠性曲线

---

## 🔄 与文献基线的对比分析

### ✅ **我们的优势**

| 维度 | 文献基线现状 | 我们的设计 | 优势程度 |
|------|--------------|------------|----------|
| **评估框架** | 简单accuracy评估 | 物理可控合成+多指标评估 | 🔥🔥🔥🔥🔥 |
| **校准评估** | 很少评估ECE/Brier | 完整ECE/Brier/可靠性曲线 | 🔥🔥🔥🔥🔥 |
| **可控实验** | 依赖真实数据收集 | 可控难度因子分析 | 🔥🔥🔥🔥🔥 |
| **统计严格性** | 很少统计检验 | 5种子+Bootstrap CI | 🔥🔥🔥🔥 |

### 🔴 **需要改进的方面**

| 维度 | 文献基线表现 | 我们的现状 | 改进必要性 |
|------|--------------|------------|------------|
| **跨域协议** | ReWiS: 3环境验证 | 仅合成数据 | 🚨🚨🚨🚨🚨 |
| **少样本学习** | FewSense: 5-shot性能 | 未实现 | 🚨🚨🚨🚨 |
| **自监督预训练** | AutoFi: 几何自监督 | 未实现 | 🚨🚨🚨 |
| **真实数据验证** | 所有方法都有真实验证 | 主要是合成 | 🚨🚨🚨🚨 |
| **计算效率** | CLNet: 24.1%计算减少 | 未优化 | 🚨🚨 |

---

## 🛠️ **实验调节建议**

### 🏆 **高优先级调节 (必须实现)**

#### 1. **添加真实数据验证**
```python
# 建议添加到d2_spec.json
"real_data_validation": {
    "datasets": ["SignFi", "Widar", "UT-HAR"],  # 如有可用
    "protocols": ["LOSO", "LORO"],
    "sim2real_ratios": [0.1, 0.2, 0.5, 1.0]
}
```

#### 2. **实现少样本学习基线**
```python
# 新增模型: src/models.py
class FewShotNet(nn.Module):
    """基于ReWiS的简化少样本学习框架"""
    def __init__(self, backbone, num_classes, n_support=5):
        super().__init__()
        self.backbone = backbone  # 可复用enhanced/cnn
        self.few_shot_head = PrototypicalHead(n_support)
        
# 新增到build_model:
elif name == "fewshot":
    return FewShotNet(EnhancedNet(...), num_classes)
```

#### 3. **严格跨域协议**
```python
# 修改train_eval.py，添加LOSO/LORO支持
def cross_domain_eval(model, data_loader, protocol="LOSO"):
    if protocol == "LOSO":
        return leave_one_subject_out(model, data_loader)
    elif protocol == "LORO": 
        return leave_one_room_out(model, data_loader)
```

### 🎯 **中优先级调节 (建议实现)**

#### 4. **自监督预训练模块**
```python
# 参考AutoFi，添加几何变换预训练
class GeometricSSL(nn.Module):
    """几何自监督学习模块"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.ssl_head = nn.Linear(backbone.embed_dim, 4)  # 旋转任务
        
    def forward_ssl(self, x, transformation_labels):
        features = self.backbone.extract_features(x)
        ssl_logits = self.ssl_head(features)
        return ssl_logits
```

#### 5. **容量匹配验证**
```python
# 添加参数计算和匹配
def ensure_capacity_matching(models, tolerance=0.1):
    """确保所有模型参数在±10%范围内"""
    param_counts = {name: sum(p.numel() for p in model.parameters()) 
                   for name, model in models.items()}
    # 验证并报告参数差异
```

#### 6. **计算效率优化**
```python
# 参考CLNet，优化模型效率
class EfficientEnhanced(EnhancedNet):
    """计算效率优化版本"""
    def __init__(self, *args, efficiency_mode=True, **kwargs):
        super().__init__(*args, **kwargs)
        if efficiency_mode:
            self.apply_efficiency_optimizations()
```

### 📊 **低优先级调节 (可选实现)**

#### 7. **多模态扩展 (如有视觉数据)**
```python
# 参考GaitFi设计
class MultimodalNet(nn.Module):
    def __init__(self, wifi_backbone, vision_backbone):
        super().__init__()
        self.wifi_branch = wifi_backbone
        self.vision_branch = vision_backbone  # 如有可用
        self.fusion = CrossModalFusion()
```

---

## 🚀 **实施优先级和时间规划**

### **第一阶段 (2周内，关键基线)**
1. ✅ **添加真实数据支持**
   - 实现LOSO/LORO协议
   - 添加sim2real标签效率分析
   
2. ✅ **FewShot基线实现**
   - 简化版prototypical networks
   - 与ReWiS性能对比

### **第二阶段 (1个月内，增强对比)**
3. ✅ **自监督预训练**
   - 几何变换SSL模块
   - 与AutoFi方法对比
   
4. ✅ **计算效率优化**
   - 模型轻量化版本
   - 推理时间和内存分析

### **第三阶段 (可选扩展)**
5. 🔄 **多模态支持** (如有数据)
6. 🔄 **压缩感知** (部署导向)

---

## 📝 **论文框架优化建议**

### **基于文献分析的章节重组**

#### **当前论文结构问题**:
- Related Work部分过于简单
- 缺乏与SOTA方法的直接对比
- 实验部分未突出跨域评估

#### **建议的新框架** (模仿高水平论文):

```latex
1. Introduction
   - 问题动机: WiFi CSI跌倒检测的信任挑战
   - 现有方法局限: 引用ReWiS, AutoFi, FewSense的不足
   - 我们的贡献: 4个核心创新点

2. Related Work  
   2.1 WiFi CSI-based HAR [ReWiS, GaitFi, EfficientFi]
   2.2 Cross-Domain Generalization [FewSense, AirFi] 
   2.3 Trustworthy ML in Sensing [指出校准评估缺失]
   2.4 Synthetic Data and Sim2Real [与计算机视觉对比]

3. Method
   3.1 Physics-Guided Synthetic Generator
   3.2 Enhanced Model with Confidence Prior  
   3.3 Evaluation Protocols (LOSO/LORO/Calibration)

4. Experiments
   4.1 Synthetic InD Capacity-Matched Validation (D1)
   4.2 Synthetic Controllable Analysis (D2) ← 现有强项
   4.3 Real-World LOSO/LORO Results ← 新增必需
   4.4 Sim2Real Label Efficiency ← 新增必需
   4.5 Few-Shot Learning Comparison ← 新增建议
```

### **重点章节草稿优先级**
1. **Abstract & Introduction**: 可立即完成
2. **Method (3.1, 3.2)**: 基于现有代码完成  
3. **Experiments (4.2)**: D2分析已有数据
4. **Related Work**: 基于文献分析完成

---

## ⚖️ **风险评估和缓解策略**

### **高风险项**
- **真实数据获取**: 如无法获得，强化合成数据的物理可信度分析
- **少样本实现复杂度**: 可实现简化版本，重点对比概念

### **缓解策略**
- **渐进实现**: 优先完成能立即展示的部分(D2分析)
- **替代方案**: 如某些基线难实现，通过理论分析和文献对比体现创新点
- **重点突出**: 强调我们独有的物理可控评估框架价值

---

## 🎯 **立即行动项**

### **今日可完成**
1. ✅ 更新Related Work (基于9篇文献)
2. ✅ 完善Abstract突出创新点
3. ✅ 写出Method 3.1-3.2草稿

### **本周目标**  
1. 🔄 实现FewShot基线简化版
2. 🔄 添加LOSO/LORO协议支持
3. 🔄 完成实验4.2章节 (D2分析)

**基于真实代码的具体调节建议已制定，优先级明确，可立即开始实施。**