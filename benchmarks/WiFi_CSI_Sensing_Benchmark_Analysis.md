# 🚀 WiFi-CSI-Sensing-Benchmark 集成分析报告

## 📋 **Benchmark概览**

- **GitHub仓库**: https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
- **类型**: PyTorch-based WiFi CSI人体感知评估框架
- **作者**: xyanchen等
- **特点**: 多模型、多数据集、有监督+无监督学习

---

## 🎯 **直接可用的实验**

### **1. 有监督学习基准测试**

#### **可用数据集 (4个)**:
- **UT-HAR**: 行为识别数据集
- **NTU-Fi-HumanID**: 人员识别数据集  
- **NTU-Fi_HAR**: 行为识别数据集
- **Widar**: 手势识别数据集

#### **可用模型 (11个)**:
```bash
MLP, LeNet, ResNet18, ResNet50, ResNet101, 
RNN, GRU, LSTM, BiLSTM, CNN+GRU, ViT
```

#### **基准测试命令**:
```bash
# 示例：ResNet18在NTU-Fi_HAR上
python run.py --model ResNet18 --dataset NTU-Fi_HAR

# 遍历所有模型和数据集组合
for model in MLP LeNet ResNet18 ResNet50 ResNet101 RNN GRU LSTM BiLSTM CNN+GRU ViT; do
    for dataset in UT_HAR_data NTU-Fi-HumanID NTU-Fi_HAR Widar; do
        python run.py --model $model --dataset $dataset
    done
done
```

### **2. 无监督学习 (自监督)**

```bash
# AutoFi自监督学习
python self_supervised.py --model MLP
python self_supervised.py --model ResNet18
```

---

## 🔄 **与当前项目的集成价值**

### **A. Sim2Real验证** ⭐⭐⭐⭐⭐

| 当前项目 | Benchmark | 集成价值 |
|----------|-----------|----------|
| **合成CSI数据** | **真实CSI数据** | **跨域泛化验证** |
| 物理引导生成器 | 4个公开数据集 | 验证生成数据真实性 |
| D2实验(540个) | 44个基准实验 | 对比合成vs真实性能 |

#### **集成实验设计**:
1. **在合成数据上训练** → **在真实数据上测试**
2. **在真实数据上训练** → **在合成数据上测试**  
3. **混合训练**: 合成+真实数据联合训练
4. **域适应**: 使用少量真实数据微调合成模型

### **B. 模型对比基准** ⭐⭐⭐⭐

| 我们的模型 | Benchmark模型 | 对比价值 |
|------------|---------------|----------|
| **enhanced** (BiLSTM+SE+Attention) | **BiLSTM** | 验证增强效果 |
| **cnn** (SimpleCNN) | **LeNet, ResNet18/50/101** | CNN架构对比 |
| **bilstm** | **BiLSTM, LSTM, GRU** | RNN变体对比 |
| **conformer_lite** | **ViT** | Transformer对比 |

### **C. 跨域泛化评估** ⭐⭐⭐⭐⭐

```bash
# LOSO (Leave-One-Subject-Out)
# LORO (Leave-One-Room-Out)  
# LODO (Leave-One-Domain-Out)
```

#### **具体实验**:
1. **训练在UT-HAR** → **测试在NTU-Fi_HAR**
2. **训练在合成数据** → **测试在所有真实数据集**
3. **少样本学习**: 10%-20%真实数据达到90%+性能

---

## 🛠️ **技术集成方案**

### **方案1: 直接集成 (推荐)**

#### **步骤1: 环境准备**
```bash
cd /workspace
git clone https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark.git benchmarks/wifi_csi_benchmark
cd benchmarks/wifi_csi_benchmark
pip install -r requirements.txt
```

#### **步骤2: 数据集下载**
```bash
# 需要手动下载并按指定结构组织:
# benchmarks/wifi_csi_benchmark/Data/
# ├── UT_HAR/
# ├── NTU-Fi_HAR/
# ├── NTU-Fi-HumanID/
# └── Widardata/
```

#### **步骤3: 模型适配**
```python
# 将我们的模型添加到benchmark中
# 修改 benchmarks/wifi_csi_benchmark/util.py:
def load_data_n_model(dataset, model, root):
    # ... 现有代码 ...
    
    # 添加我们的模型
    elif model == 'enhanced':
        from our_models import EnhancedModel
        model = EnhancedModel(input_dim=input_dim, num_classes=num_classes)
    elif model == 'conformer_lite':  
        from our_models import ConformerLite
        model = ConformerLite(input_dim=input_dim, num_classes=num_classes)
```

### **方案2: 反向集成**

#### **将benchmark模型添加到我们的项目**
```python
# src/models.py中添加:
class ResNet18_CSI(nn.Module):
    """基于benchmark的ResNet18实现"""
    # ... 从benchmark复制实现 ...

def build_model(name, F, num_classes, T=None):
    # 添加benchmark模型
    elif name == "resnet18":
        return ResNet18_CSI(input_dim=F, num_classes=num_classes)
    elif name == "vit":
        return ViT_CSI(input_dim=F, num_classes=num_classes)
    # ... 其他benchmark模型
```

#### **扩展D2实验配置**
```json
{
  "models": [
    "enhanced", "cnn", "bilstm", "conformer_lite",
    "resnet18", "vit", "lstm", "gru"
  ],
  "seeds": [0, 1, 2, 3, 4],
  "datasets": ["synthetic", "ut_har", "ntu_fi_har"],
  "cross_domain": true
}
```

---

## 📊 **建议实验序列**

### **Phase 1: 基准建立** (立即可做)
```bash
# 1. 在4个真实数据集上建立基准
python benchmarks/wifi_csi_benchmark/run.py --model ResNet18 --dataset UT_HAR_data
python benchmarks/wifi_csi_benchmark/run.py --model BiLSTM --dataset NTU-Fi_HAR

# 2. 记录基准性能 (准确率、ECE、参数量)
```

### **Phase 2: 模型对比** (1周内)
```bash
# 1. 将我们的enhanced模型集成到benchmark
# 2. 在真实数据集上对比性能
# 3. 验证enhanced模型的优势
```

### **Phase 3: Sim2Real验证** (2周内)
```bash
# 1. 合成数据训练 → 真实数据测试
# 2. 真实数据训练 → 合成数据测试  
# 3. 分析域差距 (domain gap)
```

### **Phase 4: 少样本学习** (3周内)
```bash
# 1. 10%真实数据微调合成模型
# 2. 评估达到90%+基准性能所需的真实数据量
# 3. 生成学习曲线
```

---

## 📈 **期望实验结果**

### **论文贡献点**:

1. **物理引导合成数据的有效性**:
   - 合成数据训练的模型在真实数据上性能 ≥ 80%基准
   - 证明物理参数建模的重要性

2. **增强模型架构优势**:
   - Enhanced模型 vs BiLSTM: +5-10% accuracy
   - 更好的校准性能 (更低ECE)

3. **高效Sim2Real转移**:
   - 仅需10-20%真实数据即可达到90-95%基准性能
   - 对比需要100%真实数据的传统方法

4. **跨域泛化能力**:
   - LODO实验: 在未见域上 ≥70%性能
   - 证明合成数据的泛化价值

---

## 🚧 **实施计划**

### **立即行动 (今天)**:
1. **下载benchmark代码到项目**
2. **分析数据集格式和模型接口**  
3. **设计第一个集成实验**

### **本周内**:
1. **完成benchmark环境搭建**
2. **运行基准实验，建立基线性能**
3. **开始模型集成工作**

### **本月内**:
1. **完成核心Sim2Real实验**
2. **生成对比表格和图表**
3. **撰写实验结果章节**

---

## 💡 **代码实现建议**

### **统一实验框架**:
```python
# scripts/run_sim2real_benchmark.py
def run_sim2real_experiment(
    synthetic_config: dict,
    real_datasets: list,
    models: list,
    transfer_ratios: list = [0.1, 0.2, 0.5, 1.0]
):
    """
    运行完整的Sim2Real基准实验
    
    Args:
        synthetic_config: D2合成数据配置
        real_datasets: ['UT_HAR', 'NTU-Fi_HAR', ...]
        models: ['enhanced', 'resnet18', 'bilstm', ...]
        transfer_ratios: 真实数据使用比例
    """
    results = {}
    
    for model in models:
        for real_dataset in real_datasets:
            # 1. 纯合成训练 → 真实测试
            synthetic_to_real = train_on_synthetic_test_on_real(
                model, synthetic_config, real_dataset
            )
            
            # 2. 少样本微调
            few_shot_results = []
            for ratio in transfer_ratios:
                result = few_shot_finetune(
                    model, synthetic_config, real_dataset, ratio
                )
                few_shot_results.append(result)
            
            results[f"{model}_{real_dataset}"] = {
                "synthetic_to_real": synthetic_to_real,
                "few_shot": few_shot_results
            }
    
    return results
```

**这个benchmark为我们的论文提供了强大的实验基础和对比基准！** 🎯