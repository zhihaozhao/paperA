# 🔧 SenseFi Benchmark 集成指南

## 📍 **当前情况**
- **Benchmark位置**: `D:\workspace_PHD\paperA\Benchmark\WiFi-CSI-Sensing-Benchmark-main`
- **目标位置**: `/workspace/` (当前工作环境)
- **环境**: Windows → Linux workspace

## 🚀 **集成方案**

### **方案1: 直接复制关键文件** (推荐)

由于我们主要需要分析代码结构和运行实验，可以选择性复制关键文件：

```bash
# 在workspace根目录下创建benchmark目录
mkdir -p benchmark

# 手动复制以下关键文件/目录到 workspace/benchmark/
# 需要的文件列表：
```

**需要复制的关键文件**：
- `run.py` (主训练脚本)
- `self_supervised.py` (AutoFi自监督脚本)  
- `model/` (模型定义目录)
- `dataloader/` 或 `data_loader/` (数据加载代码)
- `utils/` (工具函数)
- `config/` 或配置相关文件
- `requirements.txt` (依赖文件)

### **方案2: 符号链接** (如果在WSL环境)

如果您使用WSL，可以创建符号链接：
```bash
ln -s /mnt/d/workspace_PHD/paperA/Benchmark/WiFi-CSI-Sensing-Benchmark-main /workspace/benchmark
```

### **方案3: 完整复制**

```bash
# 如果可以访问Windows路径
cp -r /mnt/d/workspace_PHD/paperA/Benchmark/WiFi-CSI-Sensing-Benchmark-main /workspace/benchmark
```

## 📋 **我需要分析的关键文件**

请优先提供以下文件的内容，我将基于这些进行分析：

### **高优先级文件**
1. **`run.py`** - 主训练脚本
2. **`self_supervised.py`** - AutoFi实现
3. **目录结构清单** - 完整的文件组织
4. **`model/`目录内容** - 模型实现

### **中优先级文件**
5. **数据加载相关代码** - 了解数据格式
6. **配置文件** - 了解超参数设置
7. **评估脚本** - 了解指标计算

### **数据集信息**
8. **`data/`目录结构** - 确认可用数据集
9. **数据格式样例** - 了解输入输出格式

## 🎯 **提供文件内容的方式**

### **方式1: 分批提供**
```bash
# 先提供项目结构
cd D:\workspace_PHD\paperA\Benchmark\WiFi-CSI-Sensing-Benchmark-main
dir /s > project_structure.txt

# 然后逐个提供关键文件内容
type run.py
type self_supervised.py
```

### **方式2: 重点文件优先**
直接复制粘贴最重要的文件内容：
1. `run.py` (主入口)
2. `model/` 目录下的模型定义
3. 数据集目录结构

## ⚡ **一旦文件到位，立即可进行的分析**

### **实验可用性分析**
- ✅ 哪些实验可以直接运行
- ✅ 需要修改哪些配置
- ✅ 如何与我们的D2实验结合

### **代码集成方案**
- ✅ 如何复用SenseFi的数据加载器
- ✅ 如何集成基线模型到我们的框架
- ✅ 如何添加校准指标评估

### **实验设计**
- ✅ 真实数据验证实验设计
- ✅ AutoFi vs 我们方法的对比
- ✅ Sim2Real转移学习实验

## 🔍 **临时分析方案**

如果不方便复制整个benchmark，您可以：

### **提供关键信息**
1. **项目结构**: `dir /s` 的输出
2. **主要文件内容**: `run.py`, `self_supervised.py`
3. **数据集列表**: `data/` 目录下有什么
4. **模型列表**: 支持哪些基线模型

### **我将分析**
- 可直接使用的实验脚本
- 需要的环境配置
- 与我们研究的最佳结合点
- 具体的集成实施计划

---

**请选择上述方案之一，或者直接提供关键文件内容，我将基于实际代码给出精准的集成建议！** 🎯