# Git分支管理方案 📊

## 🎯 **重组完成状态**

✅ **已完成**:
- 创建了统一的 `results/main` 分支
- 上传了D3/D4验收结果
- 创建了实验版本标签
- 推送到远程仓库

---

## 📋 **推荐的分支架构**

### 🔧 **代码开发分支**
```
feat/enhanced-model-and-sweep    # 主要实验代码开发 [推荐主分支]
feat/calib-sci-har              # 校准和科学计算特性  
master                          # 稳定主分支
```

### 📊 **结果管理分支** 
```
results/main                    # 统一的实验结果分支 [新建✅]
```

### 🏷️ **实验版本标签**
```
v1.0-d2-complete               # D2容量匹配实验 [已有]
v1.1-d3-d4-cross-domain        # D3/D4跨域+Sim2Real实验 [新建✅]
v1.2-acceptance                # 完整验收版本 [新建✅]
```

---

## 🚀 **分支独立性设计**

### **核心原则**:
- **代码与结果完全分离** - 避免冲突
- **Tag-based版本管理** - 清晰的实验里程碑
- **单一责任分支** - 每个分支职责明确

### **工作流程**:

#### 1️⃣ **代码开发** (在 `feat/enhanced-model-and-sweep`)
```bash
git checkout feat/enhanced-model-and-sweep
# 开发新模型、修改代码
git add . && git commit -m "enhance model architecture"
git push origin feat/enhanced-model-and-sweep
```

#### 2️⃣ **实验结果** (在 `results/main`)  
```bash
git checkout results/main
# 添加新实验结果
git add results/ && git commit -m "Add D5 experiment results"
git tag -a v1.3-d5-ablation -m "D5 ablation study complete"
git push origin results/main --tags
```

#### 3️⃣ **版本发布** (创建标签)
```bash
# 在results/main分支上标记实验里程碑
git tag -a v2.0-paper-ready -m "All experiments complete, paper submission ready"
git push origin --tags
```

---

## 🧹 **建议的分支清理**

### **可以删除的冗余分支**:
```bash
# 远程结果分支 - 已合并到results/main
origin/results/exp-2025        # ✅ 已合并
origin/results/exp-20250815    # ✅ 内容已包含 
origin/results/exp-20250826    # ✅ 可清理
origin/result/exp-20250815     # ✅ 重复，可清理

# 临时cursor分支 - 可清理
origin/cursor/*                # ✅ 临时分支，可清理
```

### **保留的重要分支**:
```bash
# 代码开发
feat/enhanced-model-and-sweep   # 🔥 主要开发分支 - 保留
feat/calib-sci-har             # 📊 校准特性 - 保留 
master                         # 🏠 主分支 - 保留

# 结果管理  
results/main                   # 📊 统一结果分支 - 保留
```

---

## 🎯 **当前实验状态标签**

| 标签 | 描述 | 包含实验 | 状态 |
|------|------|----------|------|
| `v1.0-d2-complete` | D2容量匹配实验 | 540个配置 | ✅ 完成 |
| `v1.1-d3-d4-cross-domain` | D3/D4跨域实验 | LOSO/LORO + Sim2Real | ✅ 新建 |
| `v1.2-acceptance` | 验收完成版本 | 82.1% F1 @ 20%标签 | ✅ 新建 |

---

## 🔄 **日常工作流建议**

### **开发新功能时**:
```bash
# 在代码分支工作
git checkout feat/enhanced-model-and-sweep
git pull origin feat/enhanced-model-and-sweep

# 开发完成后
git add . && git commit -m "feat: add new model component"  
git push origin feat/enhanced-model-and-sweep
```

### **运行实验后**:
```bash
# 在结果分支工作
git checkout results/main
git pull origin results/main

# 添加新实验结果
git add results/ && git commit -m "Add D5 ablation study results"
git tag -a v1.3-d5-ablation -m "D5 ablation study: attention vs SE modules"
git push origin results/main --tags
```

### **论文投稿时**:
```bash
# 创建论文就绪标签
git tag -a v2.0-paper-submission -m "All experiments complete, ready for IoTJ/TMC submission"
git push origin --tags
```

---

## 📝 **下一步操作建议**

1. **立即可做**:
   - 继续在 `feat/enhanced-model-and-sweep` 开发新功能
   - 新实验结果提交到 `results/main`
   - 使用tags管理实验版本

2. **可选清理** (谨慎操作):
   ```bash
   # 删除远程冗余分支 (可选)
   git push origin --delete results/exp-2025
   git push origin --delete results/exp-20250815  
   git push origin --delete results/exp-20250826
   ```

3. **长期维护**:
   - 定期将 `master` 与 `feat/enhanced-model-and-sweep` 同步
   - 每个重要实验完成后创建tag
   - 保持results/main分支的clean commit history

---

*分支重组完成时间: 2025-08-18*  
*新分支: `results/main` [✅ 已推送]*  
*新标签: `v1.1-d3-d4-cross-domain`, `v1.2-acceptance` [✅ 已推送]*