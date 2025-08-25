# 🔧 Git管理命令集 - D2实验结果管理

## 📋 **实验结果Git管理策略**

### **核心思路**:
1. **功能分支**: 按实验阶段创建分支
2. **结果分支**: 专门的分支存储实验结果  
3. **标签标记**: 重要节点打tag方便回滚
4. **数据保护**: 确保实验数据不丢失

---

## 🎯 **D2实验结果管理流程**

### **步骤1: 创建实验结果分支**
```bash
# 从当前分支创建结果分支
git checkout -b experiment/d2-results

# 或从特定基础分支创建
git checkout feat/enhanced-model-and-sweep
git checkout -b experiment/d2-results
```

### **步骤2: 添加D2实验结果**
```bash
# 添加结果文件
git add results/
git add cache/
git add logs/
git add plots/
git add tables/

# 提交实验结果
git commit -m "Add D2 experiment results: 540 configurations completed

- Models: enhanced, cnn, bilstm, conformer_lite
- Seeds: 0-4 (5 seeds)
- Grid: 3x3x3 (overlap, noise, burst) = 27 configs
- Total: 4 * 5 * 27 = 540 experiments
- Results: performance metrics, calibration analysis
- Generated: plots and summary tables"
```

### **步骤3: 推送结果分支**
```bash
# 推送到远程
git push origin experiment/d2-results

# 或者设置上游
git push -u origin experiment/d2-results
```

### **步骤4: 创建里程碑标签**
```bash
# 创建D2实验完成标签
git tag -a v1.0-d2-complete -m "D2 Experiment Complete: 540 configurations

Results Summary:
- Enhanced model: best overall performance
- CNN: robust across conditions  
- BiLSTM: sensitive to label noise
- ConformerLite: best calibration

Next: Sim2Real experiments"

# 推送标签
git push origin v1.0-d2-complete
```

---

## 🔄 **后续Sim2Real实验管理**

### **创建Sim2Real分支**
```bash
# 从D2结果分支创建新分支
git checkout experiment/d2-results
git checkout -b experiment/sim2real-results
```

### **完成Sim2Real后打标签**
```bash
git tag -a v1.1-sim2real-complete -m "Sim2Real Experiments Complete

- Transfer performance on SenseFi benchmarks
- Few-shot learning efficiency analysis
- Cross-domain generalization assessment
- Ready for paper submission"

git push origin v1.1-sim2real-complete
```

---

## 📝 **论文更新管理**

### **论文修改分支策略**
```bash
# 当前情况：论文引用更新
git checkout feat/enhanced-model-and-sweep

# 检查状态
git status

# 如果有未提交的更改
git add paper/main.tex paper/refs.bib
git commit -m "Update paper citations with real references

- Added SenseFi benchmark citation (yang2023sensefi)
- Replaced all placeholder references (reference1-12)
- Added physics modeling references
- Added trustworthy ML citations (calibration, SE networks)
- Added Sim2Real transfer learning references
- Total: 21 authoritative references covering all domains"

# 推送论文更新
git push origin feat/enhanced-model-and-sweep
```

### **论文最终版本管理**
```bash
# 实验完成后，合并结果到论文分支
git checkout feat/enhanced-model-and-sweep
git merge experiment/d2-results --no-ff -m "Merge D2 experimental results into paper

- Added D2 protocol results (540 configurations)
- Updated tables with actual performance metrics
- Generated figures for paper submission"

# 创建论文提交版本标签
git tag -a v2.0-paper-submission -m "Paper Submission Ready

- Complete D2 experimental validation
- Sim2Real benchmark integration
- All placeholders filled with real results
- Ready for TMC/IoTJ submission"

git push origin v2.0-paper-submission
```

---

## 🔄 **分支合并策略**

### **将实验结果合并到主分支**
```bash
# 合并到主开发分支
git checkout feat/enhanced-model-and-sweep
git merge experiment/d2-results --no-ff

# 合并到main分支（如果需要）
git checkout main  
git merge feat/enhanced-model-and-sweep --no-ff
git tag -a v3.0-project-complete -m "Project Complete: Ready for Publication"
```

### **紧急回滚策略**
```bash
# 查看所有标签
git tag -l

# 回滚到特定版本
git checkout v1.0-d2-complete
git checkout -b hotfix/rollback-d2

# 或者重置到特定提交
git reset --hard v1.0-d2-complete
```

---

## 📊 **查看和管理命令**

### **查看分支和标签**
```bash
# 查看所有分支
git branch -a

# 查看所有标签
git tag -l

# 查看标签详情
git show v1.0-d2-complete

# 查看分支图
git log --oneline --graph --all
```

### **清理和维护**
```bash
# 删除本地已合并分支
git branch -d experiment/old-branch

# 删除远程分支
git push origin --delete experiment/old-branch

# 清理远程引用
git remote prune origin
```

---

## ⚡ **当前推荐操作序列**

### **立即执行 (解决git问题)**:
```bash
# 1. 检查状态
git status

# 2. 如果有冲突，重置
git reset --hard HEAD

# 3. 强制拉取最新
git fetch origin
git reset --hard origin/feat/enhanced-model-and-sweep

# 4. 重新提交论文更新
git add paper/main.tex paper/refs.bib
git commit -m "Update paper citations with authoritative references"
git push origin feat/enhanced-model-and-sweep
```

### **D2结果管理**:
```bash
# 1. 创建D2结果分支
git checkout -b experiment/d2-results

# 2. 在远程GPU服务器上执行：
# scp -r gpu_server:/path/to/d2/results/* ./results/
# scp -r gpu_server:/path/to/plots/* ./plots/
# scp -r gpu_server:/path/to/logs/* ./logs/

# 3. 提交结果
git add results/ plots/ logs/ tables/
git commit -m "Add D2 experiment results: 540 configurations completed"
git push origin experiment/d2-results

# 4. 打标签
git tag -a v1.0-d2-complete -m "D2 Experiment Complete"
git push origin v1.0-d2-complete
```

---

## 🚨 **紧急情况处理**

### **Git卡住处理**:
```bash
# 终止当前操作
Ctrl+C

# 检查git进程
ps aux | grep git

# 强制终止git进程（如果需要）
killall git

# 重置状态
git reset --hard HEAD
git clean -fd
```

### **合并冲突处理**:
```bash
# 查看冲突文件
git status

# 编辑冲突文件或选择策略
git checkout --ours conflicted_file   # 保留本地版本
git checkout --theirs conflicted_file # 保留远程版本

# 标记解决
git add conflicted_file
git commit -m "Resolve merge conflict"
```

这个Git管理策略确保了实验数据的安全性和可追溯性！