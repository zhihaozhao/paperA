# 本地同步多仓库结构指南

## 🎯 同步策略

基于你的多仓库架构，我推荐以下同步方案：

### 方案1：完整独立仓库（推荐）
为每个子仓库创建独立的GitHub仓库，完全解耦

### 方案2：子模块方式  
在主仓库中使用git submodules引用独立仓库

## 📋 方案1：独立仓库同步（推荐）

### 第1步：在GitHub创建新仓库

需要在GitHub上创建以下4个仓库：
```
1. zhihaozhao/WiFi-CSI-Sensing-Core        (主仓库，可重命名现有的paperA)
2. zhihaozhao/WiFi-CSI-Sensing-Results     (实验结果)  
3. zhihaozhao/WiFi-CSI-Journal-Paper       (期刊论文)
4. zhihaozhao/WiFi-CSI-PhD-Thesis          (博士论文)
```

### 第2步：推送服务器仓库到GitHub

**在服务器上执行：**

```bash
# 推送主仓库更新
cd /workspace
git add README.md
git commit -m "Update: Multi-repository architecture"
git push origin feat/enhanced-model-and-sweep

# 推送实验结果仓库
cd repos/WiFi-CSI-Sensing-Results
git remote add origin https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results.git
git push -u origin master

# 推送期刊论文仓库  
cd ../WiFi-CSI-Journal-Paper
git remote add origin https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper.git
git push -u origin master

# 推送博士论文仓库
cd ../WiFi-CSI-PhD-Thesis  
git remote add origin https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis.git
git push -u origin master
```

### 第3步：本地同步（PowerShell安全命令）

**在本地执行（使用PowerShell）：**

```powershell
# 创建项目根目录
cd D:\workspace_AI\
mkdir WiFi-CSI-Project
cd WiFi-CSI-Project

# 克隆主仓库
git clone https://github.com/zhihaozhao/paperA.git WiFi-CSI-Sensing-Core
cd WiFi-CSI-Sensing-Core
git checkout feat/enhanced-model-and-sweep

# 返回项目根目录  
cd ..

# 克隆实验结果仓库
git clone https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results.git

# 克隆期刊论文仓库
git clone https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper.git

# 克隆博士论文仓库
git clone https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis.git
```

### 第4步：本地项目结构

同步后的本地目录结构：
```
D:\workspace_AI\WiFi-CSI-Project\
├── WiFi-CSI-Sensing-Core/          # 主仓库(算法和脚本)
│   ├── src/
│   ├── scripts/  
│   ├── eval/
│   └── README.md
├── WiFi-CSI-Sensing-Results/       # 实验结果
│   ├── results/
│   ├── results_gpu/
│   └── tables/
├── WiFi-CSI-Journal-Paper/         # 期刊论文
│   ├── paper/
│   └── references/
└── WiFi-CSI-PhD-Thesis/           # 博士论文
    └── 论文/
```

## 📋 方案2：子模块同步（备选）

如果你希望保持单一主仓库入口：

### 本地设置子模块

```powershell
# 克隆主仓库
git clone https://github.com/zhihaozhao/paperA.git
cd paperA

# 添加子模块
git submodule add https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results.git repos/results
git submodule add https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper.git repos/paper  
git submodule add https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis.git repos/thesis

# 初始化子模块
git submodule init
git submodule update
```

### 日常同步命令

```powershell
# 更新所有子模块
git submodule foreach git pull origin master

# 推送主仓库更改
git add .
git commit -m "Update submodules"
git push origin feat/enhanced-model-and-sweep
```

## 🔧 开发工作流程

### 日常开发（本地）

```powershell
# 算法开发
cd WiFi-CSI-Sensing-Core
git checkout feat/enhanced-model-and-sweep
# 开发代码...
git add .
git commit -m "Implement new feature"
git push origin feat/enhanced-model-and-sweep

# 添加实验结果
cd ..\WiFi-CSI-Sensing-Results
git add results/new_experiment.json
git commit -m "Add new experiment results"  
git push origin master

# 论文写作
cd ..\WiFi-CSI-Journal-Paper
# 编辑LaTeX文件...
git add paper/main.tex
git commit -m "Update paper content"
git push origin master
```

### 环境配置

**本地Python环境设置：**
```powershell
# 激活本地环境
conda activate D:\workspace_AI\Anaconda3\envs\py310

# 安装依赖
cd WiFi-CSI-Sensing-Core
pip install -r requirements.txt
```

### 与服务器同步

**推送本地更改到服务器：**
```powershell
# 本地开发完成后，推送到GitHub
git push origin feat/enhanced-model-and-sweep

# 服务器拉取更新
# (在服务器SSH中执行)
git pull origin feat/enhanced-model-and-sweep
```

## 📝 同步最佳实践

### 1. 分仓库管理
- **代码开发**：在Core仓库进行
- **实验数据**：在Results仓库管理
- **论文写作**：在对应Paper/Thesis仓库

### 2. 分支策略
- **主仓库**：使用`feat/enhanced-model-and-sweep`进行开发
- **结果仓库**：通常使用`master`分支
- **论文仓库**：可以使用`draft`、`revision`等分支

### 3. PowerShell命令规范
```powershell
# ✅ 安全的分步命令
git add .
git commit -m "message"  
git push origin branch-name

# ❌ 避免使用管道和连接符
# git add . && git commit -m "message" | cat
```

### 4. 同步检查列表
- [ ] 主仓库代码是否已推送
- [ ] 实验结果是否已更新
- [ ] 论文修改是否已保存
- [ ] 本地环境是否正确配置

## 🚨 故障排除

### 常见问题

1. **权限问题**
   ```powershell
   # 配置Git凭据
   git config --global user.name "your-name"
   git config --global user.email "your-email"
   ```

2. **分支冲突**
   ```powershell
   # 拉取最新更改
   git fetch origin
   git merge origin/feat/enhanced-model-and-sweep
   ```

3. **子模块问题**（如使用方案2）
   ```powershell
   # 重新同步子模块
   git submodule update --init --recursive
   ```

---

**选择建议**：
- **方案1（独立仓库）**：更适合长期项目，完全解耦，协作友好
- **方案2（子模块）**：适合希望统一入口的场景，但管理稍复杂

推荐使用**方案1**，它完全解决了你提到的文件耦合问题！