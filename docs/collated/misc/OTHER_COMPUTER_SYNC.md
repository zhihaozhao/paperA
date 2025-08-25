# 💻 其他电脑上的简化同步指南

## ✅ 好消息：大大简化了！

现在所有内容都已经在GitHub上了，其他电脑只需要**简单克隆**即可！

## 🚀 其他电脑上的操作（5分钟完成）

### 第1步：创建项目目录
```powershell
# 在新电脑上
cd D:\workspace_AI\  # 或你喜欢的路径
mkdir WiFi-CSI-Project
cd WiFi-CSI-Project
```

### 第2步：克隆所有4个仓库
```powershell
# 一次性克隆所有仓库
git clone https://github.com/zhihaozhao/paperA.git WiFi-CSI-Sensing-Core
git clone https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results.git
git clone https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper.git
git clone https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis.git

# 设置主仓库分支
cd WiFi-CSI-Sensing-Core
git checkout feat/enhanced-model-and-sweep
cd ..
```

### 第3步：配置Python环境
```powershell
# 激活环境
conda activate your_env_name  # 根据新电脑的环境名

# 安装依赖
cd WiFi-CSI-Sensing-Core
pip install -r requirements.txt
cd ..
```

## 🎉 完成！

就这么简单！你将得到完全相同的项目结构：

```
D:\workspace_AI\WiFi-CSI-Project\
├── WiFi-CSI-Sensing-Core\          # ✅ 有完整代码
├── WiFi-CSI-Sensing-Results\       # ✅ 有实验数据
├── WiFi-CSI-Journal-Paper\         # ✅ 有论文文件
└── WiFi-CSI-PhD-Thesis\           # ✅ 有论文章节
```

## 🔄 日常同步

### 拉取最新更改
```powershell
# 在每个仓库中拉取更新
cd WiFi-CSI-Sensing-Core
git pull origin feat/enhanced-model-and-sweep

cd ..\WiFi-CSI-Sensing-Results
git pull origin main

cd ..\WiFi-CSI-Journal-Paper
git pull origin main

cd ..\WiFi-CSI-PhD-Thesis
git pull origin main
```

### 批量更新脚本
创建 `update-all.bat`：
```batch
@echo off
echo 更新所有仓库...

cd WiFi-CSI-Sensing-Core
git pull origin feat/enhanced-model-and-sweep

cd ..\WiFi-CSI-Sensing-Results
git pull origin main

cd ..\WiFi-CSI-Journal-Paper
git pull origin main

cd ..\WiFi-CSI-PhD-Thesis
git pull origin main

echo 所有仓库更新完成！
pause
```

## 👥 团队协作

### 给团队成员的指导
只需要发送这个简单指令：

```powershell
# 团队成员执行（一键设置）
mkdir WiFi-CSI-Project && cd WiFi-CSI-Project
git clone https://github.com/zhihaozhao/paperA.git WiFi-CSI-Sensing-Core
git clone https://github.com/zhihaozhao/WiFi-CSI-Sensing-Results.git
git clone https://github.com/zhihaozhao/WiFi-CSI-Journal-Paper.git
git clone https://github.com/zhihaozhao/WiFi-CSI-PhD-Thesis.git
```

## 💡 对比：之前 vs 现在

### 😫 之前（复杂）：
- 下载压缩包
- 解压文件
- 复制到多个目录
- 分别推送
- 处理权限问题

### 😊 现在（简单）：
- 4个git clone命令
- 完成！

## 🔒 权限管理

### 如果是私有仓库：
给团队成员添加协作者权限到4个仓库：
- WiFi-CSI-Sensing-Results
- WiFi-CSI-Journal-Paper  
- WiFi-CSI-PhD-Thesis
- paperA (主仓库)

### 选择性权限：
- **导师**：只给论文仓库权限
- **合作者**：给结果和代码仓库权限
- **审稿人**：可以创建只读分支

---

**总结：第一次设置复杂，但之后的每台电脑都超级简单！** 🎯