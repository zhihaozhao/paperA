# 🪟 Windows下的完整设置指南

## 📁 第1步：下载压缩包

### 方法A：通过服务器下载（推荐）
如果你有服务器访问权限，将以下文件下载到本地：
```
服务器文件：/workspace/wifi-csi-repos-backup.tar.gz (4.5MB)
下载到：D:\workspace_AI\
```

### 方法B：创建分片下载（如果文件太大）
我可以将压缩包分割成小文件供你下载。

## 📦 第2步：解压.tar.gz文件

### 使用7-Zip（推荐）
1. **下载安装7-Zip**：https://www.7-zip.org/
2. **右键点击** `wifi-csi-repos-backup.tar.gz`
3. **选择** "7-Zip" → "Extract Here"
4. **会得到** `repos/` 文件夹

### 使用Windows内置工具
1. **先解压.gz**：右键 → "Extract All" 得到 `.tar` 文件
2. **再解压.tar**：可能需要安装额外工具

### 使用PowerShell命令（如果有tar命令）
```powershell
tar -xzf wifi-csi-repos-backup.tar.gz
```

## 💻 第3步：Windows PowerShell操作

### 打开PowerShell
1. **按** `Win + R`
2. **输入** `powershell`
3. **按** `Enter`

### 执行设置命令

```powershell
# 进入工作目录
cd D:\workspace_AI\WiFi-CSI-Project

# 确认目录结构
Get-ChildItem

# 应该看到：
# WiFi-CSI-Sensing-Core\         (有内容)
# WiFi-CSI-Sensing-Results\      (空仓库 - 正常)
# WiFi-CSI-Journal-Paper\        (空仓库 - 正常)
# WiFi-CSI-PhD-Thesis\          (空仓库 - 正常)
# repos\                         (解压得到的内容)
```

## 🚀 第4步：复制内容并推送

### 复制实验结果仓库
```powershell
Write-Host "处理实验结果仓库..." -ForegroundColor Green

# 复制文件（排除.git目录）
Copy-Item -Path "repos\WiFi-CSI-Sensing-Results\*" -Destination "WiFi-CSI-Sensing-Results\" -Recurse -Force -Exclude ".git"

# 进入仓库并推送
cd WiFi-CSI-Sensing-Results
git add .
git commit -m "Initial commit: WiFi-CSI experimental results and data tables"
git push origin main
cd ..

Write-Host "实验结果仓库推送完成！" -ForegroundColor Green
```

### 复制期刊论文仓库
```powershell
Write-Host "处理期刊论文仓库..." -ForegroundColor Green

Copy-Item -Path "repos\WiFi-CSI-Journal-Paper\*" -Destination "WiFi-CSI-Journal-Paper\" -Recurse -Force -Exclude ".git"

cd WiFi-CSI-Journal-Paper
git add .
git commit -m "Initial commit: WiFi-CSI journal paper LaTeX sources and references"
git push origin main
cd ..

Write-Host "期刊论文仓库推送完成！" -ForegroundColor Green
```

### 复制博士论文仓库
```powershell
Write-Host "处理博士论文仓库..." -ForegroundColor Green

Copy-Item -Path "repos\WiFi-CSI-PhD-Thesis\*" -Destination "WiFi-CSI-PhD-Thesis\" -Recurse -Force -Exclude ".git"

cd WiFi-CSI-PhD-Thesis
git add .
git commit -m "Initial commit: WiFi-CSI PhD thesis LaTeX sources and chapters"
git push origin main
cd ..

Write-Host "博士论文仓库推送完成！" -ForegroundColor Green
```

## 🧹 第5步：清理临时文件

```powershell
Write-Host "清理临时文件..." -ForegroundColor Yellow

# 删除临时文件
Remove-Item -Path "repos" -Recurse -Force
Remove-Item -Path "wifi-csi-repos-backup.tar.gz" -Force

Write-Host "清理完成！" -ForegroundColor Green
```

## ⚙️ 第6步：配置Python环境

```powershell
# 激活conda环境
conda activate D:\workspace_AI\Anaconda3\envs\py310

# 进入主仓库
cd WiFi-CSI-Sensing-Core

# 安装依赖
pip install -r requirements.txt

# 验证环境
python -c "import torch; print('PyTorch可用:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

Write-Host "Python环境配置完成！" -ForegroundColor Green
```

## 🔧 Windows特有的注意事项

### 1. 路径分隔符
- 使用 `\` 而不是 `/`
- PowerShell会自动处理大部分路径问题

### 2. 文件权限
- 如果遇到权限问题，以管理员身份运行PowerShell

### 3. 中文路径问题
- 确保路径中的中文字符正确显示
- 如有问题，使用英文路径

### 4. Git配置
```powershell
# 如果是第一次使用Git，需要配置
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 📋 Windows下的执行清单

- [ ] 下载 `wifi-csi-repos-backup.tar.gz`
- [ ] 安装7-Zip（如果没有）
- [ ] 解压压缩包得到 `repos/` 目录
- [ ] 在PowerShell中执行复制和推送命令
- [ ] 清理临时文件
- [ ] 配置Python环境
- [ ] 验证GitHub仓库内容

## 🆘 常见问题解决

### 压缩包解压问题
```powershell
# 如果tar命令不可用，使用PowerShell的Expand-Archive
# 但这个命令不支持.tar.gz，需要先转换格式
```

### Git推送问题
```powershell
# 如果推送失败，检查网络和凭据
git config --list
```

### 路径问题
```powershell
# 使用绝对路径避免问题
$projectPath = "D:\workspace_AI\WiFi-CSI-Project"
cd $projectPath
```

---

**现在你处于哪个步骤？需要我帮你下载压缩包或提供其他Windows特定的帮助吗？** 🪟