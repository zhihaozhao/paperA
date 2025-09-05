# Xavier Python 3 Environment Setup Guide

## 📋 环境要求

升级Xavier从Python 2.7.17到Python 3环境，支持PyTorch和CUDA。

## 🚀 安装步骤

### 1. 系统更新
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. 安装Python 3.8+ 
```bash
# 安装Python 3.8 (推荐版本)
sudo apt install -y python3.8 python3.8-dev python3.8-venv
sudo apt install -y python3-pip

# 设置Python 3为默认
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
```

### 3. 创建虚拟环境
```bash
# 创建项目虚拟环境
python3.8 -m venv ~/xavier_env

# 激活环境
source ~/xavier_env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### 4. 安装PyTorch (Xavier专用)
```bash
# NVIDIA Jetson Xavier适用的PyTorch安装
# PyTorch 1.12.0 for JetPack 5.0
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.12.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.12.0-cp38-cp38-linux_aarch64.whl

# Torchvision (编译安装)
sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision && python setup.py install && cd ..
```

### 5. 安装项目依赖
```bash
# 下载项目代码
git clone <repo_url> && cd paperA
git checkout feat/enhanced-model-and-sweep

# 安装依赖
pip install -r xavier_requirements_python3.txt
```

## 🔧 验证安装

```bash
# 检查Python版本
python --version  # 应该显示Python 3.8.x

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 检查其他依赖
python -c "import numpy, psutil; print('Dependencies OK')"
```

## 🎯 运行D1效率测量

```bash
# 激活环境
source ~/xavier_env/bin/activate

# 运行测量脚本
python measure_d1_true_efficiency_xavier.py --device cuda

# 或使用执行脚本
chmod +x run_d1_true_xavier_efficiency.sh
./run_d1_true_xavier_efficiency.sh
```

## 📊 预期输出

成功运行后应看到：
```
🔧 D1 True Parameter Configuration Xavier Efficiency Measurement
📊 Target: PASE-Net & CNN ~64K parameters, BiLSTM capacity-matched
🖥️  Device: cuda
✅ Detected ARM64 architecture (Xavier/Jetson platform)
🖥️  NVIDIA GPU detected: Xavier AGX

📋 D1 True Configuration Parameter Validation:
  PASE-Net: 64,123 (64.1K)
  CNN: 63,847 (63.8K)  
  BiLSTM: 67,234 (67.2K)
```

## 🔍 故障排除

**PyTorch CUDA错误:**
```bash
# 重新安装CUDA兼容的PyTorch
pip uninstall torch torchvision
# 重新按步骤4安装
```

**内存不足:**
```bash
# 增加swap空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**权限错误:**
```bash
# 确保脚本有执行权限
chmod +x *.sh
chmod +x *.py
```

## ✅ 环境检查清单

- [ ] Python 3.8+安装成功
- [ ] PyTorch CUDA版本安装成功
- [ ] 虚拟环境创建并激活
- [ ] 所有依赖包安装成功
- [ ] CUDA可用性验证通过
- [ ] 测量脚本语法检查通过
- [ ] Xavier GPU正确识别

---

🎯 **目标**: 在现代Python 3环境中成功运行D1真实参数配置的Xavier效率测量！