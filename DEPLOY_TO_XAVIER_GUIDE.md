# Xavier设备部署和运行指南

## 概述
本指南将帮助你在NVIDIA AGX Xavier设备上部署和运行D1实验的补充测量脚本。

## 设备信息
- **设备IP**: 192.168.2.36
- **用户名**: nvidia
- **目标目录**: ~/workspace_PHD/paperA
- **环境**: JetPack 4.6 + PyTorch 1.8 + CUDA 10

## 步骤1: 准备文件传输

### 1.1 创建部署包
在本地工作区运行以下命令，创建需要传输的文件列表：

```bash
cd /workspace
tar -czf xavier_scripts.tar.gz \
    measure_all_models_xavier.py \
    measure_conformer_lite_xavier.py \
    run_conformer_lite_xavier.sh \
    src/models.py \
    XAVIER_SCRIPT_UPDATE_SUMMARY.md \
    DEPLOY_TO_XAVIER_GUIDE.md
```

### 1.2 传输文件到Xavier
```bash
# 传输脚本包
scp xavier_scripts.tar.gz nvidia@192.168.2.36:~/workspace_PHD/paperA/

# 或者逐个传输重要文件
scp measure_all_models_xavier.py nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp measure_conformer_lite_xavier.py nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp run_conformer_lite_xavier.sh nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp src/models.py nvidia@192.168.2.36:~/workspace_PHD/paperA/src/
```

## 步骤2: SSH连接到Xavier

```bash
ssh nvidia@192.168.2.36
```

## 步骤3: 在Xavier上设置环境

### 3.1 解压文件（如果使用了tar包）
```bash
cd ~/workspace_PHD/paperA
tar -xzf xavier_scripts.tar.gz
```

### 3.2 设置权限
```bash
chmod +x run_conformer_lite_xavier.sh
chmod +x measure_all_models_xavier.py
chmod +x measure_conformer_lite_xavier.py
```

### 3.3 检查环境
```bash
# 检查Python和PyTorch版本
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 检查GPU状态
nvidia-smi
```

### 3.4 安装依赖（如果需要）
```bash
# 如果缺少psutil
pip3 install psutil

# 或者使用conda（如果可用）
conda install psutil
```

## 步骤4: 运行实验

### 4.1 运行所有模型测量
```bash
cd ~/workspace_PHD/paperA
python3 measure_all_models_xavier.py --device cuda --T 128 --F 52 --classes 8
```

### 4.2 运行Conformer-lite专用测量
```bash
python3 measure_conformer_lite_xavier.py --device cuda --T 128 --F 52 --classes 8
```

### 4.3 使用shell脚本运行
```bash
./run_conformer_lite_xavier.sh
```

## 步骤5: 收集结果

### 5.1 检查生成的文件
```bash
ls -la results_gpu/
ls -la xavier_*.json
```

### 5.2 查看结果摘要
```bash
# 查看最新的结果文件
python3 -c "
import json
import glob
import os

# 找到最新的结果文件
result_files = glob.glob('xavier_*.json')
if result_files:
    latest_file = max(result_files, key=os.path.getctime)
    print(f'Latest result file: {latest_file}')
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print('\\n=== Results Summary ===')
    if 'results' in data:
        for model_name, result in data['results'].items():
            if 'error' not in result:
                print(f'{model_name}: {result[\"parameters\"]:,} params, {result[\"inference_mean_ms\"]:.2f}ms')
            else:
                print(f'{model_name}: ERROR - {result[\"error\"]}')
"
```

## 步骤6: 提交到Git

### 6.1 添加结果文件
```bash
# 添加结果文件到git
git add results_gpu/
git add xavier_*.json
git add measure_*_xavier.py
git add run_conformer_lite_xavier.sh

# 提交更改
git commit -m "Add Xavier D1 experiment measurement results

- Added measure_all_models_xavier.py for all D1 models
- Added measure_conformer_lite_xavier.py for Conformer-lite specific measurement  
- Added run_conformer_lite_xavier.sh for automated execution
- Results from NVIDIA AGX Xavier 32G (JetPack 4.6 + PyTorch 1.8 + CUDA 10)
- All models tested with real D1 experiment configurations
- Paper 1 Table 1 comparison included"

# 推送到远程仓库
git push origin main
```

### 6.2 创建结果摘要
```bash
# 创建结果摘要文件
cat > xavier_results_summary.md << 'EOF'
# Xavier D1 Experiment Results Summary

## Device Information
- **Device**: NVIDIA AGX Xavier 32G
- **JetPack**: 4.6
- **PyTorch**: 1.8.x
- **CUDA**: 10.x
- **Date**: $(date)

## Model Performance Results

### Parameter Counts
| Model | Parameters | Paper 1 Table 1 | Status |
|-------|------------|------------------|--------|
| Enhanced | 640,713 | 640,713 | ✅ Match |
| CNN | 644,216 | 644,216 | ✅ Match |
| BiLSTM | 583,688 | 583,688 | ✅ Match |
| Conformer-lite | 1,448,064 | 1,498,672 | ⚠️ Different |

### GPU Latency Results
| Model | Xavier GPU (ms) | Paper 1 Table 1 (ms) | Status |
|-------|-----------------|----------------------|--------|
| Enhanced | [Measured] | 5.29 | [Status] |
| CNN | [Measured] | 0.90 | [Status] |
| BiLSTM | [Measured] | 8.97 | [Status] |
| Conformer-lite | [Measured] | 5.16 | [Status] |

## Files Generated
- `xavier_d1_all_models_YYYYMMDD_HHMMSS.json`
- `xavier_conformer_lite_YYYYMMDD_HHMMSS.json`
- `results_gpu/` directory with all measurement data

## Notes
- All models use real D1 experiment configurations from src/models.py
- Measurements performed with proper warmup and multiple runs
- Results include detailed statistics (mean, std, min, max, median, P95, P99)
EOF

git add xavier_results_summary.md
git commit -m "Add Xavier results summary"
git push origin main
```

## 步骤7: 传输结果回本地（可选）

如果需要将结果传输回本地：

```bash
# 在本地机器上运行
scp nvidia@192.168.2.36:~/workspace_PHD/paperA/xavier_*.json ./
scp nvidia@192.168.2.36:~/workspace_PHD/paperA/results_gpu/* ./results_gpu/
```

## 故障排除

### 常见问题

1. **CUDA不可用**
   ```bash
   # 检查CUDA状态
   nvidia-smi
   # 重启CUDA服务（如果需要）
   sudo systemctl restart nvidia-persistenced
   ```

2. **权限问题**
   ```bash
   chmod +x *.py *.sh
   ```

3. **依赖缺失**
   ```bash
   pip3 install psutil numpy scipy matplotlib
   ```

4. **内存不足**
   ```bash
   # 监控内存使用
   free -h
   # 清理缓存
   sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

## 预期结果

运行成功后，你应该看到：

1. **参数量匹配**: Enhanced, CNN, BiLSTM应该与Paper 1 Table 1完全匹配
2. **Conformer-lite**: 参数量接近1,498,672（当前配置约1,448,064）
3. **GPU延迟**: 在Xavier设备上的实际GPU延迟测量
4. **JSON报告**: 详细的性能数据和统计信息

## 联系信息

如果在运行过程中遇到问题，请提供：
- 错误信息
- 设备状态（nvidia-smi输出）
- Python/PyTorch版本信息
- 具体的命令和输出

这样我可以帮助你快速解决问题。