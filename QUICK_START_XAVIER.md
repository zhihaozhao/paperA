# Xavier设备快速部署指南

## 🚀 一键部署和运行

### 方法1: 使用部署脚本（推荐）

```bash
# 在本地工作区运行
cd /workspace
./deploy_to_xavier.sh
```

这个脚本会：
1. 检查必要文件
2. 创建部署包
3. 传输到Xavier设备
4. 在Xavier上自动部署
5. 提供运行指导

### 方法2: 手动部署

```bash
# 1. 传输文件
scp measure_all_models_xavier.py nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp measure_conformer_lite_xavier.py nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp run_conformer_lite_xavier.sh nvidia@192.168.2.36:~/workspace_PHD/paperA/
scp src/models.py nvidia@192.168.2.36:~/workspace_PHD/paperA/src/

# 2. SSH连接
ssh nvidia@192.168.2.36

# 3. 在Xavier上运行
cd ~/workspace_PHD/paperA
chmod +x *.py *.sh
./run_xavier_experiments.sh
```

## 🎯 运行实验

### 在Xavier设备上运行以下任一命令：

```bash
# 选项1: 使用自动化脚本（推荐）
./run_xavier_experiments.sh

# 选项2: 运行所有模型测量
python3 measure_all_models_xavier.py --device cuda

# 选项3: 运行Conformer-lite专用测量
python3 measure_conformer_lite_xavier.py --device cuda

# 选项4: 使用原始脚本
./run_conformer_lite_xavier.sh
```

## 📊 预期结果

运行成功后，你将得到：

### 1. 参数量验证
- ✅ Enhanced: 640,713 参数
- ✅ CNN: 644,216 参数  
- ✅ BiLSTM: 583,688 参数
- ⚠️ Conformer-lite: ~1,448,064 参数（与目标1,498,672差异50,608）

### 2. GPU延迟测量
- 在Xavier设备上的实际GPU推理时间
- 与Paper 1 Table 1的对比

### 3. 生成文件
- `xavier_d1_all_models_YYYYMMDD_HHMMSS.json`
- `xavier_conformer_lite_YYYYMMDD_HHMMSS.json`
- `results_gpu/` 目录
- `xavier_experiment_summary.md`

## 📝 提交到Git

```bash
# 在Xavier设备上运行
git add xavier_*.json xavier_*.md results_gpu/
git commit -m "Add Xavier D1 experiment measurement results

- All D1 models measured on NVIDIA AGX Xavier 32G
- JetPack 4.6 + PyTorch 1.8 + CUDA 10 environment
- Real GPU latency measurements
- Paper 1 Table 1 comparison included
- Models: Enhanced, CNN, BiLSTM, Conformer-lite"

git push origin main
```

## 🔧 故障排除

### 常见问题：

1. **CUDA不可用**
   ```bash
   nvidia-smi  # 检查GPU状态
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **权限问题**
   ```bash
   chmod +x *.py *.sh
   ```

3. **依赖缺失**
   ```bash
   pip3 install psutil
   ```

4. **内存不足**
   ```bash
   free -h  # 检查内存
   ```

## 📋 文件清单

### 本地文件（已准备）：
- ✅ `measure_all_models_xavier.py` - 所有模型测量
- ✅ `measure_conformer_lite_xavier.py` - Conformer-lite专用
- ✅ `run_conformer_lite_xavier.sh` - 原始运行脚本
- ✅ `run_xavier_experiments.sh` - 自动化实验脚本
- ✅ `deploy_to_xavier.sh` - 一键部署脚本
- ✅ `src/models.py` - 更新的模型定义

### Xavier设备上生成的文件：
- 📄 `xavier_d1_all_models_*.json` - 所有模型结果
- 📄 `xavier_conformer_lite_*.json` - Conformer-lite详细结果
- 📄 `xavier_experiment_summary.md` - 实验摘要
- 📁 `results_gpu/` - 结果目录

## 🎉 完成！

运行完成后，你将获得：
1. **真实的Xavier GPU延迟数据**
2. **与Paper 1 Table 1的完整对比**
3. **所有D1模型的详细性能分析**
4. **可提交到Git的完整结果**

这些结果将填补Paper 1 Table 1中缺失的Conformer-lite真实数据！