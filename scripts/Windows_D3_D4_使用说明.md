# Windows下D3和D4实验使用说明

## 环境要求
- Windows 10/11
- Conda Python 3.10 环境 (py310) [[memory:6363638]]
- WiFi-CSI-Sensing-Benchmark数据集
- 已安装依赖：numpy, torch, scikit-learn, matplotlib, pandas, scipy

## 实验说明 (基于D3_D4_Experiment_Plans.md)

### D3实验: 跨域泛化评估
基于WiFi-CSI-Sensing-Benchmark真实数据进行跨域泛化验证

#### D3.1 LOSO (Leave-One-Subject-Out)
**目标**: 跨主体泛化评估，测试模型对不同人员的泛化能力
**配置**: 
- 4个模型：enhanced, cnn, bilstm, conformer_lite (容量匹配±10%)
- 5个种子：0,1,2,3,4 (与D2一致)
- 100个训练轮次，早停机制
- 成功标准：Falling F1 ≥ 0.75, Macro F1 ≥ 0.80, ECE ≤ 0.15

#### D3.2 LORO (Leave-One-Room-Out)  
**目标**: 跨房间泛化评估，测试模型对不同环境的鲁棒性
**配置**:
- 同D3.1的模型和种子配置
- 基于benchmark数据的真实房间划分
- 关注环境变化对模型性能的影响

### D4实验: Sim2Real标签效率评估
从D2预训练模型开始，评估在真实数据上的标签效率

#### D4.1 标签效率扫描
**目标**: 评估10-20%标签达到≥90-95%全监督性能
**配置**:
- 标签比例：1%, 5%, 10%, 15%, 20%, 50%, 100%
- 4种迁移方法：zero_shot, linear_probe, fine_tune, temp_scale
- 成功标准：零样本Falling F1 ≥ 0.60, 迁移提升 ≥ 15%

## 使用方法

### 1. 快速开始 (推荐)
```cmd
:: 进入scripts目录
cd scripts

:: 运行主控制脚本
run_d3_d4_windows.bat
```
然后根据菜单选择要运行的实验。

### 2. 单独运行D3 LOSO实验
```cmd
cd scripts
run_d3_loso.bat
```

### 3. 单独运行D3 LORO实验
```cmd
cd scripts  
run_d3_loro.bat
```

### 4. 单独运行D4 Sim2Real实验
```cmd
cd scripts
run_d4_loro.bat
```

### 4. 快速测试模式
```cmd
:: 设置快速模式（减少轮数和种子）
set QUICK_MODE=1
run_d3_d4_windows.bat
```

### 5. 自定义参数
```cmd
:: D3 LOSO自定义示例
set MODELS=enhanced,cnn
set PYTHON_ENV=py310
set EPOCHS=50
set SEEDS=0,1,2
run_d3_loso.bat

:: D4 Sim2Real自定义示例
set MODELS=enhanced,bilstm
set LABEL_RATIOS=0.05,0.10,0.20,1.00
set TRANSFER_METHODS=zero_shot,fine_tune
set SEEDS=0,1,2
run_d4_loro.bat
```

## 输出结果

### 结果文件结构
```
results/
├── d3/                            # D3 跨域泛化实验
│   ├── loso/                      # LOSO跨主体结果
│   │   ├── loso_enhanced_seed0.json
│   │   ├── loso_cnn_seed0.json
│   │   ├── loso_bilstm_seed0.json
│   │   ├── loso_conformer_lite_seed0.json
│   │   └── d3_loso_summary.json   # LOSO汇总统计
│   └── loro/                      # LORO跨房间结果
│       ├── loro_enhanced_seed0.json
│       ├── loro_cnn_seed0.json
│       ├── ...
│       └── d3_loro_summary.json   # LORO汇总统计
└── d4/                            # D4 Sim2Real实验
    └── sim2real/                  # Sim2Real标签效率结果
        ├── sim2real_enhanced_0.01_zero_shot_seed0.json
        ├── sim2real_enhanced_0.01_fine_tune_seed0.json
        ├── sim2real_enhanced_0.05_zero_shot_seed0.json
        ├── ...
        └── d4_sim2real_summary.json  # Sim2Real汇总统计
```

### 关键指标 (基于D3_D4实验计划)
- **Macro F1**: 四类平均F1分数 (目标≥0.80)
- **Falling F1**: 跌倒检测F1分数 (目标≥0.75，关键指标)
- **AUPRC Falling**: 跌倒类的精确率-召回率曲线下面积  
- **ECE**: 期望校准误差 (目标≤0.15，越小越好)
- **NLL**: 负对数似然 (校准质量指标)
- **Mutual Misclass**: 类别间误分类分析

## 故障排除

### 常见问题

1. **conda环境激活失败**
   ```cmd
   conda info --envs  # 查看可用环境
   conda create -n py310 python=3.10  # 创建py310环境
   ```

2. **Python模块导入失败**
   ```cmd
   conda activate py310
   pip install numpy torch scikit-learn matplotlib pandas scipy
   ```

3. **"找不到接受实际参数"错误**
   - 确保使用正确的参数名
   - 检查脚本是否有语法错误
   - 确认Python路径设置正确

4. **内存不足**
   ```cmd
   set QUICK_MODE=1  # 启用快速模式
   set SEEDS=0,1     # 减少种子数量
   ```

## 高级配置

### 环境变量覆盖
```cmd
:: D3实验自定义参数
set MODELS=enhanced,cnn,bilstm,conformer_lite
set PYTHON_ENV=py310  
set EPOCHS=100
set SEEDS=0,1,2,3,4
set BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main
set OUTPUT_DIR=results\d3\custom

:: 运行D3实验
run_d3_loso.bat

:: D4实验自定义参数
set MODELS=enhanced,cnn
set LABEL_RATIOS=0.01,0.05,0.10,0.15,0.20,0.50,1.00
set TRANSFER_METHODS=zero_shot,linear_probe,fine_tune,temp_scale
set D2_MODELS_PATH=checkpoints\d2
set OUTPUT_DIR=results\d4\custom

:: 运行D4实验
run_d4_loro.bat
```

### 模型选择 (容量匹配±10%)
支持的模型类型：
- `enhanced`: 增强BiLSTM模型（主模型，推荐）
- `cnn`: 卷积神经网络基线
- `bilstm`: 双向LSTM基线  
- `conformer_lite`: 轻量级Conformer模型

### 迁移学习方法 (D4专用)
- `zero_shot`: 直接评估，无适应
- `linear_probe`: 冻结骨干网络，仅训练分类器
- `fine_tune`: 端到端微调，低学习率
- `temp_scale`: 仅校准适应，最小计算开销

## 注意事项

1. **数据要求**: 需要WiFi-CSI-Sensing-Benchmark数据集，如无数据将运行模拟模式
2. **D2模型依赖**: D4实验需要D2预训练模型，如无模型将从零开始训练
3. **实验规模**: 
   - D3 LOSO: 4模型×5种子 = 20个实验
   - D3 LORO: 4模型×5种子 = 20个实验  
   - D4 Sim2Real: 4模型×7比例×4方法×5种子 = 560个实验
4. **运行时间**: D3约2-4小时，D4约8-16小时（完整模式，取决于硬件）
5. **成功标准**: 详见D3_D4_Experiment_Plans.md，关注达标率统计
6. **复现性**: 使用相同种子确保结果可复现 [[memory:6364081]]

## 联系支持
如遇到问题，请检查：
1. Conda环境是否正确激活
2. 项目依赖是否完整安装  
3. 工作目录是否为项目根目录
4. Python模块路径是否正确设置