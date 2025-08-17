# Windows下D3和D4实验使用说明

## 环境要求
- Windows 10/11
- Conda Python 3.10 环境 (py310)
- 已安装依赖：numpy, torch, scikit-learn, matplotlib, pandas

## 实验说明

### D3实验 (LOSO - Leave-One-Subject-Out)
**目标**: 跨主体泛化评估，测试模型在不同人员间的泛化能力
**特点**: 
- 使用中等难度扰动模拟不同主体特征
- 关注Falling类检测的跨人员稳定性
- 默认3个种子，20个训练轮次

### D4实验 (LORO - Leave-One-Room-Out)  
**目标**: 跨房间泛化评估，测试模型在不同环境的鲁棒性
**特点**:
- 使用高难度扰动模拟不同房间环境
- 包含5种扰动配置：默认、子载波相关、环境突发、增益漂移、组合
- 默认4个种子，25个训练轮次

## 使用方法

### 1. 快速开始 (推荐)
```cmd
:: 进入scripts目录
cd scripts

:: 运行主控制脚本
run_d3_d4_windows.bat
```
然后根据菜单选择要运行的实验。

### 2. 单独运行D3实验
```cmd
cd scripts
run_d3_loso.bat
```

### 3. 单独运行D4实验  
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
:: 自定义参数示例
set MODEL=enhanced
set PYTHON_ENV=py310
set EPOCHS=15
set SEEDS=0,1,2,3,4
run_d3_loso.bat
```

## 输出结果

### 结果文件结构
```
results/
├── loso/                          # D3 LOSO结果
│   ├── loso_enhanced_seed0.json   # 各种子详细结果
│   ├── loso_enhanced_seed1.json
│   ├── loso_enhanced_seed2.json
│   └── d3_loso_summary.json       # D3汇总统计
└── loro/                          # D4 LORO结果
    ├── loro_enhanced_default_seed0.json  # 各扰动+种子结果
    ├── loro_enhanced_sc_corr_seed0.json
    ├── ...
    └── d4_loro_summary.json       # D4汇总统计
```

### 关键指标
- **Macro F1**: 四类平均F1分数
- **Falling F1**: 跌倒检测F1分数（关键指标）
- **AUPRC**: 跌倒类的精确率-召回率曲线下面积
- **ECE**: 期望校准误差（越小越好）

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
:: 自定义所有参数
set MODEL=enhanced
set PYTHON_ENV=py310  
set EPOCHS=30
set SEEDS=0,1,2,3,4,5
set DIFFICULTY=hard
set POSITIVE_CLASS=3
set OUTPUT_DIR=results\custom

:: 运行实验
run_d3_loso.bat
```

### 模型选择
支持的模型类型：
- `enhanced`: 增强BiLSTM模型（推荐）
- `lstm`: 基础LSTM模型
- `tcn`: 时间卷积网络
- `transformer`: 小型Transformer模型

### 扰动参数含义
- `sc_corr_rho`: 子载波相关性 (0.0-1.0)
- `env_burst_rate`: 环境突发频率 (0.0-0.1)  
- `gain_drift_std`: 增益漂移标准差 (0.0-0.01)

## 注意事项

1. **内存使用**: 完整实验可能需要较多内存，建议关闭其他程序
2. **运行时间**: D3约15-30分钟，D4约45-90分钟（取决于硬件）
3. **结果解读**: 关注跨域场景下Falling F1和ECE的变化趋势
4. **复现性**: 使用相同种子确保结果可复现

## 联系支持
如遇到问题，请检查：
1. Conda环境是否正确激活
2. 项目依赖是否完整安装  
3. 工作目录是否为项目根目录
4. Python模块路径是否正确设置