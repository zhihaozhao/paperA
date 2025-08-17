@echo off
setlocal enabledelayedexpansion

:: D3 实验: LORO (Leave-One-Room-Out) 跨房间泛化实验
:: 基于WiFi-CSI-Sensing-Benchmark真实数据
:: 支持环境自动检测: 远程GPU(base) vs 本地CPU(py310)

echo ========================================
echo   D3 LORO 跨房间泛化实验
echo   基于WiFi CSI Benchmark真实数据
echo ========================================

:: 环境自动检测 (如果未在父脚本中设置)
if "%PYTHON_ENV%"=="" (
    if exist "D:\anaconda\python.exe" (
        echo [检测] 远程GPU环境 - 使用conda base
        set PYTHON_ENV=base
        set ENV_TYPE=remote_gpu
    ) else if exist "D:\workspace_AI\Anaconda3\envs\py310\python.exe" (
        echo [检测] 本地CPU环境 - 使用conda py310
        set PYTHON_ENV=py310
        set ENV_TYPE=local_cpu
    ) else (
        echo [默认] 使用py310环境
        set PYTHON_ENV=py310
        set ENV_TYPE=default
    )
)

:: 设置默认参数 (基于D3_D4实验计划)
if "%MODELS%"=="" set MODELS=enhanced,cnn,bilstm,conformer_lite
if "%SEEDS%"=="" set SEEDS=0,1,2,3,4
if "%EPOCHS%"=="" set EPOCHS=100
if "%BENCHMARK_PATH%"=="" set BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\d3\loro

echo 实验配置:
echo   模型列表: %MODELS%
echo   随机种子: %SEEDS%
echo   训练轮数: %EPOCHS%
echo   数据集路径: %BENCHMARK_PATH%
echo   输出目录: %OUTPUT_DIR%
echo   Python环境: %PYTHON_ENV% (%ENV_TYPE%)

:: 激活conda环境
echo.
echo [环境] 激活conda环境 %PYTHON_ENV%...
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [错误] 无法激活conda环境 %PYTHON_ENV%
    if "%ENV_TYPE%"=="remote_gpu" (
        echo 远程GPU环境：请确保conda base环境可用
    ) else (
        echo 本地CPU环境：请确保已安装conda并创建了py310环境
    )
    pause
    exit /b 1
)

:: 切换到项目根目录
cd /d "%~dp0\.."
echo [信息] 项目根目录: %CD%

:: 检查benchmark数据集
if not exist "%BENCHMARK_PATH%" (
    echo [警告] 未找到benchmark数据集: %BENCHMARK_PATH%
    echo [信息] 创建占位目录并继续实验...
    mkdir "%BENCHMARK_PATH%" 2>nul
)

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" 2>nul

:: 验证关键模块
echo [验证] 检查关键模块导入...
python -c "from src.train_cross_domain import main; print('✓ train_cross_domain模块可用')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] train_cross_domain模块导入失败
    echo 请确保所有依赖已安装: torch, numpy, scikit-learn
    pause
    exit /b 1
)

:: 计算总实验数
set TOTAL_RUNS=0
for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        set /a TOTAL_RUNS+=1
    )
)

echo.
echo [开始] 运行 %TOTAL_RUNS% 个LORO实验配置...

set CURRENT_RUN=0
:: 运行所有模型和种子的组合
for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        set /a CURRENT_RUN+=1
        echo.
        echo [!CURRENT_RUN!/%TOTAL_RUNS%] LORO: model=%%m, seed=%%s
        
        set OUTPUT_FILE=%OUTPUT_DIR%\loro_%%m_seed%%s.json
        
        echo [执行] 运行LORO跨房间实验...
        python -m src.train_cross_domain ^
            --model %%m ^
            --protocol loro ^
            --benchmark_path "%BENCHMARK_PATH%" ^
            --seed %%s ^
            --epochs %EPOCHS% ^
            --output_dir "%OUTPUT_DIR%" ^
            --out "!OUTPUT_FILE!"
        
        if !ERRORLEVEL! neq 0 (
            echo [错误] 模型 %%m 种子 %%s 实验失败
            echo [继续] 跳过到下一个实验...
        ) else (
            echo [完成] 结果保存到: !OUTPUT_FILE!
        )
    )
)

:: 生成D3 LORO汇总报告
echo.
echo [汇总] 生成D3 LORO实验报告...
python -c "
import json, glob, numpy as np
from pathlib import Path
import os

# 读取所有LORO结果文件
output_dir = '%OUTPUT_DIR%'.replace('\\\\', '/')
pattern = f'{output_dir}/loro_*_seed*.json'
files = glob.glob(pattern)

print(f'找到 {len(files)} 个结果文件')

results = []
models = []
seeds = []

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        # 提取文件名信息
        fname = os.path.basename(f)
        parts = fname.replace('.json', '').split('_')
        model = '_'.join(parts[1:-1])  # loro_MODEL_seedN -> MODEL
        seed = int(parts[-1].replace('seed', ''))
        
        if model not in models:
            models.append(model)
        if seed not in seeds:
            seeds.append(seed)
        
        # 提取指标 (根据D3实验要求)
        if 'aggregate_stats' in data:
            stats = data['aggregate_stats']
            results.append({
                'model': model,
                'seed': seed,
                'macro_f1': stats.get('macro_f1', {}).get('mean', 0.0),
                'falling_f1': stats.get('falling_f1', {}).get('mean', 0.0),
                'ece': stats.get('ece', {}).get('mean', 0.0),
                'auprc_falling': stats.get('auprc_falling', {}).get('mean', 0.0),
                'file': f
            })
        else:
            # Fallback for different result structure
            results.append({
                'model': model,
                'seed': seed,
                'macro_f1': data.get('macro_f1', 0.0),
                'falling_f1': data.get('falling_f1', 0.0),
                'ece': data.get('ece', 0.0),
                'auprc_falling': data.get('auprc', 0.0),
                'file': f
            })
            
    except Exception as e:
        print(f'[警告] 无法读取 {f}: {e}')

if results:
    # 按模型分组统计
    summary = {'models': {}}
    
    for model in models:
        model_results = [r for r in results if r['model'] == model]
        if model_results:
            macro_f1s = [r['macro_f1'] for r in model_results]
            falling_f1s = [r['falling_f1'] for r in model_results] 
            eces = [r['ece'] for r in model_results]
            auprcs = [r['auprc_falling'] for r in model_results]
            
            summary['models'][model] = {
                'n_seeds': len(model_results),
                'macro_f1_mean': float(np.mean(macro_f1s)),
                'macro_f1_std': float(np.std(macro_f1s)),
                'falling_f1_mean': float(np.mean(falling_f1s)),
                'falling_f1_std': float(np.std(falling_f1s)),
                'ece_mean': float(np.mean(eces)),
                'ece_std': float(np.std(eces)),
                'auprc_mean': float(np.mean(auprcs)),
                'auprc_std': float(np.std(auprcs))
            }
    
    summary['experiment_info'] = {
        'protocol': 'LORO',
        'total_configs': len(results),
        'models': models,
        'seeds': sorted(seeds),
        'success_criteria': {
            'falling_f1_target': 0.75,
            'macro_f1_target': 0.80,
            'ece_target': 0.15
        }
    }
    
    summary_file = os.path.join('%OUTPUT_DIR%', 'd3_loro_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print('\\n=== D3 LORO 实验汇总 ===')
    for model, stats in summary['models'].items():
        print(f'\\n{model.upper()}模型 (n={stats[\"n_seeds\"]}):')
        print(f'  Macro F1: {stats[\"macro_f1_mean\"]:.3f} ± {stats[\"macro_f1_std\"]:.3f}')
        print(f'  Falling F1: {stats[\"falling_f1_mean\"]:.3f} ± {stats[\"falling_f1_std\"]:.3f}')
        print(f'  ECE: {stats[\"ece_mean\"]:.3f} ± {stats[\"ece_std\"]:.3f}')
        print(f'  AUPRC: {stats[\"auprc_mean\"]:.3f} ± {stats[\"auprc_std\"]:.3f}')
        
        # 检查成功标准
        falling_ok = stats['falling_f1_mean'] >= 0.75
        macro_ok = stats['macro_f1_mean'] >= 0.80
        ece_ok = stats['ece_mean'] <= 0.15
        print(f'  成功标准: Falling F1 {'✓' if falling_ok else '✗'}, Macro F1 {'✓' if macro_ok else '✗'}, ECE {'✓' if ece_ok else '✗'}')
    
    print(f'\\n汇总保存到: {summary_file}')
else:
    print('[错误] 没有找到有效的结果文件')
    print('请检查实验是否成功运行')
"

echo.
echo ========================================
echo [成功] D3 LORO 实验全部完成!
echo ========================================
echo 结果目录: %OUTPUT_DIR%
echo.
echo 下一步建议:
echo   1. 查看汇总: type "%OUTPUT_DIR%\d3_loro_summary.json"
echo   2. 运行D4实验: scripts\run_d4_loro.bat
echo   3. 验证结果: python scripts\validate_d3_acceptance.py --protocol loro
echo   4. 对比LOSO结果: python scripts\compare_loso_loro.py
echo ========================================