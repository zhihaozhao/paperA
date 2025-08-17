@echo off
setlocal enabledelayedexpansion

:: D3 实验: LOSO (Leave-One-Subject-Out) 跨主体泛化实验
:: 适用于Windows conda py310环境

echo [D3 LOSO实验] 启动跨主体泛化评估...

:: 设置默认参数 (可通过环境变量覆盖)
if "%MODEL%"=="" set MODEL=enhanced
if "%EPOCHS%"=="" set EPOCHS=20
if "%SEEDS%"=="" set SEEDS=0,1,2
if "%DIFFICULTY%"=="" set DIFFICULTY=mid
if "%POSITIVE_CLASS%"=="" set POSITIVE_CLASS=3
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\loso
if "%PYTHON_ENV%"=="" set PYTHON_ENV=py310

echo 参数配置:
echo   模型: %MODEL%
echo   训练轮数: %EPOCHS%
echo   随机种子: %SEEDS%
echo   难度: %DIFFICULTY%
echo   正类索引: %POSITIVE_CLASS%
echo   输出目录: %OUTPUT_DIR%
echo   Python环境: %PYTHON_ENV%

:: 激活conda环境
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [错误] 无法激活conda环境 %PYTHON_ENV%
    exit /b 1
)

:: 切换到项目根目录
cd /d "%~dp0\.."
echo [信息] 工作目录: %CD%

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: 解析种子列表并运行实验
for %%s in (%SEEDS%) do (
    echo.
    echo [%%s] 开始 LOSO 实验 - 种子: %%s
    
    :: 模拟跨主体数据 (使用不同难度和扰动模拟不同主体)
    set OUTPUT_FILE=%OUTPUT_DIR%\loso_%MODEL%_seed%%s.json
    
    echo [%%s] 运行跨主体实验...
    python -m src.train_cross_domain ^
        --source synth ^
        --target synth ^
        --model %MODEL% ^
        --epochs %EPOCHS% ^
        --seed %%s ^
        --difficulty %DIFFICULTY% ^
        --positive_class %POSITIVE_CLASS% ^
        --n_samples 1500 ^
        --T 128 ^
        --F 30 ^
        --sc_corr_rho 0.6 ^
        --env_burst_rate 0.03 ^
        --gain_drift_std 0.002 ^
        --out "!OUTPUT_FILE!"
    
    if !ERRORLEVEL! neq 0 (
        echo [错误] 种子 %%s 实验失败
        goto :error_exit
    )
    
    echo [%%s] 完成 - 结果保存到: !OUTPUT_FILE!
)

:: 生成汇总报告
echo.
echo [汇总] 生成 D3 LOSO 实验报告...
python -c "
import json, glob, numpy as np, pandas as pd
from pathlib import Path

# 读取所有结果文件
pattern = '%OUTPUT_DIR%/loso_%MODEL%_seed*.json'
files = glob.glob(pattern.replace('\\', '/'))
results = []

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        metrics = data.get('target_test_metrics', {})
        seed = data.get('args', {}).get('seed', 0)
        
        results.append({
            'seed': seed,
            'macro_f1': metrics.get('macro_f1', 0.0),
            'falling_f1': metrics.get('f1_fall', 0.0),
            'auprc': metrics.get('auprc', 0.0),
            'ece': metrics.get('ece', 0.0),
            'file': f
        })
    except Exception as e:
        print(f'[警告] 无法读取 {f}: {e}')

if results:
    df = pd.DataFrame(results)
    summary = {
        'n_seeds': len(df),
        'macro_f1_mean': float(df['macro_f1'].mean()),
        'macro_f1_std': float(df['macro_f1'].std()),
        'falling_f1_mean': float(df['falling_f1'].mean()),
        'falling_f1_std': float(df['falling_f1'].std()),
        'auprc_mean': float(df['auprc'].mean()),
        'auprc_std': float(df['auprc'].std()),
        'ece_mean': float(df['ece'].mean()),
        'ece_std': float(df['ece'].std())
    }
    
    summary_file = '%OUTPUT_DIR%/d3_loso_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print('\\n=== D3 LOSO 实验汇总 ===')
    print(f'种子数量: {summary[\"n_seeds\"]}')
    print(f'Macro F1: {summary[\"macro_f1_mean\"]:.3f} ± {summary[\"macro_f1_std\"]:.3f}')
    print(f'Falling F1: {summary[\"falling_f1_mean\"]:.3f} ± {summary[\"falling_f1_std\"]:.3f}')
    print(f'AUPRC: {summary[\"auprc_mean\"]:.3f} ± {summary[\"auprc_std\"]:.3f}')
    print(f'ECE: {summary[\"ece_mean\"]:.3f} ± {summary[\"ece_std\"]:.3f}')
    print(f'\\n汇总保存到: {summary_file}')
else:
    print('[错误] 没有找到有效的结果文件')
"

if %ERRORLEVEL% neq 0 (
    echo [警告] 汇总报告生成失败，但实验已完成
)

echo.
echo [成功] D3 LOSO 实验全部完成!
echo 结果目录: %OUTPUT_DIR%
goto :eof

:error_exit
echo [失败] D3 实验异常退出
exit /b 1