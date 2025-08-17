@echo off
setlocal enabledelayedexpansion

:: D4 实验: LORO (Leave-One-Room-Out) 跨房间泛化实验 + 高难度扰动
:: 适用于Windows conda py310环境

echo [D4 LORO实验] 启动跨房间泛化评估 + 高难度扰动...

:: 设置默认参数 (可通过环境变量覆盖)
if "%MODEL%"=="" set MODEL=enhanced
if "%EPOCHS%"=="" set EPOCHS=25
if "%SEEDS%"=="" set SEEDS=0,1,2,3
if "%DIFFICULTY%"=="" set DIFFICULTY=hard
if "%POSITIVE_CLASS%"=="" set POSITIVE_CLASS=3
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\loro
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

:: 定义扰动配置 (模拟不同房间环境)
set PERTURBATIONS=default sc_corr env_burst gain_drift combined

:: 解析种子列表并运行不同扰动实验
for %%s in (%SEEDS%) do (
    for %%p in (%PERTURBATIONS%) do (
        echo.
        echo [%%s-%%p] 开始 LORO 实验 - 种子: %%s, 扰动: %%p
        
        :: 根据扰动类型设置参数
        if "%%p"=="default" (
            set SC_CORR=None
            set ENV_BURST=0.0
            set GAIN_DRIFT=0.0
        ) else if "%%p"=="sc_corr" (
            set SC_CORR=0.8
            set ENV_BURST=0.0
            set GAIN_DRIFT=0.0
        ) else if "%%p"=="env_burst" (
            set SC_CORR=None
            set ENV_BURST=0.08
            set GAIN_DRIFT=0.0
        ) else if "%%p"=="gain_drift" (
            set SC_CORR=None
            set ENV_BURST=0.0
            set GAIN_DRIFT=0.005
        ) else if "%%p"=="combined" (
            set SC_CORR=0.7
            set ENV_BURST=0.05
            set GAIN_DRIFT=0.003
        )
        
        set OUTPUT_FILE=%OUTPUT_DIR%\loro_%MODEL%_%%p_seed%%s.json
        
        echo [%%s-%%p] 运行跨房间实验 - 扰动配置: %%p
        python -m src.train_cross_domain ^
            --source synth ^
            --target synth ^
            --model %MODEL% ^
            --epochs %EPOCHS% ^
            --seed %%s ^
            --difficulty %DIFFICULTY% ^
            --positive_class %POSITIVE_CLASS% ^
            --n_samples 2000 ^
            --T 128 ^
            --F 30 ^
            --sc_corr_rho !SC_CORR! ^
            --env_burst_rate !ENV_BURST! ^
            --gain_drift_std !GAIN_DRIFT! ^
            --out "!OUTPUT_FILE!"
        
        if !ERRORLEVEL! neq 0 (
            echo [错误] 种子 %%s 扰动 %%p 实验失败
            goto :error_exit
        )
        
        echo [%%s-%%p] 完成 - 结果保存到: !OUTPUT_FILE!
    )
)

:: 生成汇总报告
echo.
echo [汇总] 生成 D4 LORO 实验报告...
python -c "
import json, glob, numpy as np, pandas as pd
from pathlib import Path

# 读取所有结果文件
pattern = '%OUTPUT_DIR%/loro_%MODEL%_*_seed*.json'
files = glob.glob(pattern.replace('\\', '/'))
results = []

perturbations = ['default', 'sc_corr', 'env_burst', 'gain_drift', 'combined']

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        metrics = data.get('target_test_metrics', {})
        args = data.get('args', {})
        seed = args.get('seed', 0)
        
        # 从文件名推断扰动类型
        fname = Path(f).name
        perturbation = 'unknown'
        for p in perturbations:
            if f'_{p}_' in fname:
                perturbation = p
                break
        
        results.append({
            'seed': seed,
            'perturbation': perturbation,
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
    
    # 按扰动类型分组统计
    summary = {}
    for pert in perturbations:
        pert_df = df[df['perturbation'] == pert]
        if len(pert_df) > 0:
            summary[pert] = {
                'n_seeds': len(pert_df),
                'macro_f1_mean': float(pert_df['macro_f1'].mean()),
                'macro_f1_std': float(pert_df['macro_f1'].std()),
                'falling_f1_mean': float(pert_df['falling_f1'].mean()),
                'falling_f1_std': float(pert_df['falling_f1'].std()),
                'auprc_mean': float(pert_df['auprc'].mean()),
                'auprc_std': float(pert_df['auprc'].std()),
                'ece_mean': float(pert_df['ece'].mean()),
                'ece_std': float(pert_df['ece'].std())
            }
    
    summary_file = '%OUTPUT_DIR%/d4_loro_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print('\\n=== D4 LORO 实验汇总 ===')
    for pert, stats in summary.items():
        print(f'\\n{pert.upper()} 扰动 (n={stats[\"n_seeds\"]}):')
        print(f'  Macro F1: {stats[\"macro_f1_mean\"]:.3f} ± {stats[\"macro_f1_std\"]:.3f}')
        print(f'  Falling F1: {stats[\"falling_f1_mean\"]:.3f} ± {stats[\"falling_f1_std\"]:.3f}')
        print(f'  AUPRC: {stats[\"auprc_mean\"]:.3f} ± {stats[\"auprc_std\"]:.3f}')
        print(f'  ECE: {stats[\"ece_mean\"]:.3f} ± {stats[\"ece_std\"]:.3f}')
    print(f'\\n汇总保存到: {summary_file}')
else:
    print('[错误] 没有找到有效的结果文件')
"

if %ERRORLEVEL% neq 0 (
    echo [警告] 汇总报告生成失败，但实验已完成
)

echo.
echo [成功] D4 LORO 实验全部完成!
echo 结果目录: %OUTPUT_DIR%
goto :eof

:error_exit
echo [失败] D4 实验异常退出
exit /b 1