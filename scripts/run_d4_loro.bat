@echo off
setlocal enabledelayedexpansion

:: D4 实验: Sim2Real 标签效率评估实验
:: 从D2预训练模型开始，在真实数据上进行标签效率测试
:: 支持环境自动检测: 远程GPU(base) vs 本地CPU(py310)

echo ========================================
echo   D4 Sim2Real 标签效率实验
echo   从合成预训练到真实数据迁移
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
if "%LABEL_RATIOS%"=="" set LABEL_RATIOS=0.01,0.05,0.10,0.15,0.20,0.50,1.00
if "%TRANSFER_METHODS%"=="" set TRANSFER_METHODS=zero_shot,linear_probe,fine_tune,temp_scale
if "%BENCHMARK_PATH%"=="" set BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main
if "%D2_MODELS_PATH%"=="" set D2_MODELS_PATH=checkpoints\d2
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\d4\sim2real

echo 实验配置:
echo   模型列表: %MODELS%
echo   随机种子: %SEEDS%
echo   标签比例: %LABEL_RATIOS%
echo   迁移方法: %TRANSFER_METHODS%
echo   D2模型路径: %D2_MODELS_PATH%
echo   真实数据路径: %BENCHMARK_PATH%
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

:: 检查D2预训练模型
if not exist "%D2_MODELS_PATH%" (
    echo [警告] 未找到D2预训练模型: %D2_MODELS_PATH%
    echo [信息] 创建占位目录，实验将从零开始训练...
    mkdir "%D2_MODELS_PATH%" 2>nul
)

:: 检查benchmark数据集
if not exist "%BENCHMARK_PATH%" (
    echo [警告] 未找到benchmark数据集: %BENCHMARK_PATH%
    echo [信息] 创建占位目录并继续实验...
    mkdir "%BENCHMARK_PATH%" 2>nul
)

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" 2>nul

:: 验证Sim2Real模块
echo [验证] 检查Sim2Real模块导入...
python -c "from src.sim2real import Sim2RealEvaluator; print('✓ sim2real模块可用')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [警告] sim2real模块不可用，将使用train_cross_domain的Sim2Real功能
    python -c "from src.train_cross_domain import main; print('✓ 使用train_cross_domain模块')" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [错误] 关键模块导入失败
        echo 请确保所有依赖已安装: torch, numpy, scikit-learn
        pause
        exit /b 1
    )
)

:: 计算总实验数
set TOTAL_RUNS=0
for %%m in (%MODELS%) do (
    for %%r in (%LABEL_RATIOS%) do (
        for %%t in (%TRANSFER_METHODS%) do (
            for %%s in (%SEEDS%) do (
                set /a TOTAL_RUNS+=1
            )
        )
    )
)

echo.
echo [开始] 运行 %TOTAL_RUNS% 个Sim2Real实验配置...
echo [提示] 这将需要较长时间，建议使用快速模式进行初始测试

set CURRENT_RUN=0
:: 运行所有组合实验
for %%m in (%MODELS%) do (
    for %%r in (%LABEL_RATIOS%) do (
        for %%t in (%TRANSFER_METHODS%) do (
            for %%s in (%SEEDS%) do (
                set /a CURRENT_RUN+=1
                echo.
                echo [!CURRENT_RUN!/%TOTAL_RUNS%] Sim2Real: model=%%m, ratio=%%r, method=%%t, seed=%%s
                
                set OUTPUT_FILE=%OUTPUT_DIR%\sim2real_%%m_%%r_%%t_seed%%s.json
                
                echo [执行] 运行Sim2Real标签效率实验...
                python -m src.train_cross_domain ^
                    --model %%m ^
                    --protocol sim2real ^
                    --benchmark_path "%BENCHMARK_PATH%" ^
                    --d2_model_path "%D2_MODELS_PATH%\%%m_best.pth" ^
                    --label_ratio %%r ^
                    --transfer_method %%t ^
                    --seed %%s ^
                    --output_dir "%OUTPUT_DIR%" ^
                    --out "!OUTPUT_FILE!"
                
                if !ERRORLEVEL! neq 0 (
                    echo [错误] 模型 %%m 比例 %%r 方法 %%t 种子 %%s 实验失败
                    echo [继续] 跳过到下一个实验...
                ) else (
                    echo [完成] 结果保存到: !OUTPUT_FILE!
                )
            )
        )
    )
)

:: 生成D4 Sim2Real汇总报告
echo.
echo [汇总] 生成D4 Sim2Real实验报告...
python -c "
import json, glob, numpy as np
from pathlib import Path
import os

# 读取所有Sim2Real结果文件
output_dir = '%OUTPUT_DIR%'.replace('\\\\', '/')
pattern = f'{output_dir}/sim2real_*_seed*.json'
files = glob.glob(pattern)

print(f'找到 {len(files)} 个结果文件')

results = []
models = []
ratios = []
methods = []
seeds = []

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        
        # 提取文件名信息: sim2real_MODEL_RATIO_METHOD_seedN.json
        fname = os.path.basename(f)
        parts = fname.replace('.json', '').split('_')
        
        if len(parts) >= 5:  # sim2real_model_ratio_method_seedN
            model = parts[1]
            ratio = float(parts[2])
            method = parts[3]
            seed = int(parts[-1].replace('seed', ''))
            
            if model not in models:
                models.append(model)
            if ratio not in ratios:
                ratios.append(ratio)
            if method not in methods:
                methods.append(method)
            if seed not in seeds:
                seeds.append(seed)
            
            # 提取指标
            if 'target_metrics' in data:
                metrics = data['target_metrics']
                results.append({
                    'model': model,
                    'label_ratio': ratio,
                    'transfer_method': method,
                    'seed': seed,
                    'macro_f1': metrics.get('macro_f1', 0.0),
                    'falling_f1': metrics.get('falling_f1', 0.0),
                    'ece': metrics.get('ece', 0.0),
                    'zero_shot_f1': data.get('zero_shot_metrics', {}).get('falling_f1', 0.0),
                    'file': f
                })
            
    except Exception as e:
        print(f'[警告] 无法读取 {f}: {e}')

if results:
    # 按模型和方法分组统计
    summary = {'models': {}, 'label_efficiency': {}}
    
    ratios.sort()
    
    for model in models:
        summary['models'][model] = {}
        
        for method in methods:
            method_results = [r for r in results if r['model'] == model and r['transfer_method'] == method]
            
            if method_results:
                # 按标签比例统计
                ratio_stats = {}
                for ratio in ratios:
                    ratio_results = [r for r in method_results if r['label_ratio'] == ratio]
                    if ratio_results:
                        falling_f1s = [r['falling_f1'] for r in ratio_results]
                        macro_f1s = [r['macro_f1'] for r in ratio_results]
                        eces = [r['ece'] for r in ratio_results]
                        
                        ratio_stats[str(ratio)] = {
                            'n_seeds': len(ratio_results),
                            'falling_f1_mean': float(np.mean(falling_f1s)),
                            'falling_f1_std': float(np.std(falling_f1s)),
                            'macro_f1_mean': float(np.mean(macro_f1s)),
                            'macro_f1_std': float(np.std(macro_f1s)),
                            'ece_mean': float(np.mean(eces)),
                            'ece_std': float(np.std(eces))
                        }
                
                summary['models'][model][method] = ratio_stats
    
    # 标签效率分析
    for ratio in ratios:
        ratio_results = [r for r in results if r['label_ratio'] == ratio]
        if ratio_results:
            best_results = {}
            for model in models:
                model_results = [r for r in ratio_results if r['model'] == model]
                if model_results:
                    # 找到该模型在该比例下的最佳方法
                    best_f1 = max(r['falling_f1'] for r in model_results)
                    best_results[model] = best_f1
            
            summary['label_efficiency'][str(ratio)] = best_results
    
    summary['experiment_info'] = {
        'protocol': 'Sim2Real',
        'total_configs': len(results),
        'models': models,
        'label_ratios': ratios,
        'transfer_methods': methods,
        'seeds': sorted(seeds),
        'success_criteria': {
            'label_efficiency_target': '10-20% labels achieve >=90% of full-supervision',
            'zero_shot_falling_f1_target': 0.60,
            'zero_shot_macro_f1_target': 0.70,
            'transfer_gain_target': 0.15  # 15% improvement over zero-shot
        }
    }
    
    summary_file = os.path.join('%OUTPUT_DIR%', 'd4_sim2real_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print('\\n=== D4 Sim2Real 标签效率汇总 ===')
    
    # 显示标签效率曲线
    print('\\n标签效率分析:')
    for ratio_str, model_results in summary['label_efficiency'].items():
        ratio = float(ratio_str)
        print(f'\\n{ratio*100:5.1f}% 标签:')
        for model, f1 in model_results.items():
            print(f'  {model:12s}: Falling F1 = {f1:.3f}')
    
    # 显示零样本基线
    zero_shot_results = [r for r in results if r['transfer_method'] == 'zero_shot']
    if zero_shot_results:
        print('\\n零样本基线:')
        for model in models:
            model_zero = [r for r in zero_shot_results if r['model'] == model]
            if model_zero:
                avg_f1 = np.mean([r['falling_f1'] for r in model_zero])
                print(f'  {model:12s}: Falling F1 = {avg_f1:.3f}')
    
    # 检查成功标准
    print('\\n成功标准评估:')
    full_sup_results = [r for r in results if r['label_ratio'] == 1.0]
    low_ratio_results = [r for r in results if 0.1 <= r['label_ratio'] <= 0.2]
    
    if full_sup_results and low_ratio_results:
        for model in models:
            full_f1s = [r['falling_f1'] for r in full_sup_results if r['model'] == model]
            low_f1s = [r['falling_f1'] for r in low_ratio_results if r['model'] == model]
            
            if full_f1s and low_f1s:
                full_avg = np.mean(full_f1s)
                low_avg = np.mean(low_f1s)
                efficiency = low_avg / full_avg if full_avg > 0 else 0
                
                efficiency_ok = efficiency >= 0.9
                print(f'  {model:12s}: {low_avg:.3f}/{full_avg:.3f} = {efficiency:.1%} {'✓' if efficiency_ok else '✗'}')
    
    print(f'\\n汇总保存到: {summary_file}')
else:
    print('[错误] 没有找到有效的结果文件')
    print('请检查实验是否成功运行')
"

echo.
echo ========================================
echo [成功] D4 Sim2Real 实验全部完成!
echo ========================================
echo 结果目录: %OUTPUT_DIR%
echo.
echo 下一步建议:
echo   1. 查看汇总: type "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   2. 绘制标签效率曲线: python scripts\plot_sim2real_curves.py
echo   3. 验证结果: python scripts\validate_d4_acceptance.py
echo   4. 导出发表用图表: python scripts\export_d4_summary.py
echo ========================================