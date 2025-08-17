@echo off
echo ===============================================
echo      D2实验结果验收执行脚本
echo ===============================================

echo [1] 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo Error: Python未找到，请检查Python安装
    pause
    exit /b 1
)

echo [2] 检查结果文件...
if not exist "results\" (
    echo Error: results目录不存在，请确保已拉取results/exp-2025分支
    pause
    exit /b 1
)

echo [3] 运行D2验收脚本...
python scripts\validate_d2_acceptance.py results\

if %errorlevel% equ 0 (
    echo [✅] D2验收脚本执行成功！
) else (
    echo [❌] D2验收脚本执行失败
    pause
    exit /b 1
)

echo [4] 生成详细分析报告...
python scripts\generate_d2_analysis_report.py results\ --output reports\d2_analysis.html

echo [5] 创建结果摘要...
python scripts\create_results_summary.py results\ --format markdown --output D2_Results_Summary.md

echo ===============================================
echo      D2验收完成！请查看生成的报告
echo ===============================================
pause