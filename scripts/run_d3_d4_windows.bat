@echo off
setlocal enabledelayedexpansion

:: D3+D4 主控制脚本: Windows conda py310环境下的跨域实验
:: D3: LOSO (Leave-One-Subject-Out) 跨主体实验
:: D4: LORO (Leave-One-Room-Out) 跨房间高难度实验

echo ====================================
echo   WiFi CSI 跨域泛化实验 (D3+D4)
echo   适用于Windows conda py310环境
echo ====================================

:: 设置默认参数
if "%MODEL%"=="" set MODEL=enhanced
if "%PYTHON_ENV%"=="" set PYTHON_ENV=py310
if "%QUICK_MODE%"=="" set QUICK_MODE=0

:: 显示配置
echo 运行配置:
echo   模型: %MODEL%
echo   Python环境: %PYTHON_ENV%
echo   快速模式: %QUICK_MODE%

:: 激活conda环境
echo.
echo [初始化] 激活conda环境 %PYTHON_ENV%...
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [错误] 无法激活conda环境 %PYTHON_ENV%
    echo 请确保已安装conda并创建了py310环境
    pause
    exit /b 1
)

:: 验证Python环境
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] Python不可用，请检查conda环境
    pause
    exit /b 1
)

:: 切换到项目根目录
cd /d "%~dp0\.."
echo [信息] 项目根目录: %CD%

:: 验证项目结构
if not exist "src\train_cross_domain.py" (
    echo [错误] 找不到 src\train_cross_domain.py，请确保在正确的项目目录
    pause
    exit /b 1
)

echo.
echo [验证] Python模块导入测试...
python -c "from src.train_cross_domain import main; from src.train_eval import main; print('模块导入成功')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] Python模块导入失败，请检查依赖安装
    echo 建议运行: pip install numpy torch scikit-learn matplotlib pandas
    pause
    exit /b 1
)

:: 设置实验参数 (基于D3_D4实验计划)
if "%QUICK_MODE%"=="1" (
    echo [快速模式] 使用较少轮数和种子进行测试
    set D3_MODELS=enhanced,cnn
    set D3_EPOCHS=10
    set D3_SEEDS=0,1
    set D4_MODELS=enhanced,cnn
    set D4_SEEDS=0,1
    set D4_LABEL_RATIOS=0.10,0.50,1.00
    set D4_TRANSFER_METHODS=zero_shot,fine_tune
) else (
    echo [完整模式] 使用D3_D4实验计划的完整配置
    set D3_MODELS=enhanced,cnn,bilstm,conformer_lite
    set D3_EPOCHS=100
    set D3_SEEDS=0,1,2,3,4
    set D4_MODELS=enhanced,cnn,bilstm,conformer_lite
    set D4_SEEDS=0,1,2,3,4
    set D4_LABEL_RATIOS=0.01,0.05,0.10,0.15,0.20,0.50,1.00
    set D4_TRANSFER_METHODS=zero_shot,linear_probe,fine_tune,temp_scale
)

:: 询问用户要运行哪些实验
echo.
echo 请选择要运行的实验:
echo [1] D3 LOSO (跨主体泛化)
echo [2] D3 LORO (跨房间泛化)
echo [3] D4 Sim2Real (标签效率)
echo [4] D3全部 (LOSO + LORO)
echo [5] 全部实验 (D3 + D4)
echo [6] 退出
echo.
set /p CHOICE=请输入选择 (1-6): 

if "%CHOICE%"=="1" goto :run_d3_loso
if "%CHOICE%"=="2" goto :run_d3_loro
if "%CHOICE%"=="3" goto :run_d4_sim2real
if "%CHOICE%"=="4" goto :run_d3_all
if "%CHOICE%"=="5" goto :run_all
if "%CHOICE%"=="6" goto :eof
echo [错误] 无效选择，退出
goto :eof

:run_d3_loso
echo.
echo === 开始 D3 LOSO 跨主体实验 ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"
goto :finish

:run_d3_loro
echo.
echo === 开始 D3 LORO 跨房间实验 ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loro.bat"
goto :finish

:run_d4_sim2real
echo.
echo === 开始 D4 Sim2Real 标签效率实验 ===
set MODELS=%D4_MODELS%
set SEEDS=%D4_SEEDS%
call "%~dp0\run_d4_loro.bat"
goto :finish

:run_d3_all
echo.
echo === 开始 D3 LOSO 跨主体实验 ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"

if %ERRORLEVEL% neq 0 (
    echo [错误] D3 LOSO实验失败
    goto :error_finish
)

echo.
echo === 开始 D3 LORO 跨房间实验 ===
call "%~dp0\run_d3_loro.bat"
goto :finish

:run_all
echo.
echo === 开始 D3 LOSO 跨主体实验 ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"

if %ERRORLEVEL% neq 0 (
    echo [错误] D3 LOSO实验失败，跳过后续实验
    goto :error_finish
)

echo.
echo === 开始 D3 LORO 跨房间实验 ===
call "%~dp0\run_d3_loro.bat"

if %ERRORLEVEL% neq 0 (
    echo [错误] D3 LORO实验失败，跳过D4
    goto :error_finish
)

echo.
echo === 开始 D4 Sim2Real 标签效率实验 ===
set MODELS=%D4_MODELS%
set SEEDS=%D4_SEEDS%
call "%~dp0\run_d4_loro.bat"

if %ERRORLEVEL% neq 0 (
    echo [错误] D4实验失败
    goto :error_finish
)

goto :finish

:finish
echo.
echo =====================================
echo   所有选定实验已完成!
echo =====================================
echo.
echo 结果文件位置:
if exist "results\d3\loso" echo   D3 LOSO: results\d3\loso\
if exist "results\d3\loro" echo   D3 LORO: results\d3\loro\
if exist "results\d4\sim2real" echo   D4 Sim2Real: results\d4\sim2real\
echo.
echo 您可以查看以下汇总文件:
if exist "results\d3\loso\d3_loso_summary.json" echo   D3 LOSO汇总: results\d3\loso\d3_loso_summary.json
if exist "results\d3\loro\d3_loro_summary.json" echo   D3 LORO汇总: results\d3\loro\d3_loro_summary.json
if exist "results\d4\sim2real\d4_sim2real_summary.json" echo   D4 Sim2Real汇总: results\d4\sim2real\d4_sim2real_summary.json
echo.
pause
goto :eof

:error_finish
echo.
echo =====================================
echo   实验执行中出现错误!
echo =====================================
echo 请检查上述错误信息并重新运行
pause
exit /b 1