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

:: 设置快速模式参数
if "%QUICK_MODE%"=="1" (
    echo [快速模式] 使用较少轮数和种子进行测试
    set D3_EPOCHS=5
    set D3_SEEDS=0,1
    set D4_EPOCHS=5
    set D4_SEEDS=0,1
) else (
    echo [完整模式] 使用完整轮数和种子
    set D3_EPOCHS=20
    set D3_SEEDS=0,1,2
    set D4_EPOCHS=25
    set D4_SEEDS=0,1,2,3
)

:: 询问用户要运行哪些实验
echo.
echo 请选择要运行的实验:
echo [1] 仅 D3 LOSO 实验
echo [2] 仅 D4 LORO 实验  
echo [3] D3 + D4 全部实验
echo [4] 退出
echo.
set /p CHOICE=请输入选择 (1-4): 

if "%CHOICE%"=="1" goto :run_d3
if "%CHOICE%"=="2" goto :run_d4
if "%CHOICE%"=="3" goto :run_both
if "%CHOICE%"=="4" goto :eof
echo [错误] 无效选择，退出
goto :eof

:run_d3
echo.
echo === 开始 D3 LOSO 实验 ===
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"
goto :finish

:run_d4
echo.
echo === 开始 D4 LORO 实验 ===
set EPOCHS=%D4_EPOCHS%
set SEEDS=%D4_SEEDS%
call "%~dp0\run_d4_loro.bat"
goto :finish

:run_both
echo.
echo === 开始 D3 LOSO 实验 ===
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"

if %ERRORLEVEL% neq 0 (
    echo [错误] D3实验失败，跳过D4
    goto :error_finish
)

echo.
echo === 开始 D4 LORO 实验 ===
set EPOCHS=%D4_EPOCHS%
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
if exist "results\loso" echo   D3 LOSO: results\loso\
if exist "results\loro" echo   D4 LORO: results\loro\
echo.
echo 您可以查看以下汇总文件:
if exist "results\loso\d3_loso_summary.json" echo   D3汇总: results\loso\d3_loso_summary.json
if exist "results\loro\d4_loro_summary.json" echo   D4汇总: results\loro\d4_loro_summary.json
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