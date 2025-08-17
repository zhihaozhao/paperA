@echo off
setlocal enabledelayedexpansion

:: D3+D4 Main Control Script: Cross-domain experiments on Windows conda environment
:: D3: LOSO/LORO Cross-domain generalization experiments (cross-subject/cross-room)
:: D4: Sim2Real Label efficiency evaluation (synthetic to real transfer)
:: Supports auto environment detection: Remote GPU(base) vs Local CPU(py310)

echo ====================================
echo   WiFi CSI Cross-Domain Experiments (D3+D4)
echo   Supports Remote GPU and Local CPU environments
echo ====================================

:: Auto environment detection
if "%PYTHON_ENV%"=="" (
    if exist "D:\anaconda\python.exe" (
        echo [DETECTED] Remote GPU environment - using conda base
        set PYTHON_ENV=base
        set ENV_TYPE=remote_gpu
    ) else if exist "D:\workspace_AI\Anaconda3\envs\py310\python.exe" (
        echo [DETECTED] Local CPU environment - using conda py310
        set PYTHON_ENV=py310
        set ENV_TYPE=local_cpu
    ) else (
        echo [DEFAULT] Using py310 environment
        set PYTHON_ENV=py310
        set ENV_TYPE=default
    )
)

:: Set default parameters
if "%MODEL%"=="" set MODEL=enhanced
if "%QUICK_MODE%"=="" set QUICK_MODE=0

:: Display configuration
echo Runtime Configuration:
echo   Model: %MODEL%
echo   Python Environment: %PYTHON_ENV% (%ENV_TYPE%)
echo   Quick Mode: %QUICK_MODE%

:: Activate conda environment
echo.
echo [INIT] Activating conda environment %PYTHON_ENV%...
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate conda environment %PYTHON_ENV%
    if "%ENV_TYPE%"=="remote_gpu" (
        echo Remote GPU environment: Please ensure conda base environment is available
    ) else (
        echo Local CPU environment: Please ensure conda is installed with py310 environment
    )
    pause
    exit /b 1
)

:: Verify Python environment
python --version 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not available, please check conda environment
    pause
    exit /b 1
)

:: Switch to project root directory
cd /d "%~dp0\.."
echo [INFO] Project root directory: %CD%

:: Verify project structure
if not exist "src\train_cross_domain.py" (
    echo [ERROR] Cannot find src\train_cross_domain.py, please ensure correct project directory
    pause
    exit /b 1
)

echo.
echo [VERIFY] Testing Python module imports...
python -c "import src.train_cross_domain; print('Module import successful')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python module import failed, please check dependency installation
    echo Suggestion: pip install numpy torch scikit-learn matplotlib pandas
    pause
    exit /b 1
)

:: Set experiment parameters (based on D3_D4 experiment plan)
if "%QUICK_MODE%"=="1" (
    echo [QUICK MODE] Using reduced epochs and seeds for testing
    set D3_MODELS=enhanced,cnn
    set D3_EPOCHS=10
    set D3_SEEDS=0,1
    set D4_MODELS=enhanced,cnn
    set D4_SEEDS=0,1
    set D4_LABEL_RATIOS=0.10,0.50,1.00
    set D4_TRANSFER_METHODS=zero_shot,fine_tune
) else (
    echo [FULL MODE] Using complete D3_D4 experiment plan configuration
    set D3_MODELS=enhanced,cnn,bilstm,conformer_lite
    set D3_EPOCHS=100
    set D3_SEEDS=0,1,2,3,4
    set D4_MODELS=enhanced,cnn,bilstm,conformer_lite
    set D4_SEEDS=0,1,2,3,4
    set D4_LABEL_RATIOS=0.01,0.05,0.10,0.15,0.20,0.50,1.00
    set D4_TRANSFER_METHODS=zero_shot,linear_probe,fine_tune,temp_scale
)

:: Ask user which experiments to run
echo.
echo Please select experiments to run:
echo [1] D3 LOSO (Cross-Subject Generalization)
echo [2] D3 LORO (Cross-Room Generalization)
echo [3] D4 Sim2Real (Label Efficiency)
echo [4] All D3 (LOSO + LORO)
echo [5] All Experiments (D3 + D4)
echo [6] Exit
echo.
set /p CHOICE=Please enter choice (1-6): 

if "%CHOICE%"=="1" goto :run_d3_loso
if "%CHOICE%"=="2" goto :run_d3_loro
if "%CHOICE%"=="3" goto :run_d4_sim2real
if "%CHOICE%"=="4" goto :run_d3_all
if "%CHOICE%"=="5" goto :run_all
if "%CHOICE%"=="6" goto :eof
echo [ERROR] Invalid choice, exiting
goto :eof

:run_d3_loso
echo.
echo === Starting D3 LOSO Cross-Subject Experiments ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"
goto :finish

:run_d3_loro
echo.
echo === Starting D3 LORO Cross-Room Experiments ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loro.bat"
goto :finish

:run_d4_sim2real
echo.
echo === Starting D4 Sim2Real Label Efficiency Experiments ===
set MODELS=%D4_MODELS%
set SEEDS=%D4_SEEDS%
call "%~dp0\run_d4_loro.bat"
goto :finish

:run_d3_all
echo.
echo === Starting D3 LOSO Cross-Subject Experiments ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] D3 LOSO experiments failed
    goto :error_finish
)

echo.
echo === Starting D3 LORO Cross-Room Experiments ===
call "%~dp0\run_d3_loro.bat"
goto :finish

:run_all
echo.
echo === Starting D3 LOSO Cross-Subject Experiments ===
set MODELS=%D3_MODELS%
set EPOCHS=%D3_EPOCHS%
set SEEDS=%D3_SEEDS%
call "%~dp0\run_d3_loso.bat"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] D3 LOSO experiments failed, skipping subsequent experiments
    goto :error_finish
)

echo.
echo === Starting D3 LORO Cross-Room Experiments ===
call "%~dp0\run_d3_loro.bat"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] D3 LORO experiments failed, skipping D4
    goto :error_finish
)

echo.
echo === Starting D4 Sim2Real Label Efficiency Experiments ===
set MODELS=%D4_MODELS%
set SEEDS=%D4_SEEDS%
call "%~dp0\run_d4_loro.bat"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] D4 experiments failed
    goto :error_finish
)

goto :finish

:finish
echo.
echo =====================================
echo   All Selected Experiments Completed!
echo =====================================
echo.
echo Result file locations:
if exist "results\d3\loso" echo   D3 LOSO: results\d3\loso\
if exist "results\d3\loro" echo   D3 LORO: results\d3\loro\
if exist "results\d4\sim2real" echo   D4 Sim2Real: results\d4\sim2real\
echo.
echo You can check the following summary files:
if exist "results\d3\loso\d3_loso_summary.json" echo   D3 LOSO Summary: results\d3\loso\d3_loso_summary.json
if exist "results\d3\loro\d3_loro_summary.json" echo   D3 LORO Summary: results\d3\loro\d3_loro_summary.json
if exist "results\d4\sim2real\d4_sim2real_summary.json" echo   D4 Sim2Real Summary: results\d4\sim2real\d4_sim2real_summary.json
echo.
pause
goto :eof

:error_finish
echo.
echo =====================================
echo   Error occurred during experiments!
echo =====================================
echo Please check the above error messages and re-run
pause
exit /b 1