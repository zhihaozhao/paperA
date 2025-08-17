@echo off
setlocal enabledelayedexpansion

:: D3 Experiment: LORO (Leave-One-Room-Out) Cross-Room Generalization
:: Based on WiFi-CSI-Sensing-Benchmark real data
:: Supports auto environment detection: Remote GPU(base) vs Local CPU(py310)

echo ========================================
echo   D3 LORO Cross-Room Generalization
echo   Based on WiFi CSI Benchmark Real Data
echo ========================================

:: Auto environment detection (if not set in parent script)
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

:: Set default parameters (based on D3_D4 experiment plan)
if "%MODELS%"=="" set MODELS=enhanced,cnn,bilstm,conformer_lite
if "%SEEDS%"=="" set SEEDS=0,1,2,3,4
if "%EPOCHS%"=="" set EPOCHS=100
if "%BENCHMARK_PATH%"=="" set BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main\Data
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\d3\loro

echo Experiment Configuration:
echo   Model List: %MODELS%
echo   Random Seeds: %SEEDS%
echo   Training Epochs: %EPOCHS%
echo   Dataset Path: %BENCHMARK_PATH%
echo   Output Directory: %OUTPUT_DIR%
echo   Python Environment: %PYTHON_ENV% (%ENV_TYPE%)

:: Activate conda environment
echo.
echo [ENV] Activating conda environment %PYTHON_ENV%...
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

:: Switch to project root directory
cd /d "%~dp0\.."
echo [INFO] Project root directory: %CD%

:: Set PYTHONPATH to include project root
set PYTHONPATH=%CD%;%PYTHONPATH%
echo [INFO] PYTHONPATH set to: %CD%

:: Check benchmark dataset
if not exist "%BENCHMARK_PATH%" (
    echo [WARNING] Benchmark dataset not found: %BENCHMARK_PATH%
    echo [INFO] Creating placeholder directory and continuing experiment...
    mkdir "%BENCHMARK_PATH%" 2>nul
)

:: Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" 2>nul
echo [INFO] Output directory: %OUTPUT_DIR%

:: Initialize experiment summary
echo [INFO] Initializing D3 LORO experiment summary...
set TOTAL_CONFIGS=0
set SUCCESS_CONFIGS=0
set FAILED_CONFIGS=0

:: Create experiment summary file
echo { > "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "experiment_type": "D3_LORO", >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "start_time": "%DATE% %TIME%", >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "configuration": { >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "models": "%MODELS%", >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "seeds": "%SEEDS%", >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "epochs": %EPOCHS%, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "protocol": "loro", >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "benchmark_path": "%BENCHMARK_PATH%" >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   }, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "results": [ >> "%OUTPUT_DIR%\d3_loro_summary.json"

:: Main experiment loop
echo.
echo === Starting D3 LORO Cross-Room Experiments ===
echo.

for %%m in (%MODELS%) do (
    for %%s in (%SEEDS%) do (
        set /a TOTAL_CONFIGS+=1
        echo [EXPERIMENT !TOTAL_CONFIGS!] Model: %%m, Seed: %%s, Protocol: loro
        
        :: Run single experiment
        set OUTPUT_FILE=%OUTPUT_DIR%\loro_%%m_seed%%s.json
        python -m src.train_cross_domain --model %%m --seed %%s --epochs %EPOCHS% --protocol loro --benchmark_path "%BENCHMARK_PATH%" --output_dir "%OUTPUT_DIR%" --out "!OUTPUT_FILE!"
        
        if !ERRORLEVEL! equ 0 (
            set /a SUCCESS_CONFIGS+=1
            echo [SUCCESS] Model %%m, Seed %%s completed successfully
        ) else (
            set /a FAILED_CONFIGS+=1
            echo [FAILED] Model %%m, Seed %%s failed
        )
        echo.
    )
)

:: Close results array and complete summary
echo   ], >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "completion_stats": { >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "total_configs": %TOTAL_CONFIGS%, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "successful": %SUCCESS_CONFIGS%, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "failed": %FAILED_CONFIGS%, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo     "success_rate": "!SUCCESS_CONFIGS!/%TOTAL_CONFIGS%" >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   }, >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo   "end_time": "%DATE% %TIME%" >> "%OUTPUT_DIR%\d3_loro_summary.json"
echo } >> "%OUTPUT_DIR%\d3_loro_summary.json"

:: Final statistics and validation
echo ========================================
echo   D3 LORO Experiment Summary
echo ========================================
echo Total Configurations: %TOTAL_CONFIGS%
echo Successful: %SUCCESS_CONFIGS%
echo Failed: %FAILED_CONFIGS%
echo Success Rate: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS%
echo.
echo Results saved to: %OUTPUT_DIR%\d3_loro_summary.json

:: Validate D3 acceptance criteria
echo.
echo [VALIDATION] Checking D3 LORO acceptance criteria...
if %SUCCESS_CONFIGS% geq 16 (
    echo [PASS] Coverage: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS% successful configs ^(target: ≥80%%^)
) else (
    echo [FAIL] Coverage: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS% successful configs ^(target: ≥80%%^)
)

:: Check if results directory has substantial results
for /f %%i in ('dir /b "%OUTPUT_DIR%\*.json" 2^>nul ^| find /c /v ""') do set RESULT_COUNT=%%i
if %RESULT_COUNT% geq 5 (
    echo [PASS] Result files: %RESULT_COUNT% result files generated
) else (
    echo [WARN] Result files: %RESULT_COUNT% result files generated ^(expected: ≥5^)
)

echo.
echo D3 LORO Cross-Room experiment completed!
echo Next step: Run D4 Sim2Real label efficiency experiment
echo.
pause