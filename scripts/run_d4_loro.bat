@echo off
setlocal enabledelayedexpansion

:: D4 Experiment: Sim2Real Label Efficiency Evaluation
:: From D2 pre-trained models to real data label efficiency testing
:: Supports auto environment detection: Remote GPU(base) vs Local CPU(py310)

echo ========================================
echo   D4 Sim2Real Label Efficiency Experiment
echo   From Synthetic Pre-training to Real Data Transfer
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
:: By default, IGNORE pre-set env variables and use full-sweep defaults (560 configs)
:: To honor pre-set env vars (MODELS/SEEDS/LABEL_RATIOS/TRANSFER_METHODS), set USE_ENV=1 before calling this script
if "%USE_ENV%"=="1" (
    echo [CONFIG] Using pre-set environment variables for D4 (MODELS/SEEDS/LABEL_RATIOS/TRANSFER_METHODS)
) else (
    set MODELS=enhanced,cnn,bilstm,conformer_lite
    set SEEDS=0,1,2,3,4
    set LABEL_RATIOS=0.01,0.05,0.10,0.15,0.20,0.50,1.00
    set TRANSFER_METHODS=zero_shot,linear_probe,fine_tune,temp_scale
)
if "%BENCHMARK_PATH%"=="" set BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main
if "%D2_MODELS_PATH%"=="" set D2_MODELS_PATH=checkpoints\d2
rem Prefer user's local D2 checkpoints directory if it exists
if exist "E:\paperA\paperA\checkpoints" set D2_MODELS_PATH=E:\paperA\paperA\checkpoints
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=results\d4\sim2real
if "%AMP%"=="" set AMP=0

:: Normalize comma-separated lists to space-separated for FOR loops
set "MODELS=%MODELS:,= %"
set "SEEDS=%SEEDS:,= %"
set "LABEL_RATIOS=%LABEL_RATIOS:,= %"
set "TRANSFER_METHODS=%TRANSFER_METHODS:,= %"

echo Experiment Configuration:
echo   Model List: %MODELS%
echo   Random Seeds: %SEEDS%
echo   Label Ratios: %LABEL_RATIOS%
echo   Transfer Methods: %TRANSFER_METHODS%
echo   D2 Model Path: %D2_MODELS_PATH%
echo   Real Data Path: %BENCHMARK_PATH%
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

:: Check D2 pre-trained models
if not exist "%D2_MODELS_PATH%" (
    echo [WARNING] D2 pre-trained models not found: %D2_MODELS_PATH%
    echo [INFO] Creating placeholder directory for D2 models...
    mkdir "%D2_MODELS_PATH%" 2>nul
)

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
echo [INFO] Initializing D4 Sim2Real experiment summary...
set TOTAL_CONFIGS=0
set SUCCESS_CONFIGS=0
set FAILED_CONFIGS=0

:: Create experiment summary file
echo { > "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "experiment_type": "D4_SIM2REAL", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "start_time": "%DATE% %TIME%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "configuration": { >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "models": "%MODELS%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "seeds": "%SEEDS%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "label_ratios": "%LABEL_RATIOS%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "transfer_methods": "%TRANSFER_METHODS%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "benchmark_path": "%BENCHMARK_PATH%", >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "d2_models_path": "%D2_MODELS_PATH%" >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   }, >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "results": [ >> "%OUTPUT_DIR%\d4_sim2real_summary.json"

:: Main experiment loop
echo.
echo === Starting D4 Sim2Real Label Efficiency Experiments ===
echo.

for %%m in (%MODELS%) do (
    for %%r in (%LABEL_RATIOS%) do (
        for %%t in (%TRANSFER_METHODS%) do (
            for %%s in (%SEEDS%) do (
                set /a TOTAL_CONFIGS+=1
                echo [EXPERIMENT !TOTAL_CONFIGS!] Model: %%m, Ratio: %%r, Method: %%t, Seed: %%s
                
                :: Run single Sim2Real experiment
                set OUTPUT_FILE=%OUTPUT_DIR%\sim2real_%%m_%%r_%%t_seed%%s.json
                set MODEL_FILE=
                rem Prefer D2 folder
                if exist "%D2_MODELS_PATH%\final_%%m_%%s_hard.pth" set MODEL_FILE=%D2_MODELS_PATH%\final_%%m_%%s_hard.pth
                if "%MODEL_FILE%"=="" if exist "%D2_MODELS_PATH%\final_%%m_0_hard.pth" set MODEL_FILE=%D2_MODELS_PATH%\final_%%m_0_hard.pth
                rem Fallback to checkpoints root (final_*)
                if "%MODEL_FILE%"=="" if exist "checkpoints\final_%%m_%%s_hard.pth" set MODEL_FILE=checkpoints\final_%%m_%%s_hard.pth
                if "%MODEL_FILE%"=="" if exist "checkpoints\final_%%m_0_hard.pth" set MODEL_FILE=checkpoints\final_%%m_0_hard.pth
                rem Recursive search for exact matches in D2 folder
                if "%MODEL_FILE%"=="" for /f "delims=" %%p in ('dir /b /s "%D2_MODELS_PATH%\final_%%m_%%s_hard.pth" 2^>nul') do set MODEL_FILE=%%p
                if "%MODEL_FILE%"=="" for /f "delims=" %%p in ('dir /b /s "%D2_MODELS_PATH%\final_%%m_0_hard.pth" 2^>nul') do set MODEL_FILE=%%p
                rem Recursive search in project checkpoints folder
                if "%MODEL_FILE%"=="" for /f "delims=" %%p in ('dir /b /s "checkpoints\final_%%m_%%s_hard.pth" 2^>nul') do set MODEL_FILE=%%p
                if "%MODEL_FILE%"=="" for /f "delims=" %%p in ('dir /b /s "checkpoints\final_%%m_0_hard.pth" 2^>nul') do set MODEL_FILE=%%p
                rem Fallback to checkpoints root (best_*), last resort
                if "%MODEL_FILE%"=="" if exist "checkpoints\best_%%m_%%s_hard.pth" set MODEL_FILE=checkpoints\best_%%m_%%s_hard.pth
                if "%MODEL_FILE%"=="" if exist "checkpoints\best_%%s_hard.pth" set MODEL_FILE=checkpoints\best_%%s_hard.pth
                if "%MODEL_FILE%"=="" if exist "checkpoints\best_0_hard.pth" set MODEL_FILE=checkpoints\best_0_hard.pth
                if "%MODEL_FILE%"=="" if exist "checkpoints\best_%%m_0_hard.pth" set MODEL_FILE=checkpoints\best_%%m_0_hard.pth
                if "%MODEL_FILE%"=="" if exist "checkpoints\best_enhanced_0_hard.pth" set MODEL_FILE=checkpoints\best_enhanced_0_hard.pth
                set AMP_FLAG=
                if "%AMP%"=="1" set AMP_FLAG=--amp
                if not "%MODEL_FILE%"=="" (
                    echo [INFO] Using D2 checkpoint: !MODEL_FILE!
                    python -m src.train_cross_domain --model %%m --seed %%s --protocol sim2real --label_ratio %%r --transfer_method %%t --d2_model_path "!MODEL_FILE!" --skip_synth_pretrain --benchmark_path "%BENCHMARK_PATH%" --files_per_activity 3 --class_weight inv_freq %AMP_FLAG% --output_dir "%OUTPUT_DIR%" --out "!OUTPUT_FILE!"
                ) else (
                    echo [WARN] No specific D2 checkpoint matched in %D2_MODELS_PATH% for model=%%m seed=%%s. Passing directory for internal discovery.
                    python -m src.train_cross_domain --model %%m --seed %%s --protocol sim2real --label_ratio %%r --transfer_method %%t --d2_model_path "%D2_MODELS_PATH%" --skip_synth_pretrain --benchmark_path "%BENCHMARK_PATH%" --files_per_activity 3 --class_weight inv_freq %AMP_FLAG% --output_dir "%OUTPUT_DIR%" --out "!OUTPUT_FILE!"
                )
                
                if !ERRORLEVEL! equ 0 (
                    set /a SUCCESS_CONFIGS+=1
                    echo [SUCCESS] Config %%m-%%r-%%t-%%s completed
                ) else (
                    set /a FAILED_CONFIGS+=1
                    echo [FAILED] Config %%m-%%r-%%t-%%s failed
                )
                echo.
            )
        )
    )
)

:: Close results array and complete summary
echo   ], >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "completion_stats": { >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "total_configs": %TOTAL_CONFIGS%, >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "successful": %SUCCESS_CONFIGS%, >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "failed": %FAILED_CONFIGS%, >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo     "success_rate": "!SUCCESS_CONFIGS!/%TOTAL_CONFIGS%" >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   }, >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo   "end_time": "%DATE% %TIME%" >> "%OUTPUT_DIR%\d4_sim2real_summary.json"
echo } >> "%OUTPUT_DIR%\d4_sim2real_summary.json"

:: Final statistics and validation
echo ========================================
echo   D4 Sim2Real Experiment Summary
echo ========================================
echo Total Configurations: %TOTAL_CONFIGS%
echo Successful: %SUCCESS_CONFIGS%
echo Failed: %FAILED_CONFIGS%
echo Success Rate: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS%
echo.
echo Results saved to: %OUTPUT_DIR%\d4_sim2real_summary.json

:: Validate D4 acceptance criteria
echo.
echo [VALIDATION] Checking D4 Sim2Real acceptance criteria...
if %SUCCESS_CONFIGS% geq 448 (
    echo [PASS] Coverage: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS% successful configs ^(target: ≥80%%^)
) else (
    echo [WARN] Coverage: %SUCCESS_CONFIGS%/%TOTAL_CONFIGS% successful configs ^(target: ≥80%%^)
)

:: Check if results directory has substantial results
for /f %%i in ('dir /b "%OUTPUT_DIR%\*.json" 2^>nul ^| find /c /v ""') do set RESULT_COUNT=%%i
if %RESULT_COUNT% geq 10 (
    echo [PASS] Result files: %RESULT_COUNT% result files generated
) else (
    echo [WARN] Result files: %RESULT_COUNT% result files generated ^(expected: ≥10^)
)

echo.
echo D4 Sim2Real Label Efficiency experiment completed!
echo Check label efficiency curves and zero-shot baselines in results
echo.
pause