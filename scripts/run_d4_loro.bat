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

echo.
echo [ENV] Activating conda environment %PYTHON_ENV%...
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate conda environment %PYTHON_ENV%
    pause
    exit /b 1
)

cd /d "%~dp0\.."
echo [INFO] Project root directory: %CD%
set PYTHONPATH=%CD%;%PYTHONPATH%

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" 2>nul

echo.
echo === Starting D4 Sim2Real Label Efficiency Experiments ===
echo.

for %%m in (%MODELS%) do (
  for %%r in (%LABEL_RATIOS%) do (
    for %%t in (%TRANSFER_METHODS%) do (
      for %%s in (%SEEDS%) do (
        set OUTPUT_FILE=%OUTPUT_DIR%\sim2real_%%m_%%r_%%t_seed%%s.json
        if exist "!OUTPUT_FILE!" (
          echo [SKIP] Exists: !OUTPUT_FILE! (%%m, %%r, %%t, seed=%%s)
        ) else (
          set MODEL_FILE=
          if exist "%D2_MODELS_PATH%\final_%%m_%%s_hard.pth" set MODEL_FILE=%D2_MODELS_PATH%\final_%%m_%%s_hard.pth
          if "%%MODEL_FILE%%"=="" if exist "%D2_MODELS_PATH%\final_%%m_0_hard.pth" set MODEL_FILE=%D2_MODELS_PATH%\final_%%m_0_hard.pth
          if "%%MODEL_FILE%%"=="" if exist "checkpoints\final_%%m_%%s_hard.pth" set MODEL_FILE=checkpoints\final_%%m_%%s_hard.pth
          if "%%MODEL_FILE%%"=="" if exist "checkpoints\final_%%m_0_hard.pth" set MODEL_FILE=checkpoints\final_%%m_0_hard.pth
          if "%%MODEL_FILE%%"=="" for /f "delims=" %%p in ('dir /b /s "%D2_MODELS_PATH%\final_%%m_%%s_hard.pth" 2^>nul') do set MODEL_FILE=%%p
          if "%%MODEL_FILE%%"=="" for /f "delims=" %%p in ('dir /b /s "%D2_MODELS_PATH%\final_%%m_0_hard.pth" 2^>nul') do set MODEL_FILE=%%p
          if "%%MODEL_FILE%%"=="" for /f "delims=" %%p in ('dir /b /s "checkpoints\final_%%m_%%s_hard.pth" 2^>nul') do set MODEL_FILE=%%p
          if "%%MODEL_FILE%%"=="" for /f "delims=" %%p in ('dir /b /s "checkpoints\final_%%m_0_hard.pth" 2^>nul') do set MODEL_FILE=%%p
          set AMP_FLAG=
          if "%AMP%"=="1" set AMP_FLAG=--amp
          if not "%%MODEL_FILE%%"=="" (
            echo [INFO] Using D2 checkpoint: %%MODEL_FILE%%
            python -m src.train_cross_domain --resume --model %%m --seed %%s --protocol sim2real --label_ratio %%r --transfer_method %%t --d2_model_path "%%MODEL_FILE%%" --skip_synth_pretrain --benchmark_path "%BENCHMARK_PATH%" --files_per_activity 3 --class_weight inv_freq %AMP_FLAG% --output_dir "%OUTPUT_DIR%" --out "!OUTPUT_FILE!"
          ) else (
            set EXPECTED_FILE=%D2_MODELS_PATH%\final_%%m_%%s_hard.pth
            echo [WARN] No specific D2 checkpoint matched in %D2_MODELS_PATH% for model=%%m seed=%%s. Passing directory.
            echo [INFO] Expected exact checkpoint: !EXPECTED_FILE!
            python -m src.train_cross_domain --resume --model %%m --seed %%s --protocol sim2real --label_ratio %%r --transfer_method %%t --d2_model_path "%D2_MODELS_PATH%" --skip_synth_pretrain --benchmark_path "%BENCHMARK_PATH%" --files_per_activity 3 --class_weight inv_freq %AMP_FLAG% --output_dir "%OUTPUT_DIR%" --out "!OUTPUT_FILE!"
          )
        )
        echo.
      )
    )
  )
)

echo Done.