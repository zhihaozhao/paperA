@echo off
setlocal enabledelayedexpansion

:: Quick test script for D3 LOSO experiment fix
:: Tests single enhanced model with seed 0 to verify --out parameter fix

echo ========================================
echo   D3 LOSO Quick Test (Fix Verification)
echo ========================================

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

echo Test Configuration:
echo   Model: enhanced
echo   Seed: 0
echo   Epochs: 5 (quick test)
echo   Protocol: loso
echo   Python Environment: %PYTHON_ENV% (%ENV_TYPE%)

:: Activate conda environment
echo.
echo [ENV] Activating conda environment %PYTHON_ENV%...
call conda activate %PYTHON_ENV%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate conda environment %PYTHON_ENV%
    pause
    exit /b 1
)

:: Switch to project root directory
cd /d "%~dp0\.."
echo [INFO] Project root directory: %CD%

:: Set PYTHONPATH to include project root
set PYTHONPATH=%CD%;%PYTHONPATH%
echo [INFO] PYTHONPATH set to: %CD%

:: Test module import
echo.
echo [TEST] Testing module import...
python -c "import src.train_cross_domain; print('Module import successful')"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Module import failed!
    pause
    exit /b 1
)

:: Create test output directory
set OUTPUT_DIR=results\test_d3_loso
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%" 2>nul

:: Run single test experiment
echo.
echo [TEST] Running single D3 LOSO experiment with --out parameter...
set OUTPUT_FILE=%OUTPUT_DIR%\test_loso_enhanced_seed0.json
echo Command: python -m src.train_cross_domain --model enhanced --seed 0 --epochs 5 --protocol loso --output_dir "%OUTPUT_DIR%" --out "%OUTPUT_FILE%"

python -m src.train_cross_domain --model enhanced --seed 0 --epochs 5 --protocol loso --output_dir "%OUTPUT_DIR%" --out "%OUTPUT_FILE%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Test experiment completed successfully!
    echo [INFO] Output file: %OUTPUT_FILE%
    if exist "%OUTPUT_FILE%" (
        echo [VERIFIED] Output file was created correctly
        type "%OUTPUT_FILE%" | findstr "experiment_type"
    ) else (
        echo [WARNING] Output file not found
    )
) else (
    echo.
    echo [FAILED] Test experiment failed
    echo Please check the error messages above
)

echo.
echo ========================================
echo   D3 LOSO Quick Test Completed
echo ========================================
pause