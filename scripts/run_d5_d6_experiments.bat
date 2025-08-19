@echo off
setlocal enabledelayedexpansion

rem === D5 & D6 Experiment Runner ===
rem Usage: run_d5_d6_experiments.bat [smoke|full]
rem Default: smoke (quick test)

set "MODE=%1"
if "%MODE%"=="" set "MODE=smoke"

rem === Python interpreter discovery ===
if not defined PY_EXE (
  for %%P in (
    "D:\\workspace_AI\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\ProgramData\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\Users\\%USERNAME%\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\Users\\%USERNAME%\\miniconda3\\envs\\py310\\python.exe"
  ) do (
    if exist %%~P set "PY_EXE=%%~P" & goto :py_found
  )
  set "PY_EXE=python"
)
:py_found
echo [INFO] Using Python: %PY_EXE%

echo ========================================
echo Starting D5 & D6 Experiments - Mode: %MODE%
echo ========================================

rem === Step 1: D5 Smoke Test (if smoke mode) ===
if "%MODE%"=="smoke" (
  echo.
  echo [STEP 1] Running D5 Smoke Test...
  echo [INFO] 2 models x 2 seeds = 4 runs (quick validation)
  
  "%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d5_smoke_spec.json --dry_run
  if errorlevel 1 (
    echo [ERROR] Dry run failed!
    pause
    exit /b 1
  )
  
  "%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d5_smoke_spec.json --resume
  if errorlevel 1 (
    echo [ERROR] D5 smoke test failed!
    pause
    exit /b 1
  )
  
  echo [SUCCESS] D5 smoke test completed!
  goto :d6_smoke
)

rem === Step 1: D5 Full Experiment ===
if "%MODE%"=="full" (
  echo.
  echo [STEP 1] Running D5 Full Ablation Study...
  echo [INFO] 4 models x 5 seeds = 20 runs (full ablation)
  
  "%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d5_full_spec.json --dry_run
  if errorlevel 1 (
    echo [ERROR] Dry run failed!
    pause
    exit /b 1
  )
  
  "%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d5_full_spec.json --resume
  if errorlevel 1 (
    echo [ERROR] D5 full experiment failed!
    pause
    exit /b 1
  )
  
  echo [SUCCESS] D5 full experiment completed!
  goto :d6_full
)

rem === Step 2: D6 Smoke Test ===
:d6_smoke
echo.
echo [STEP 2] Running D6 Smoke Test...
echo [INFO] 2 models x 2 seeds = 4 runs (robustness validation)

"%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d6_spec.json --dry_run
if errorlevel 1 (
  echo [ERROR] D6 dry run failed!
  pause
  exit /b 1
)

"%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d6_spec.json --resume
if errorlevel 1 (
  echo [ERROR] D6 smoke test failed!
  pause
  exit /b 1
)

echo [SUCCESS] D6 smoke test completed!
goto :analysis

rem === Step 2: D6 Full Experiment ===
:d6_full
echo.
echo [STEP 2] Running D6 Full Robustness Analysis...
echo [INFO] 2 models x 2 seeds = 4 runs (full robustness)

"%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d6_spec.json --dry_run
if errorlevel 1 (
  echo [ERROR] D6 dry run failed!
  pause
  exit /b 1
)

"%PY_EXE%" scripts\run_sweep_from_json.py --spec scripts\d6_spec.json --resume
if errorlevel 1 (
  echo [ERROR] D6 full experiment failed!
  pause
  exit /b 1
)

echo [SUCCESS] D6 full experiment completed!

rem === Step 3: Analysis and Summary ===
:analysis
echo.
echo [STEP 3] Generating Analysis and Summary...

rem Export results summary
"%PY_EXE%" scripts\export_summary.py --pattern "results_gpu\d5\*.json" --out_csv results\metrics\d5_summary.csv
"%PY_EXE%" scripts\export_summary.py --pattern "results_gpu\d6\*.json" --out_csv results\metrics\d6_summary.csv
if errorlevel 1 (
  echo [WARNING] Export summary failed, but continuing...
)

rem Generate plots (if plotting scripts exist)
if exist "src\plotting.py" (
  "%PY_EXE%" -c "import src.plotting as p; p.plot_ablation_results('results_gpu/d5', 'plots/d5_ablation.pdf')"
  if errorlevel 1 (
    echo [WARNING] D5 plot generation failed, but continuing...
  )
  "%PY_EXE%" -c "import src.plotting as p; p.plot_robustness_results('results_gpu/d6', 'plots/d6_robustness.pdf')"
  if errorlevel 1 (
    echo [WARNING] D6 plot generation failed, but continuing...
  )
)

echo.
echo ========================================
echo D5 & D6 Experiments Completed!
echo ========================================
echo.
echo Results locations:
echo   D5: results_gpu\d5\
echo   D6: results_gpu\d6\
echo Summary CSVs: results\metrics\d5_summary.csv, results\metrics\d6_summary.csv
echo.
echo Next steps:
echo 1. Check results_gpu\d5\ and results_gpu\d6\ for individual JSON files
echo 2. Review results\metrics\d5_summary.csv and results\metrics\d6_summary.csv
echo 3. Run acceptance criteria validation
echo.

pause
