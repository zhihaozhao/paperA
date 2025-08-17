@echo off
setlocal enabledelayedexpansion

:: Benchmark Data Checker for WiFi-CSI-Sensing-Benchmark
:: Diagnoses data path and file availability issues

echo ========================================
echo   Benchmark Data Checker
echo   WiFi-CSI-Sensing-Benchmark Diagnosis
echo ========================================

:: Switch to project root
cd /d "%~dp0\.."
echo [INFO] Project root: %CD%

:: Check multiple possible benchmark paths
set PATHS[0]=benchmarks\WiFi-CSI-Sensing-Benchmark-main
set PATHS[1]=benchmarks\WiFi-CSI-Sensing-Benchmark-main\Data
set PATHS[2]=benchmarks\wifi_csi_benchmark
set PATHS[3]=benchmarks\wifi_csi_benchmark\Data

echo.
echo [CHECK] Searching for benchmark directories...

for /L %%i in (0,1,3) do (
    set PATH_TO_CHECK=!PATHS[%%i]!
    echo.
    echo Checking: !PATH_TO_CHECK!
    
    if exist "!PATH_TO_CHECK!" (
        echo   [FOUND] Directory exists
        
        :: Check for expected dataset subdirectories
        set DATASETS[0]=NTU-Fi_HAR
        set DATASETS[1]=UT_HAR
        set DATASETS[2]=Widardata
        set DATASETS[3]=NTU-Fi-HumanID
        
        for /L %%j in (0,1,3) do (
            set DATASET_DIR=!PATH_TO_CHECK!\!DATASETS[%%j]!
            if exist "!DATASET_DIR!" (
                echo   [DATASET] !DATASETS[%%j]! - FOUND
                
                :: Count data files in this dataset
                for /f %%k in ('dir /b /s "!DATASET_DIR!\*.h5" "!DATASET_DIR!\*.npz" "!DATASET_DIR!\*.csv" "!DATASET_DIR!\*.mat" 2^>nul ^| find /c /v ""') do (
                    echo     ^> %%k data files found
                )
            ) else (
                echo   [DATASET] !DATASETS[%%j]! - MISSING
            )
        )
        
        :: Check for any data files in root directory
        for /f %%k in ('dir /b "!PATH_TO_CHECK!\*.h5" "!PATH_TO_CHECK!\*.npz" "!PATH_TO_CHECK!\*.csv" 2^>nul ^| find /c /v ""') do (
            if %%k gtr 0 echo   [FILES] %%k data files in root directory
        )
        
    ) else (
        echo   [MISSING] Directory does not exist
    )
)

:: Provide guidance based on findings
echo.
echo ========================================
echo   Recommendations
echo ========================================

if exist "benchmarks\WiFi-CSI-Sensing-Benchmark-main\Data\NTU-Fi_HAR" (
    echo [SUCCESS] Benchmark data found! Use this path in scripts:
    echo   BENCHMARK_PATH=benchmarks\WiFi-CSI-Sensing-Benchmark-main\Data
) else if exist "benchmarks\wifi_csi_benchmark\Data\NTU-Fi_HAR" (
    echo [SUCCESS] Benchmark data found! Use this path in scripts:
    echo   BENCHMARK_PATH=benchmarks\wifi_csi_benchmark\Data
) else (
    echo [SETUP NEEDED] Benchmark data not found. Please:
    echo.
    echo 1. Download from: https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark
    echo 2. Extract to: benchmarks\WiFi-CSI-Sensing-Benchmark-main\
    echo 3. Ensure Data subdirectory contains: NTU-Fi_HAR, UT_HAR, Widardata, NTU-Fi-HumanID
    echo.
    echo Alternative: Clone directly:
    echo   git clone https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark.git benchmarks\WiFi-CSI-Sensing-Benchmark-main
    echo.
    echo [CURRENT] Scripts will use realistic synthetic data as fallback
    echo Expected results with synthetic fallback: F1=0.7-0.9 ^(more realistic than F1=1.0^)
)

echo.
echo ========================================
pause