# D3 LOSO (Leave-One-Subject-Out) Cross-Domain Experiments - Windows PowerShell Version
# Compatible with Windows conda prompt environment

param(
    [string]$PYTHON = "python",
    [int]$EPOCHS = 100,
    [string]$EXTRA_ARGS = "",
    [string]$BENCHMARK_PATH = "benchmarks\WiFi-CSI-Sensing-Benchmark-main"
)

# Environment setup
$ErrorActionPreference = "Stop"
$ROOT = Get-Location
$OUT_DIR = Join-Path $ROOT "results\d3\loso"
$BENCHMARK_FULL_PATH = Join-Path $ROOT $BENCHMARK_PATH

# Create output directory
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

# D3 Configuration - based on D2 validated models
$models = @("enhanced", "cnn", "bilstm", "conformer_lite")
$seeds = @(0, 1, 2, 3, 4)  # Consistent with D2

Write-Host "[INFO] Starting D3 LOSO Cross-Domain Experiments..." -ForegroundColor Green
Write-Host "[INFO] Benchmark Path: $BENCHMARK_FULL_PATH" -ForegroundColor Cyan
Write-Host "[INFO] Models: $($models -join ', ')" -ForegroundColor Cyan
Write-Host "[INFO] Seeds: $($seeds -join ', ')" -ForegroundColor Cyan
Write-Host "[INFO] Output Directory: $OUT_DIR" -ForegroundColor Cyan

# Check benchmark dataset availability
if (-not (Test-Path $BENCHMARK_FULL_PATH)) {
    Write-Host "[ERROR] Benchmark dataset not found at $BENCHMARK_FULL_PATH" -ForegroundColor Red
    Write-Host "[ERROR] Please ensure WiFi-CSI-Sensing-Benchmark dataset is available" -ForegroundColor Red
    Write-Host "[INFO] Creating placeholder benchmark directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $BENCHMARK_FULL_PATH | Out-Null
    
    # Create a note file about benchmark requirement
    $benchmarkNote = @"
WiFi-CSI-Sensing-Benchmark Dataset Required
==========================================

This directory should contain the WiFi CSI benchmark dataset.

Expected structure:
- benchmarks/WiFi-CSI-Sensing-Benchmark-main/
  - data.h5 or data.npz (CSI data with labels, subjects, rooms)
  - README.md (dataset description)

If you don't have the benchmark dataset yet, you can:
1. Download from the official source
2. Or run synthetic-only experiments first
3. Or modify the scripts to use your own real CSI data

For now, the scripts will attempt to load data and gracefully handle missing files.
"@
    Set-Content -Path (Join-Path $BENCHMARK_FULL_PATH "README_REQUIRED.txt") -Value $benchmarkNote
}

# Check if D2 pre-trained models exist
$D2_MODELS_DIR = Join-Path $ROOT "checkpoints\d2"
if (-not (Test-Path $D2_MODELS_DIR)) {
    Write-Host "[WARNING] D2 models directory not found at $D2_MODELS_DIR" -ForegroundColor Yellow
    Write-Host "[WARNING] Creating directory and will train from scratch" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $D2_MODELS_DIR | Out-Null
}

# Run LOSO experiments
$total_runs = $models.Count * $seeds.Count
$current_run = 0

foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $current_run++
        Write-Host ""
        Write-Host "[RUN $current_run/$total_runs] LOSO: model=$model, seed=$seed" -ForegroundColor Yellow
        
        # Construct command
        $cmd = "$PYTHON -m src.train_cross_domain --model $model --protocol loso --benchmark_path `"$BENCHMARK_FULL_PATH`" --seed $seed --epochs $EPOCHS --output_dir `"$OUT_DIR`""
        
        if ($EXTRA_ARGS) {
            $cmd += " $EXTRA_ARGS"
        }
        
        Write-Host "[CMD] $cmd" -ForegroundColor Gray
        
        try {
            Invoke-Expression $cmd
            Write-Host "[OK] Completed LOSO for $model seed $seed" -ForegroundColor Green
        }
        catch {
            Write-Host "[ERROR] Failed LOSO for $model seed $seed`: $_" -ForegroundColor Red
            Write-Host "[INFO] Continuing with next experiment..." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "[INFO] D3 LOSO experiments completed!" -ForegroundColor Green
Write-Host "[INFO] Results saved to: $OUT_DIR" -ForegroundColor Cyan
Write-Host "[INFO] Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run validation: python scripts\validate_d3_acceptance.py --protocol loso"
Write-Host "  2. Generate summary: python scripts\export_d3_summary.py --protocol loso"
Write-Host "  3. Run LORO experiments: PowerShell scripts\run_d3_loro.ps1"

# Generate quick summary if Python is available
try {
    $summaryScript = @"
import json, glob
from pathlib import Path

loso_files = glob.glob('$($OUT_DIR.Replace('\', '/'))/*.json')
print(f'Generated {len(loso_files)} LOSO result files')

if loso_files:
    sample_file = loso_files[0]
    with open(sample_file) as f:
        data = json.load(f)
    
    n_folds = len(data.get('fold_results', []))
    model = data.get('model', 'unknown')
    
    print(f'Sample: {model} with {n_folds} LOSO folds')
    
    if 'aggregate_stats' in data:
        stats = data['aggregate_stats']
        if 'macro_f1' in stats:
            print(f'  Macro F1: {stats["macro_f1"]["mean"]:.3f}±{stats["macro_f1"]["std"]:.3f}')
        if 'falling_f1' in stats:
            print(f'  Falling F1: {stats["falling_f1"]["mean"]:.3f}±{stats["falling_f1"]["std"]:.3f}')
"@

    Write-Host ""
    Write-Host "[INFO] Generating quick summary..." -ForegroundColor Cyan
    echo $summaryScript | & $PYTHON
}
catch {
    Write-Host "[INFO] Quick summary generation skipped (Python not available or no results yet)"
}