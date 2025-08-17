# D3 Single Experiment Test - Windows PowerShell Version
# Tests a single LOSO experiment to verify setup before running full suite

param(
    [string]$MODEL = "enhanced",
    [string]$PROTOCOL = "loso", 
    [int]$SEED = 0,
    [int]$EPOCHS = 10,  # Reduced for quick testing
    [string]$PYTHON = "python"
)

Write-Host "D3 Single Experiment Test" -ForegroundColor Green
Write-Host "=" * 30

$ROOT = Get-Location
$BENCHMARK_PATH = Join-Path $ROOT "benchmarks\WiFi-CSI-Sensing-Benchmark-main"
$OUT_DIR = Join-Path $ROOT "results\d3\test"

# Create test output directory
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

Write-Host "[INFO] Testing single $PROTOCOL experiment..." -ForegroundColor Cyan
Write-Host "[INFO] Model: $MODEL" -ForegroundColor Cyan
Write-Host "[INFO] Protocol: $PROTOCOL" -ForegroundColor Cyan
Write-Host "[INFO] Seed: $SEED" -ForegroundColor Cyan
Write-Host "[INFO] Epochs: $EPOCHS (reduced for testing)" -ForegroundColor Cyan

# Test command
$cmd = "$PYTHON -m src.train_cross_domain --model $MODEL --protocol $PROTOCOL --benchmark_path `"$BENCHMARK_PATH`" --seed $SEED --epochs $EPOCHS --output_dir `"$OUT_DIR`""

Write-Host "`n[CMD] $cmd" -ForegroundColor Gray

try {
    Write-Host "`n[INFO] Running test experiment..." -ForegroundColor Yellow
    Invoke-Expression $cmd
    
    # Check if output was created
    $outputPattern = Join-Path $OUT_DIR "$PROTOCOL" "$($PROTOCOL)_$($MODEL)_seed$($SEED).json"
    if (Test-Path $outputPattern) {
        Write-Host "`n✅ Test experiment completed successfully!" -ForegroundColor Green
        Write-Host "✅ Output file created: $outputPattern" -ForegroundColor Green
        
        # Show result summary
        try {
            $resultData = Get-Content $outputPattern | ConvertFrom-Json
            $stats = $resultData.aggregate_stats
            if ($stats) {
                Write-Host "`n📊 Test Results:" -ForegroundColor Cyan
                if ($stats.macro_f1) {
                    Write-Host "  Macro F1: $($stats.macro_f1.mean)" -ForegroundColor White
                }
                if ($stats.falling_f1) {
                    Write-Host "  Falling F1: $($stats.falling_f1.mean)" -ForegroundColor White
                }
            }
        } catch {
            Write-Host "  (Result details parsing failed, but file exists)" -ForegroundColor Yellow
        }
        
        Write-Host "`n🚀 Ready to run full experiments:" -ForegroundColor Green
        Write-Host "  PowerShell .\scripts\run_d3_loso.ps1" -ForegroundColor White
        Write-Host "  PowerShell .\scripts\run_d3_loro.ps1" -ForegroundColor White
        
    } else {
        Write-Host "`n⚠️  Test completed but no output file found" -ForegroundColor Yellow
        Write-Host "   This may indicate data loading issues" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "`n❌ Test experiment failed: $_" -ForegroundColor Red
    
    Write-Host "`n🔧 Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Check Python environment:" -ForegroundColor White
    Write-Host "   python --version" -ForegroundColor Gray
    Write-Host "   python -c `"import torch, numpy; print('Dependencies OK')`"" -ForegroundColor Gray
    
    Write-Host "`n2. Check source files:" -ForegroundColor White
    Write-Host "   dir src\*.py" -ForegroundColor Gray
    
    Write-Host "`n3. Run prerequisites check:" -ForegroundColor White
    Write-Host "   PowerShell .\scripts\check_d3_prerequisites.ps1" -ForegroundColor Gray
    
    Write-Host "`n4. Check for detailed error in the output above" -ForegroundColor White
}

Write-Host "`n" + "=" * 50