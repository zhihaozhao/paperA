# D3 Experiments Prerequisites Check - Windows PowerShell Version
# Checks requirements and provides setup guidance for D3 cross-domain experiments

Write-Host "D3 Cross-Domain Experiments Prerequisites Check" -ForegroundColor Green
Write-Host "=" * 50

$ROOT = Get-Location
$allGood = $true

# Check 1: Python environment
Write-Host "`n[CHECK 1] Python Environment" -ForegroundColor Cyan
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python available: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python not found in PATH" -ForegroundColor Red
        $allGood = $false
    }
} catch {
    Write-Host "❌ Python not available" -ForegroundColor Red
    $allGood = $false
}

# Check 2: Required Python packages
Write-Host "`n[CHECK 2] Python Dependencies" -ForegroundColor Cyan
$requiredPackages = @("torch", "numpy")

foreach ($package in $requiredPackages) {
    try {
        $result = & python -c "import $package; print('OK')" 2>&1
        if ($LASTEXITCODE -eq 0 -and $result -eq "OK") {
            Write-Host "✅ $package available" -ForegroundColor Green
        } else {
            Write-Host "❌ $package not found" -ForegroundColor Red
            $allGood = $false
        }
    } catch {
        Write-Host "❌ $package not available" -ForegroundColor Red
        $allGood = $false
    }
}

# Check 3: Source code files
Write-Host "`n[CHECK 3] Source Code Files" -ForegroundColor Cyan
$requiredFiles = @(
    "src\models.py",
    "src\data_real.py", 
    "src\metrics.py",
    "src\train_cross_domain.py"
)

foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $ROOT $file
    if (Test-Path $fullPath) {
        Write-Host "✅ $file exists" -ForegroundColor Green
    } else {
        Write-Host "❌ $file missing" -ForegroundColor Red
        $allGood = $false
    }
}

# Check 4: Benchmark dataset
Write-Host "`n[CHECK 4] Benchmark Dataset" -ForegroundColor Cyan
$benchmarkPath = Join-Path $ROOT "benchmarks\WiFi-CSI-Sensing-Benchmark-main"
if (Test-Path $benchmarkPath) {
    Write-Host "✅ Benchmark directory exists: $benchmarkPath" -ForegroundColor Green
    
    # Check for data files
    $dataFiles = Get-ChildItem -Path $benchmarkPath -Filter "*.h5" -ErrorAction SilentlyContinue
    $dataFiles += Get-ChildItem -Path $benchmarkPath -Filter "*.npz" -ErrorAction SilentlyContinue
    
    if ($dataFiles.Count -gt 0) {
        Write-Host "✅ Data files found: $($dataFiles.Count) files" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No data files (.h5/.npz) found in benchmark directory" -ForegroundColor Yellow
        Write-Host "   Experiments will run in mock mode without real data" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Benchmark directory not found: $benchmarkPath" -ForegroundColor Yellow
    Write-Host "   Will create placeholder and run in mock mode" -ForegroundColor Yellow
}

# Check 5: D2 pre-trained models
Write-Host "`n[CHECK 5] D2 Pre-trained Models" -ForegroundColor Cyan
$d2ModelsPath = Join-Path $ROOT "checkpoints\d2"
if (Test-Path $d2ModelsPath) {
    $modelFiles = Get-ChildItem -Path $d2ModelsPath -Filter "*.pth" -ErrorAction SilentlyContinue
    if ($modelFiles.Count -gt 0) {
        Write-Host "✅ D2 models found: $($modelFiles.Count) model files" -ForegroundColor Green
        foreach ($model in $modelFiles[0..2]) {  # Show first 3
            Write-Host "   - $($model.Name)"
        }
    } else {
        Write-Host "⚠️  No .pth model files in D2 directory" -ForegroundColor Yellow
        Write-Host "   D3 experiments will train from scratch" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  D2 models directory not found" -ForegroundColor Yellow
    Write-Host "   Creating directory - experiments will train from scratch" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $d2ModelsPath | Out-Null
}

# Check 6: Output directories
Write-Host "`n[CHECK 6] Output Directories" -ForegroundColor Cyan
$outputDirs = @("results", "results\d3", "results\d3\loso", "results\d3\loro")

foreach ($dir in $outputDirs) {
    $fullPath = Join-Path $ROOT $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Force -Path $fullPath | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "✅ Directory exists: $dir" -ForegroundColor Green
    }
}

# Summary and recommendations
Write-Host "`n" + "=" * 50
if ($allGood) {
    Write-Host "🎉 All critical prerequisites met!" -ForegroundColor Green
    Write-Host "`nReady to run D3 experiments!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some prerequisites missing" -ForegroundColor Yellow
    Write-Host "`nPlease address the issues above before running experiments." -ForegroundColor Yellow
}

Write-Host "`n📋 Windows Execution Guide:" -ForegroundColor Cyan
Write-Host "Instead of bash, use PowerShell:"
Write-Host "  PowerShell .\scripts\run_d3_loso.ps1" -ForegroundColor White
Write-Host "  PowerShell .\scripts\run_d3_loro.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Or run individual experiments:"
Write-Host "  python -m src.train_cross_domain --model enhanced --protocol loso --seed 0" -ForegroundColor White

Write-Host "`n🔧 If dependencies missing:" -ForegroundColor Cyan
Write-Host "  pip install torch numpy" -ForegroundColor White
Write-Host "  conda install pytorch numpy -c pytorch" -ForegroundColor White

Write-Host "`n💡 Benchmark dataset options:" -ForegroundColor Cyan
Write-Host "  1. Download WiFi-CSI-Sensing-Benchmark and place in benchmarks/" -ForegroundColor White
Write-Host "  2. Use your own CSI data (modify src\data_real.py)" -ForegroundColor White  
Write-Host "  3. Run synthetic-only experiments first" -ForegroundColor White