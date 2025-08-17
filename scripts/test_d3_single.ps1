param(
    [Parameter(Mandatory=$false)]
    [int]$EPOCHS = 20,
    
    [Parameter(Mandatory=$false)]
    [string]$MODEL = "enhanced",
    
    [Parameter(Mandatory=$false)]
    [int]$SEED = 0,
    
    [Parameter(Mandatory=$false)]
    [string]$DIFFICULTY = "mid",
    
    [Parameter(Mandatory=$false)]
    [string]$OUTPUT_DIR = "results/loso",
    
    [Parameter(Mandatory=$false)]
    [int]$POSITIVE_CLASS = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Set UTF-8 encoding
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['Out-File:Encoding']   = 'utf8'
$PSDefaultParameterValues['Set-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'

# Change to project root
Set-Location $PSScriptRoot\..
Write-Host "[INFO] Working directory: $(Get-Location)"

# Create output directory
New-Item -ItemType Directory -Path $OUTPUT_DIR -Force | Out-Null

# Construct output filename
$outputFile = "${OUTPUT_DIR}\loso_${MODEL}_seed${SEED}.json"

Write-Host "[INFO] Running LOSO experiment:"
Write-Host "  Model: $MODEL"
Write-Host "  Epochs: $EPOCHS" 
Write-Host "  Seed: $SEED"
Write-Host "  Difficulty: $DIFFICULTY"
Write-Host "  Positive Class: $POSITIVE_CLASS"
Write-Host "  Output: $outputFile"

# Run the cross-domain training script
python -m src.train_cross_domain `
    --source "synth" `
    --target "real" `
    --model $MODEL `
    --epochs $EPOCHS `
    --seed $SEED `
    --difficulty $DIFFICULTY `
    --positive_class $POSITIVE_CLASS `
    --out $outputFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] LOSO experiment completed successfully"
    Write-Host "Results saved to: $outputFile"
} else {
    Write-Host "[ERROR] LOSO experiment failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}