Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
Write-Host "[INFO] ROOT =" (Get-Location)

$seeds = 0..7
$models = @("enhanced","lstm")
New-Item -ItemType Directory -Force -Path results\synth | Out-Null
foreach ($m in $models) {
  foreach ($s in $seeds) {
    python -m src.train_eval --model $m --logit_l2 0.05 --seed $s --difficulty mid --out "results\synth\${m}_s${s}.json"
  }
}