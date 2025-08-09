Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
New-Item -ItemType Directory -Force -Path results\synth_lambda | Out-Null
$lambdas = @(0,0.02,0.05,0.08,0.12,0.18)
foreach ($L in $lambdas) {
  python -m src.train_eval --model enhanced --logit_l2 $L --seed 0 --difficulty mid --out "results\synth_lambda\enhanced_l${L}.json"
}