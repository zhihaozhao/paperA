Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PSDefaultParameterValues['Out-File:Encoding']   = 'utf8'
$PSDefaultParameterValues['Set-Content:Encoding'] = 'utf8'
$PSDefaultParameterValues['Add-Content:Encoding'] = 'utf8'

# # 1) 初始化 conda 并激活 py310（按需调整候选路径）
# $condaHookCandidates = @(
#   "$env:USERPROFILE\Anaconda3\shell\condabin\conda-hook.ps1",
#   "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1",
#   "C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1",
#   "C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1"
# )
# $condaHook = $condaHookCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
# if (-not $condaHook) {
#   throw "未找到 conda-hook.ps1，请把实际路径填到 `$condaHook` 变量"
# }
# . $condaHook
# conda activate py310

# 2) 切到项目根并打印 Python 信息
Set-Location $PSScriptRoot\..
Write-Host "[INFO] ROOT = $(Get-Location)"

# 用单引号包裹 -c 字符串，内部用双引号，避免 PowerShell 误解析
python -c 'import sys,os; print("python =", sys.executable); print("cwd =", os.getcwd())'

# 3) 任务
python -m src.train_eval --model enhanced --logit_l2 0.05 --seed 0 --difficulty mid --out results\synth\enhanced_s0.json
python -m src.train_eval --model lstm     --seed 0 --difficulty mid --out results\synth\lstm_s0.json
python -m src.plotting
python -m src.tables

# 4) 编译论文
Push-Location paper
if (Get-Command latexmk -ErrorAction SilentlyContinue) {
  latexmk -pdf -silent main.tex
  if (!$?) { pdflatex main.tex }
} else {
  pdflatex main.tex
}
Pop-Location

Write-Host "Done. See plots/, tables/, paper\main.pdf"