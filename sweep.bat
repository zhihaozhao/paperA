

# Windows-Compatible Sweep Scripts
# This artifact contains two versions: one for CMD (.bat) and one for PowerShell (.ps1).
# Usage:
# 1. Save as sweep.bat (for CMD) or sweep.ps1 (for PowerShell).
# 2. Run in your project root (e.g., D:\workspace_PHD\paperA):
#    - CMD: sweep.bat
#    - PowerShell: .\sweep.ps1 (may need Set-ExecutionPolicy RemoteSigned first)
# 3. It will run 3 models (enhanced, bilstm, cnn) x 10 seeds (0-9), generating results/paperA_${model}_hard_${seed}.json.
# Notes: Assumes python is in PATH; adjust paths if needed. Runs sequentially (add '&' for background if desired).

### Version 1: For Windows CMD (save as sweep.bat)
@echo off
setlocal enabledelayedexpansion

rem === Explicit Python interpreter to avoid using base env ===
rem Update this path if your environment path changes
set "PY_EXE=D:\workspace_AI\Anaconda3\envs\py310\python.exe"
set "POWERSHELL_PY_EXE=$env:PY_EXE"

for /L %%s in (0,1,9) do (
  for %%m in (enhanced bilstm cnn) do (
    set "model=%%m"
    set "seed=%%s"
    echo Running: model=!model!, seed=!seed!
    "%PY_EXE%" src\train_eval.py ^
      --model !model! ^
      --difficulty hard ^
      --seed !seed! ^
      --n_samples 20000 ^
      --epochs 100 ^
      --batch 1024 ^
      --T 128 ^
      --F 52 ^
      --early_metric macro_f1 ^
      --patience 10 ^
      --logit_l2 0.1 ^
      --out_json results\paperA_!model!_hard_!seed!.json ^
      --temp_mode logspace ^
      --temp_min 1.0 ^
      --temp_max 5.0 ^
      --temp_steps 100 ^
      --val_split 0.5 ^
      --class_overlap 0.8 ^
      --gain_drift_std 0.6 ^
      --sc_corr_rho 0.5 ^
      --env_burst_rate 0.2 ^
      --label_noise_prob 0.1 ^
      --num_classes 8 ^
      --ckpt_dir checkpoints\
  )
)

echo Sweep complete!

### Version 2: For Windows PowerShell (save as sweep.ps1)
0..9 | ForEach-Object {
  $seed = $_
  foreach ($model in @("enhanced", "bilstm", "cnn", "conformer_lite")) {
    Write-Host "Running: model=$model, seed=$seed"
    %PY_EXE% src\train_eval.py `
      --model $model `
      --difficulty hard `
      --seed $seed `
      --n_samples 20000 `
      --epochs 100 `
      --batch 768 `
      --T 128 `
      --F 52 `
      --early_metric macro_f1 `
      --patience 10 `
      --logit_l2 0.1 `
      --out_json results\paperA_${model}_hard_${seed}.json `
      --temp_mode logspace `
      --temp_min 1.0 `
      --temp_max 5.0 `
      --temp_steps 100 `
      --val_split 0.5 `
      --class_overlap 0.8 `
      --gain_drift_std 0.6 `
      --sc_corr_rho 0.5 `
      --env_burst_rate 0.2 `
      --label_noise_prob 0.1 `
      --num_classes 8 `
      --ckpt_dir checkpoints\
  }
}

Write-Host "Sweep complete!"

