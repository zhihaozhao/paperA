

rem Local CPU Small Sweep Script
rem Runs 4 models x 10 seeds with small params: n_samples=2000, epochs=20, batch=64.

rem === Version 1: For Windows CMD (save as sweep_local.bat) ===
@echo off
setlocal enabledelayedexpansion

rem === Explicit Python interpreter to avoid using base env ===
rem Update this path if your environment path changes
set "PY_EXE=D:\workspace_AI\Anaconda3\envs\py310\python.exe"

for /L %%s in (0,1,9) do (
  for %%m in (enhanced bilstm cnn conformer_lite) do (
    set "model=%%m"
    set "seed=%%s"
    echo Running: model=!model!, seed=!seed!
    "%PY_EXE%" src\train_eval.py ^
      --model !model! ^
      --difficulty hard ^
      --seed !seed! ^
      --n_samples 2000 ^
      --epochs 20 ^
      --batch 64 ^
      --T 128 ^
      --F 52 ^
      --early_metric macro_f1 ^
      --patience 10 ^
      --logit_l2 0.1 ^
      --out_json results_cpu\paperA_!model!_hard_!seed!.json ^
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
      --ckpt_dir checkpoints\ ^
	  --amp ^
	  --save_ckpt final ^
      --val_every 3 ^
      --num_workers_override 2
  )
)

echo Local small sweep complete!

rem === Version 2: For Windows PowerShell (save as sweep_local.ps1) ===
0..9 | ForEach-Object {
  $seed = $_
  foreach ($model in @("enhanced", "bilstm", "cnn", "conformer_lite")) {
    Write-Host "Running: model=$model, seed=$seed"
    python src\train_eval.py `
      --model $model `
      --difficulty hard `
      --seed $seed `
      --n_samples 2000 `
      --epochs 20 `
      --batch 64 `
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

Write-Host "Local small sweep complete!"