

rem Windows-Compatible Sweep Script (CMD only)
rem Usage: run in project root (e.g., D:\workspace_PHD\paperA): sweep.bat
rem Models: enhanced, bilstm, cnn, conformer_lite; seeds 0..9

@echo off
setlocal enabledelayedexpansion

rem === Python interpreter discovery ===
rem Allow user override: set PY_EXE=... & sweep.bat
if not defined PY_EXE (
  for %%P in (
    "D:\\workspace_AI\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\ProgramData\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\Users\\%USERNAME%\\Anaconda3\\envs\\py310\\python.exe"
    "C:\\Users\\%USERNAME%\\miniconda3\\envs\\py310\\python.exe"
  ) do (
    if exist %%~P set "PY_EXE=%%~P" & goto :py_found
  )
  set "PY_EXE=python"
)
:py_found
echo [INFO] Using Python: %PY_EXE%

for /L %%s in (0,1,9) do (
  for %%m in (enhanced bilstm cnn conformer_lite) do (
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
      --out_json results_gpu\paperA_!model!_hard_!seed!.json ^
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

