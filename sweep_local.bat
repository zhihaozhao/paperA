rem CPU Smoke Test Script (single quick run)
rem Purpose: verify pipeline/logs/JSON/registry on CPU; research results run on GPU

@echo off
setlocal enabledelayedexpansion

rem === Explicit Python interpreter to avoid using base env ===
rem Update this path if your environment changes
set "PY_EXE=D:\workspace_AI\Anaconda3\envs\py310\python.exe"

echo Running CPU smoke: model=cnn, seed=0
"%PY_EXE%" src\train_eval.py ^
  --model cnn ^
  --difficulty hard ^
  --seed 0 ^
  --F 52 ^
  --T 128 ^
  --num_classes 8 ^
  --epochs 1 ^
  --batch 16 ^
  --n_samples 200 ^
  --val_every 1 ^
  --save_ckpt none ^
  --temp_mode none ^
  --num_workers_override 0 ^
  --out_json results_cpu\smoke_cnn.json

echo CPU smoke complete.