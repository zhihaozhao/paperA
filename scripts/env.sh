#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
# 请把下面路径换成你的实际 conda python.exe 路径
# 例如 D:\miniconda3\envs\py310\python.exe
#PY_WIN= "D:\workspace_AI\Anaconda3\envs\py310\python.exe"
PY_WIN="D:\workspace_AI\Anaconda3\envs\py310\python.exe"
# 转换为 Git Bash 可识别的 /d/miniconda3/envs/py310/python.exe
drive_letter="$(echo "$PY_WIN" | sed -n 's/^\([A-Za-z]\):.*/\1/p')"
rest_path="$(echo "$PY_WIN" | sed 's/^[A-Za-z]:\\//; s#\\#/#g')"
PY="/${drive_letter,,}/$rest_path"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] PY=$PY"
"$PY" - <<'PYCODE'
import sys, os
print("[INFO] python exec =", sys.executable)
print("[INFO] cwd =", os.getcwd())
PYCODE