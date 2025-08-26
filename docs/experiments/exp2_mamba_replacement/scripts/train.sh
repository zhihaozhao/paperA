#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python src/train.py --config configs/mamba_baseline.yaml
