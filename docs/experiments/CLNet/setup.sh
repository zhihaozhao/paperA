#!/usr/bin/env bash
set -euo pipefail
# Create isolated env (conda or venv) and install dependencies
# This is a placeholder; fill with official repo requirements once linked.
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# Add project-specific deps after cloning official code
