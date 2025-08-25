# PINN Experiments (LSTM Multi-Scale and Mamba)

This folder adds two physics-informed baselines:
- `pinn_lstm_ms`: LSTM with multi-scale temporal windows and smoothness regularizer
- `pinn_mamba`: Mamba-like temporal blocks with smoothness regularizer

## Usage

```bash
# LSTM multi-scale
bash scripts/run_pinn_lstm_multiscale.sh

# Mamba-like
bash scripts/run_pinn_mamba.sh
```

## Train args (passed to train_eval.py)
- `--pinn_lambda_smooth`: weight for temporal smoothness penalty (default 0.01)
- `--pinn_lambda_energy`: weight for energy penalty (default 0.0)
- `--pinn_ms_windows`: comma-separated window sizes, e.g., "32,64,128"
- `--pinn_mamba_dmodel`: hidden dim for Mamba-like blocks (default 192)
- `--pinn_mamba_layers`: number of Mamba-like blocks (default 3)

Outputs are saved to `results/*.json` and logs to `results/logs/`.