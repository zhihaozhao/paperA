Exp2: Replace LSTM with Mamba

Objectives
- Evaluate state-space model (Mamba) as a drop-in replacement for LSTM/BiLSTM in our Enhanced pipeline under CDAE/STEA.

Sub-experiments
- M1: Baseline LSTM vs Mamba (capacity-matched)
- M2: Mamba + SE channel attention
- M3: Mamba + temporal lite attention head
- M4: Ablations on sequence length and hidden dims

Acceptance spec
- Params matched within ±10%; report throughput and F1/ECE/NLL.
- CDAE LOSO/LORO gap ≤ 3% with Mamba+SE(+attn).

Run
- See scripts/setup.sh and scripts/train.sh
