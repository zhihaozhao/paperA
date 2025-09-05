#!/usr/bin/env python3
"""
Minimal test to verify PyTorch installation on Xavier
"""

import sys
import torch

print("="*60)
print("PyTorch Installation Check")
print("="*60)

# Basic info
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# CUDA check
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    device = 'cuda'
else:
    print("✗ CUDA not available, using CPU")
    device = 'cpu'

print("\n" + "="*60)
print("Testing basic operations...")
print("="*60)

# Test basic tensor operations
try:
    x = torch.randn(2, 3, 4).to(device)
    print(f"✓ Created random tensor on {device}: shape {x.shape}")
    
    # Test simple neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2)
    ).to(device)
    
    output = model(x)
    print(f"✓ Forward pass successful: output shape {output.shape}")
    
    # Test LSTM without batch_first
    print("\nTesting LSTM (without batch_first)...")
    lstm = torch.nn.LSTM(4, 8, 1).to(device)
    # Input: (seq_len, batch, features)
    lstm_input = x.transpose(0, 1)  # Convert to (3, 2, 4)
    lstm_out, _ = lstm(lstm_input)
    print(f"✓ LSTM forward pass: input {lstm_input.shape} -> output {lstm_out.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print("="*60)