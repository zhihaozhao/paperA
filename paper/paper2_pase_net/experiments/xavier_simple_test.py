#!/usr/bin/env python3
"""
Simplified efficiency test for NVIDIA AGX Xavier
Can run without full codebase - just needs PyTorch
"""

import torch
import torch.nn as nn
import time
import json
import numpy as np
from datetime import datetime

class SimplePASENet(nn.Module):
    """Simplified PASE-Net for testing"""
    def __init__(self, input_dim=52, num_classes=6):
        super().__init__()
        # Simplified architecture matching paper description
        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        
        # SE-like attention (simplified)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 256, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention (simplified)
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)
        
        # Convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # SE attention
        se_weight = self.se(x)
        x = x * se_weight
        
        # Prepare for temporal attention
        x = x.transpose(1, 2)  # (batch, time, features)
        x, _ = self.attention(x, x, x)
        
        # Global pooling
        x = x.mean(dim=1)  # (batch, features)
        
        # Classification
        return self.classifier(x)

class SimpleCNN(nn.Module):
    """Simple CNN baseline"""
    def __init__(self, input_dim=52, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class SimpleBiLSTM(nn.Module):
    """Simple BiLSTM baseline"""
    def __init__(self, input_dim=52, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        return self.fc(out)

def count_parameters(model):
    """Count model parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_inference_time(model, input_shape, device='cuda', num_runs=100):
    """Measure inference latency"""
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def measure_memory(model, input_shape, device='cuda'):
    """Measure GPU memory usage"""
    model.eval()
    model = model.to(device)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(input_shape).to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {'allocated_mb': allocated, 'peak_mb': peak}
    else:
        return {'allocated_mb': 0, 'peak_mb': 0}

def main():
    print("="*60)
    print("NVIDIA AGX XAVIER - MODEL EFFICIENCY TEST")
    print("="*60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("✗ CUDA Not Available - Using CPU")
        device = 'cpu'
    
    # Configuration
    batch_size = 1
    time_steps = 128
    features = 52
    input_shape = (batch_size, time_steps, features)
    
    # Models to test
    models = {
        'PASE-Net': SimplePASENet(input_dim=features),
        'CNN': SimpleCNN(input_dim=features),
        'BiLSTM': SimpleBiLSTM(input_dim=features)
    }
    
    results = {}
    
    print(f"\nInput shape: {input_shape}")
    print(f"Device: {device}")
    print("-"*60)
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Count parameters
        params = count_parameters(model)
        print(f"  Parameters: {params:.3f}M")
        
        # Measure inference time
        timing = measure_inference_time(model, input_shape, device)
        print(f"  Inference: {timing['mean']:.2f}±{timing['std']:.2f}ms")
        
        # Measure memory
        memory = measure_memory(model, input_shape, device)
        print(f"  Memory: {memory['peak_mb']:.1f}MB")
        
        # Check if edge-ready
        edge_ready = timing['mean'] < 30 and memory['peak_mb'] < 100
        print(f"  Edge Ready: {'✓' if edge_ready else '✗'}")
        
        results[name] = {
            'parameters_M': round(params, 3),
            'inference_mean_ms': round(timing['mean'], 2),
            'inference_std_ms': round(timing['std'], 2),
            'memory_peak_mb': round(memory['peak_mb'], 1),
            'edge_ready': edge_ready
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"xavier_efficiency_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'timestamp': timestamp,
            'input_shape': input_shape,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    
    # Generate LaTeX table
    print(f"\n{'='*60}")
    print("LATEX TABLE FOR PAPER")
    print("="*60)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Model Efficiency on NVIDIA AGX Xavier}")
    print("\\label{tab:efficiency_real}")
    print("\\small")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Inference (ms)} & \\textbf{Memory (MB)} & \\textbf{Edge Ready} \\\\")
    print("\\midrule")
    
    for name, res in results.items():
        edge = "\\checkmark" if res['edge_ready'] else "$\\times$"
        print(f"{name} & {res['parameters_M']} & {res['inference_mean_ms']}$\\pm${res['inference_std_ms']} & {res['memory_peak_mb']} & {edge} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\textit{Note: Measured on NVIDIA AGX Xavier (32GB) with batch size 1.}")
    print("\\end{table}")

if __name__ == "__main__":
    main()