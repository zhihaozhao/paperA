
import torch
import torch.nn as nn
import time
import tracemalloc
import json
from pathlib import Path

# Import your models
from src.models import TinyNet, BiLSTM, SimpleCNN

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_shape, num_runs=100, warmup=10):
    """Measure inference latency"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times)
    }

def measure_memory_usage(model, input_shape):
    """Measure memory footprint"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = torch.randn(input_shape).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak
        }
    else:
        # CPU memory measurement
        tracemalloc.start()
        with torch.no_grad():
            _ = model(x)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_mb': current / 1024**2,
            'peak_mb': peak / 1024**2
        }

def run_efficiency_benchmark():
    """Main benchmark function"""
    
    # Configuration
    T, F = 128, 52  # Time steps, Features
    num_classes = 6
    batch_size = 1
    
    # Models to test
    models = {
        'PASE-Net': TinyNet(input_features=F, num_classes=num_classes),
        'CNN': SimpleCNN(T=T, F=F, num_classes=num_classes),
        'BiLSTM': BiLSTM(input_dim=F, num_classes=num_classes)
    }
    
    # If you have the actual PASE-Net model
    try:
        from src.models import EnhancedNet
        models['PASE-Net-Full'] = EnhancedNet(T=T, F=F, num_classes=num_classes, base_channels=64)
    except:
        pass
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Count parameters
        params = count_parameters(model) / 1e6  # Convert to millions
        
        # Measure inference time
        input_shape = (batch_size, T, F)
        timing = measure_inference_time(model, input_shape)
        
        # Measure memory
        memory = measure_memory_usage(model, input_shape)
        
        # Calculate FLOPs (approximate)
        # This is a rough estimate - you may need to use torch.profiler for accurate count
        flops_est = params * 2 * np.prod(input_shape)  # Very rough estimate
        
        results[name] = {
            'parameters_M': round(params, 3),
            'inference_ms': round(timing['mean_ms'], 2),
            'inference_std': round(timing['std_ms'], 2),
            'memory_mb': round(memory.get('peak_mb', memory.get('peak_mb', 0)), 1),
            'flops_G_estimate': round(flops_est / 1e9, 2),
            'edge_ready': timing['mean_ms'] < 30 and memory.get('peak_mb', 100) < 100
        }
        
        print(f"  Parameters: {results[name]['parameters_M']}M")
        print(f"  Inference: {results[name]['inference_ms']}±{results[name]['inference_std']}ms")
        print(f"  Memory: {results[name]['memory_mb']}MB")
        print(f"  Edge Ready: {results[name]['edge_ready']}")
    
    # Save results
    output_file = 'xavier_efficiency_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Model Efficiency on NVIDIA AGX Xavier}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Inference (ms)} & \\textbf{Memory (MB)} & \\textbf{Edge Ready} \\\\")
    print("\\midrule")
    
    for name, res in results.items():
        edge = "\\checkmark" if res['edge_ready'] else "$\\times$"
        print(f"{name} & {res['parameters_M']} & {res['inference_ms']}±{res['inference_std']} & {res['memory_mb']} & {edge} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    run_efficiency_benchmark()
