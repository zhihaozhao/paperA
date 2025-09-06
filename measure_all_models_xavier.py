#!/usr/bin/env python3
"""
D1 Experiment All Models Xavier Efficiency Measurement
Measure parameters and GPU latency for all D1 experiment models:
- Enhanced (BiLSTM-based)
- CNN 
- BiLSTM
- Conformer-lite

This script imports models from src/models.py, performs inference with warmup,
and outputs a JSON report with summary comparing results to Paper 1 Table 1.
"""

import json
import time
import torch
import torch.nn as nn
import psutil
import os
import platform
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Add src directory to path to import real models
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

try:
    from models import build_model
    print(f"✅ Successfully imported models from {src_path}")
except ImportError as e:
    print(f"❌ Failed to import models: {e}")
    print(f"Trying alternative path...")
    try:
        sys.path.insert(0, './src')
        from models import build_model
        print("✅ Successfully imported models with relative path")
    except ImportError:
        print("❌ Could not find models.py. Please run from project root directory.")
        exit(1)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_memory_usage():
    """Measure current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_inference_time(model, input_tensor, warmup_runs=30, measure_runs=100):
    """Measure inference time with proper warmup"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min_ms': min(times),
        'max_ms': max(times)
    }

def estimate_flops(model, input_shape):
    """Rough FLOPs estimation for the model"""
    params = count_parameters(model)
    T, F = input_shape[1], input_shape[2]
    
    # For smaller models, use different estimation
    estimated_flops = params * 2.0 * T * F * 0.1  # Reduced factor for small models
    return estimated_flops / 1e9  # Convert to GFLOPs

def evaluate_model_efficiency(model_name, model, input_shape=(1, 128, 52), device='cpu'):
    """Comprehensive efficiency evaluation"""
    print(f"\n{'='*15} {model_name} D1 Configuration Analysis {'='*15}")
    
    # Move model to device
    model = model.to(device)
    
    # Create input tensor
    input_tensor = torch.randn(input_shape).to(device)
    
    # Memory before inference
    memory_before = measure_memory_usage()
    
    # Parameter count
    param_count = count_parameters(model)
    param_k = param_count / 1000
    
    # Inference time measurement
    timing_results = measure_inference_time(model, input_tensor)
    
    # Memory after inference
    _ = model(input_tensor)
    memory_after = measure_memory_usage()
    memory_peak_mb = max(memory_after - memory_before + 20, 20)  # Minimum 20MB base
    
    # FLOPs estimation
    flops_g = estimate_flops(model, input_shape)
    
    # Edge readiness assessment (more lenient for smaller models)
    edge_ready = (timing_results['mean_ms'] < 50 and memory_peak_mb < 200)
    
    results = {
        'model': model_name,
        'parameters': param_count,
        'parameters_K': param_k,
        'parameters_M': param_count / 1e6,
        'inference_mean_ms': timing_results['mean_ms'],
        'inference_std_ms': timing_results['std_ms'],
        'inference_min_ms': timing_results['min_ms'],
        'inference_max_ms': timing_results['max_ms'],
        'memory_peak_mb': memory_peak_mb,
        'flops_g': flops_g,
        'edge_ready': edge_ready,
        'device': str(device),
        'input_shape': input_shape,
        'config_type': 'D1_true_parameters'
    }
    
    # Print results
    print(f"Parameters: {param_count:,} ({param_k:.1f}K)")
    print(f"Inference Time: {timing_results['mean_ms']:.2f}±{timing_results['std_ms']:.2f}ms")
    print(f"Memory Usage: {memory_peak_mb:.1f}MB")
    print(f"FLOPs: {flops_g:.2f}G")
    print(f"Edge Ready: {'✓' if edge_ready else '✗'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='D1 All Models Xavier Efficiency Measurement')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device to run inference on')
    parser.add_argument('--output', default='xavier_d1_all_models_{timestamp}.json', help='Output JSON file path')
    parser.add_argument('--T', type=int, default=128, help='Temporal dimension')
    parser.add_argument('--F', type=int, default=52, help='Frequency dimension')
    parser.add_argument('--classes', type=int, default=8, help='Number of classes')
    
    args = parser.parse_args()
    
    print(f"🔧 D1 All Models Xavier Efficiency Measurement")
    print(f"📊 Using real models from src/models.py for consistency")
    print(f"🖥️  Device: {args.device}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    input_shape = (1, args.T, args.F)
    
    # Initialize all D1 experiment models
    # Based on actual build_model function and D1 experiment requirements
    model_configs = {
        'enhanced': {'name': 'enhanced', 'description': 'Enhanced CNN + SE + Attention'},
        'cnn': {'name': 'cnn', 'description': 'Simple CNN'},
        'bilstm': {'name': 'bilstm', 'description': 'BiLSTM'},
        'conformer_lite': {'name': 'conformer_lite', 'description': 'Conformer-lite'}
    }
    
    models = {}
    
    print(f"\n📋 Building All D1 Models from src/models.py:")
    for model_key, config in model_configs.items():
        try:
            print(f"  Building {config['name']} ({config['description']})...")
            
            # Use correct signature: build_model(name, input_dim, num_classes, logit_l2=0.05)
            # input_dim corresponds to F (frequency dimension)
            model = build_model(name=config['name'], input_dim=args.F, num_classes=args.classes, logit_l2=0.05)
            
            models[model_key] = model
            param_count = count_parameters(model)
            print(f"  ✅ {config['name']}: {param_count:,} ({param_count/1000:.1f}K) parameters")
        except Exception as e:
            print(f"  ❌ Failed to build {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Run efficiency measurements
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'device': str(device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        },
        'experiment_config': {
            'T': args.T,
            'F': args.F,
            'num_classes': args.classes,
            'input_shape': input_shape,
            'description': 'D1 all models measurement using real models from src/models.py',
            'models_tested': list(model_configs.keys())
        },
        'results': {}
    }
    
    for model_key, model in models.items():
        try:
            model_results = evaluate_model_efficiency(model_key, model, input_shape, device)
            results['results'][model_key] = model_results
        except Exception as e:
            print(f"❌ Error measuring {model_key}: {e}")
            results['results'][model_key] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output.format(timestamp=timestamp)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary table
    print(f"\n📊 D1 Models Parameter Summary:")
    print(f"{'Model':<15} {'Parameters':<12} {'Inference (ms)':<15} {'Memory (MB)':<12} {'Edge Ready'}")
    print("-" * 70)
    
    for model_key, config in model_configs.items():
        if model_key in results['results'] and 'error' not in results['results'][model_key]:
            r = results['results'][model_key]
            edge_status = '✓' if r['edge_ready'] else '✗'
            print(f"{config['name']:<15} {r['parameters']:<12,} {r['inference_mean_ms']:<15.2f} {r['memory_peak_mb']:<12.1f} {edge_status}")
    
    # Paper 1 Table 1 comparison
    print(f"\n📋 Paper 1 Table 1 Comparison:")
    print(f"{'Model':<15} {'Measured Params':<15} {'Table 1 Params':<15} {'Measured GPU (ms)':<18} {'Table 1 GPU (ms)':<18} {'Status'}")
    print("-" * 100)
    
    # Expected values from Paper 1 Table 1
    paper1_expected = {
        'enhanced': {'params': 640713, 'gpu_ms': 5.29},
        'cnn': {'params': 644216, 'gpu_ms': 0.90},
        'bilstm': {'params': 583688, 'gpu_ms': 8.97},
        'conformer_lite': {'params': 1498672, 'gpu_ms': 5.16}
    }
    
    for model_key, config in model_configs.items():
        if model_key in results['results'] and 'error' not in results['results'][model_key]:
            r = results['results'][model_key]
            expected = paper1_expected.get(model_key, {'params': 0, 'gpu_ms': 0})
            
            params_match = abs(r['parameters'] - expected['params']) < 1000
            gpu_match = abs(r['inference_mean_ms'] - expected['gpu_ms']) < 0.5
            
            status = "✅ Match" if (params_match and gpu_match) else "⚠️ Different"
            
            print(f"{config['name']:<15} {r['parameters']:<15,} {expected['params']:<15,} {r['inference_mean_ms']:<18.2f} {expected['gpu_ms']:<18.2f} {status}")
    
    print(f"\n✅ D1 All Models Xavier Efficiency Measurement Complete!")
    print(f"🎯 All models tested using real architecture definitions from src/models.py")

if __name__ == "__main__":
    main()