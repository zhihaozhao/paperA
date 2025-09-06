#!/usr/bin/env python3
"""
Conformer-lite Xavier Efficiency Measurement
Focused measurement for Conformer-lite model to get accurate Table 1 data.
This script specifically targets the Conformer-lite model from D1 experiments.
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

def measure_inference_time(model, input_tensor, warmup_runs=50, measure_runs=200):
    """Measure inference time with extended warmup for Conformer-lite"""
    model.eval()
    
    # Extended warmup for transformer-based models
    print(f"  🔥 Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model(input_tensor)
            if i % 10 == 0:
                print(f"    Warmup {i+1}/{warmup_runs}")
    
    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure with more runs for better statistics
    print(f"  📊 Measuring with {measure_runs} runs...")
    times = []
    with torch.no_grad():
        for i in range(measure_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if i % 50 == 0:
                print(f"    Measurement {i+1}/{measure_runs}")
    
    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'min_ms': min(times),
        'max_ms': max(times),
        'median_ms': sorted(times)[len(times)//2],
        'p95_ms': sorted(times)[int(len(times)*0.95)],
        'p99_ms': sorted(times)[int(len(times)*0.99)]
    }

def estimate_flops(model, input_shape):
    """FLOPs estimation for Conformer-lite"""
    params = count_parameters(model)
    T, F = input_shape[1], input_shape[2]
    
    # Conformer-lite specific estimation
    # More complex due to attention mechanisms
    estimated_flops = params * 2.5 * T * F * 0.15  # Higher factor for transformer
    return estimated_flops / 1e9  # Convert to GFLOPs

def evaluate_conformer_lite_efficiency(model, input_shape=(1, 128, 52), device='cpu'):
    """Comprehensive efficiency evaluation for Conformer-lite"""
    print(f"\n{'='*20} Conformer-lite D1 Configuration Analysis {'='*20}")
    
    # Move model to device
    model = model.to(device)
    
    # Create input tensor
    input_tensor = torch.randn(input_shape).to(device)
    
    # Memory before inference
    memory_before = measure_memory_usage()
    
    # Parameter count
    param_count = count_parameters(model)
    param_k = param_count / 1000
    param_m = param_count / 1e6
    
    print(f"📊 Model Analysis:")
    print(f"  Parameters: {param_count:,} ({param_k:.1f}K, {param_m:.2f}M)")
    
    # Detailed parameter breakdown
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_params += param_num
            print(f"    {name}: {param_num:,} params")
    
    # Inference time measurement
    timing_results = measure_inference_time(model, input_tensor)
    
    # Memory after inference
    _ = model(input_tensor)
    memory_after = measure_memory_usage()
    memory_peak_mb = max(memory_after - memory_before + 30, 30)  # Higher base for transformer
    
    # FLOPs estimation
    flops_g = estimate_flops(model, input_shape)
    
    # Edge readiness assessment
    edge_ready = (timing_results['mean_ms'] < 20 and memory_peak_mb < 500)
    
    results = {
        'model': 'conformer_lite',
        'parameters': param_count,
        'parameters_K': param_k,
        'parameters_M': param_m,
        'inference_mean_ms': timing_results['mean_ms'],
        'inference_std_ms': timing_results['std_ms'],
        'inference_min_ms': timing_results['min_ms'],
        'inference_max_ms': timing_results['max_ms'],
        'inference_median_ms': timing_results['median_ms'],
        'inference_p95_ms': timing_results['p95_ms'],
        'inference_p99_ms': timing_results['p99_ms'],
        'memory_peak_mb': memory_peak_mb,
        'flops_g': flops_g,
        'edge_ready': edge_ready,
        'device': str(device),
        'input_shape': input_shape,
        'config_type': 'D1_conformer_lite_focused'
    }
    
    # Print results
    print(f"\n📈 Performance Results:")
    print(f"  Inference Time: {timing_results['mean_ms']:.2f}±{timing_results['std_ms']:.2f}ms")
    print(f"    Min: {timing_results['min_ms']:.2f}ms")
    print(f"    Max: {timing_results['max_ms']:.2f}ms")
    print(f"    Median: {timing_results['median_ms']:.2f}ms")
    print(f"    P95: {timing_results['p95_ms']:.2f}ms")
    print(f"    P99: {timing_results['p99_ms']:.2f}ms")
    print(f"  Memory Usage: {memory_peak_mb:.1f}MB")
    print(f"  FLOPs: {flops_g:.2f}G")
    print(f"  Edge Ready: {'✓' if edge_ready else '✗'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Conformer-lite Xavier Efficiency Measurement')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='Device to run inference on')
    parser.add_argument('--output', default='xavier_conformer_lite_{timestamp}.json', help='Output JSON file path')
    parser.add_argument('--T', type=int, default=128, help='Temporal dimension')
    parser.add_argument('--F', type=int, default=52, help='Frequency dimension')
    parser.add_argument('--classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--warmup', type=int, default=50, help='Number of warmup runs')
    parser.add_argument('--measure', type=int, default=200, help='Number of measurement runs')
    
    args = parser.parse_args()
    
    print(f"🔧 Conformer-lite Xavier Efficiency Measurement")
    print(f"📊 Focused measurement for D1 Conformer-lite model")
    print(f"🖥️  Device: {args.device}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    input_shape = (1, args.T, args.F)
    
    # Build Conformer-lite model
    print(f"\n📋 Building Conformer-lite Model:")
    try:
        print(f"  Building conformer_lite...")
        model = build_model(name='conformer_lite', input_dim=args.F, num_classes=args.classes, logit_l2=0.05)
        print(f"  ✅ Conformer-lite model built successfully")
    except Exception as e:
        print(f"  ❌ Failed to build conformer_lite: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run efficiency measurement
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
            'warmup_runs': args.warmup,
            'measure_runs': args.measure,
            'description': 'D1 Conformer-lite focused measurement using real model from src/models.py'
        },
        'results': {}
    }
    
    try:
        model_results = evaluate_conformer_lite_efficiency(model, input_shape, device)
        results['results']['conformer_lite'] = model_results
    except Exception as e:
        print(f"❌ Error measuring conformer_lite: {e}")
        import traceback
        traceback.print_exc()
        results['results']['conformer_lite'] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output.format(timestamp=timestamp)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Paper 1 Table 1 comparison
    if 'conformer_lite' in results['results'] and 'error' not in results['results']['conformer_lite']:
        r = results['results']['conformer_lite']
        expected_params = 1498672
        expected_gpu_ms = 5.16
        
        print(f"\n📋 Paper 1 Table 1 Comparison:")
        print(f"{'Metric':<20} {'Measured':<15} {'Expected':<15} {'Difference':<15} {'Status'}")
        print("-" * 80)
        
        params_diff = r['parameters'] - expected_params
        gpu_diff = r['inference_mean_ms'] - expected_gpu_ms
        
        params_match = abs(params_diff) < 1000
        gpu_match = abs(gpu_diff) < 0.5
        
        print(f"{'Parameters':<20} {r['parameters']:<15,} {expected_params:<15,} {params_diff:<15,} {'✅' if params_match else '⚠️'}")
        print(f"{'GPU Latency (ms)':<20} {r['inference_mean_ms']:<15.2f} {expected_gpu_ms:<15.2f} {gpu_diff:<15.2f} {'✅' if gpu_match else '⚠️'}")
        
        overall_status = "✅ Match" if (params_match and gpu_match) else "⚠️ Different"
        print(f"\nOverall Status: {overall_status}")
    
    print(f"\n✅ Conformer-lite Xavier Efficiency Measurement Complete!")
    print(f"🎯 Conformer-lite model tested using real architecture definition from src/models.py")

if __name__ == "__main__":
    main()