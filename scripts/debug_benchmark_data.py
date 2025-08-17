#!/usr/bin/env python3
"""
Debug script for WiFi-CSI-Sensing-Benchmark data loading
Tests .mat file formats and data structure to fix D3 experiments
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_real import BenchmarkCSIDataset

def inspect_mat_file(file_path):
    """Inspect a single .mat file to understand its structure"""
    try:
        from scipy.io import loadmat
        print(f"\n=== Inspecting {file_path} ===")
        
        mat_data = loadmat(file_path)
        
        # Show all non-private keys
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys: {keys}")
        
        for key in keys:
            data = mat_data[key]
            print(f"  {key}: shape={getattr(data, 'shape', 'N/A')}, dtype={getattr(data, 'dtype', type(data))}")
            
            # Show sample values for small arrays
            if hasattr(data, 'shape') and np.prod(data.shape) <= 20:
                print(f"    Sample values: {data.flatten()[:10]}")
        
        return mat_data
        
    except ImportError:
        print(f"[ERROR] scipy not available, cannot inspect .mat files")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def test_benchmark_loading():
    """Test benchmark data loading with debug output"""
    print("=== Testing WiFi-CSI-Sensing-Benchmark Data Loading ===")
    
    benchmark_path = "benchmarks/WiFi-CSI-Sensing-Benchmark-main"
    print(f"Benchmark path: {benchmark_path}")
    
    # Check if path exists
    if not Path(benchmark_path).exists():
        print(f"[ERROR] Benchmark path does not exist: {benchmark_path}")
        return False
        
    # Check Data subdirectory
    data_path = Path(benchmark_path) / "Data"
    if data_path.exists():
        print(f"[SUCCESS] Data subdirectory found: {data_path}")
        
        # List dataset subdirectories
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        print(f"Dataset subdirectories: {[d.name for d in subdirs]}")
        
        # Find and inspect sample .mat files
        for subdir in subdirs[:2]:  # Check first 2 subdirectories
            mat_files = list(subdir.glob("**/*.mat"))[:3]  # First 3 .mat files
            print(f"\n{subdir.name}: {len(mat_files)} .mat files found")
            
            for mat_file in mat_files:
                mat_data = inspect_mat_file(mat_file)
                if mat_data:
                    break  # If we successfully loaded one file, that's enough for now
    else:
        print(f"[WARNING] Data subdirectory not found, checking root: {benchmark_path}")
    
    # Test actual benchmark loading
    print(f"\n=== Testing BenchmarkCSIDataset class ===")
    try:
        benchmark = BenchmarkCSIDataset(benchmark_path)
        X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
        
        print(f"[SUCCESS] Benchmark data loaded!")
        print(f"  X.shape: {X.shape}")
        print(f"  y.shape: {y.shape}")
        print(f"  Unique labels: {np.unique(y)}")
        print(f"  Unique subjects: {len(np.unique(subjects))}")
        print(f"  Unique rooms: {len(np.unique(rooms))}")
        print(f"  Metadata: {metadata}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Benchmark loading failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("WiFi-CSI-Sensing-Benchmark Data Debug Tool")
    print("=" * 60)
    
    # Set up environment
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    success = test_benchmark_loading()
    
    if success:
        print(f"\n✅ SUCCESS: Benchmark data loading works!")
        print(f"Now D3 LOSO/LORO should use real data instead of synthetic F1=1.0")
    else:
        print(f"\n❌ FAILED: Benchmark data loading has issues")
        print(f"D3 experiments will continue using synthetic fallback data")
        
    print("\nTo run D3 experiments:")
    print("  scripts\\test_d3_quick.bat")
    print("  scripts\\run_d3_d4_windows.bat")