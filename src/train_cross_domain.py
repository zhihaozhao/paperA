import argparse
import json
import os
import numpy as np
import torch
from torch.optim import Adam
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.models import build_model
from src.data_real import BenchmarkCSIDataset, get_real_loaders_loso, get_real_loaders_loro
from src.metrics import compute_metrics
from src.calibration import ece, brier
from src.train_eval import train_one_epoch, eval_model, calibrate_temperature, softmax_logits
from src.utils.logger import init_run
from src.utils.exp_recorder import ExpRecorder

def run_loso_experiment(model_name: str, benchmark_path: str, seed: int, 
                       epochs: int = 100, output_dir: str = "results/d3/loso") -> Dict:
    """
    Run Leave-One-Subject-Out cross-validation experiment
    """
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load benchmark dataset
    benchmark = BenchmarkCSIDataset(benchmark_path)
    X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
    
    # Create LOSO splits
    loso_splits = benchmark.create_loso_splits(subjects)
    
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(loso_splits):
        test_subject = np.unique(subjects[test_idx])[0]
        
        print(f"[LOSO] Fold {fold_idx+1}/{len(loso_splits)}: Test Subject {test_subject}")
        
        # Create data loaders
        train_loader, test_loader = get_real_loaders_loso(X, y, subjects, test_subject)
        
        # Build model
        model = build_model(model_name)
        model.to(device)
        
        # Training setup
        optimizer = Adam(model.parameters(), lr=1e-3)
        recorder = ExpRecorder()
        
        # Training loop
        best_metrics = None
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                metrics = eval_model(model, test_loader, device, num_classes=4)
                recorder.log_epoch(epoch, {
                    "train_loss": float(train_loss),
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "falling_f1": float(metrics.get("falling_f1", 0.0)),
                })
                
                if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                    best_metrics = metrics
        
        # Calibration
        cal_results = calibrate_temperature(model, test_loader, device)
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'test_subject': int(test_subject),
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'metrics': best_metrics,
            'calibration': cal_results,
            'seed': seed,
            'model': model_name
        }
        results.append(fold_result)
    
    # Aggregate statistics across folds
    aggregate_stats = _aggregate_loso_results(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"loso_{model_name}_seed{seed}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'experiment': 'LOSO',
            'model': model_name,
            'seed': seed,
            'benchmark_metadata': metadata,
            'fold_results': results,
            'aggregate_stats': aggregate_stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    return aggregate_stats

def run_loro_experiment(model_name: str, benchmark_path: str, seed: int,
                       epochs: int = 100, output_dir: str = "results/d3/loro") -> Dict:
    """
    Run Leave-One-Room-Out cross-validation experiment
    """
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load benchmark dataset
    benchmark = BenchmarkCSIDataset(benchmark_path)
    X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
    
    # Create LORO splits
    loro_splits = benchmark.create_loro_splits(rooms)
    
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(loro_splits):
        test_room = np.unique(rooms[test_idx])[0]
        
        print(f"[LORO] Fold {fold_idx+1}/{len(loro_splits)}: Test Room {test_room}")
        
        # Create data loaders
        train_loader, test_loader = get_real_loaders_loro(X, y, rooms, test_room)
        
        # Build model
        model = build_model(model_name)
        model.to(device)
        
        # Training setup
        optimizer = Adam(model.parameters(), lr=1e-3)
        recorder = ExpRecorder()
        
        # Training loop
        best_metrics = None
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                metrics = eval_model(model, test_loader, device, num_classes=4)
                recorder.log_epoch(epoch, {
                    "train_loss": float(train_loss),
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "falling_f1": float(metrics.get("falling_f1", 0.0)),
                })
                
                if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                    best_metrics = metrics
        
        # Calibration
        cal_results = calibrate_temperature(model, test_loader, device)
        
        # Store fold results
        fold_result = {
            'fold': fold_idx,
            'test_room': int(test_room),
            'train_samples': len(train_idx),
            'test_samples': len(test_idx),
            'metrics': best_metrics,
            'calibration': cal_results,
            'seed': seed,
            'model': model_name
        }
        results.append(fold_result)
    
    # Aggregate statistics across folds
    aggregate_stats = _aggregate_loro_results(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / f"loro_{model_name}_seed{seed}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'experiment': 'LORO',
            'model': model_name,
            'seed': seed,
            'benchmark_metadata': metadata,
            'fold_results': results,
            'aggregate_stats': aggregate_stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    return aggregate_stats

def _aggregate_loso_results(results: List[Dict]) -> Dict:
    """Aggregate LOSO fold results into summary statistics"""
    metrics_lists = {}
    
    for result in results:
        for metric_name, metric_value in result['metrics'].items():
            if metric_name not in metrics_lists:
                metrics_lists[metric_name] = []
            if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                metrics_lists[metric_name].append(metric_value)
    
    aggregate = {}
    for metric_name, values in metrics_lists.items():
        if values:
            aggregate[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n_folds': len(values)
            }
    
    return aggregate

def _aggregate_loro_results(results: List[Dict]) -> Dict:
    """Aggregate LORO fold results into summary statistics"""
    return _aggregate_loso_results(results)  # Same aggregation logic

def main():
    parser = argparse.ArgumentParser(description="Cross-Domain LOSO/LORO Experiments")
    parser.add_argument("--model", type=str, required=True, choices=["enhanced", "cnn", "bilstm", "conformer_lite"])
    parser.add_argument("--protocol", type=str, required=True, choices=["loso", "loro"])
    parser.add_argument("--benchmark_path", type=str, default="benchmarks/WiFi-CSI-Sensing-Benchmark-main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/d3")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    if args.protocol == "loso":
        results = run_loso_experiment(
            args.model, args.benchmark_path, args.seed, 
            args.epochs, str(output_dir)
        )
    elif args.protocol == "loro":
        results = run_loro_experiment(
            args.model, args.benchmark_path, args.seed,
            args.epochs, str(output_dir)
        )
    
    print(f"[OK] {args.protocol.upper()} experiment completed. Results: {results}")

if __name__ == "__main__":
    main()