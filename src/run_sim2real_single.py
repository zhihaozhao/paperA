import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.sim2real import Sim2RealEvaluator
from src.data_real import BenchmarkCSIDataset

def main():
    parser = argparse.ArgumentParser(description="Single Sim2Real Transfer Learning Experiment")
    
    # Core experiment parameters
    parser.add_argument("--model", type=str, required=True, 
                       choices=["enhanced", "cnn", "bilstm", "conformer_lite"])
    parser.add_argument("--transfer_method", type=str, required=True,
                       choices=["zero_shot", "linear_probe", "fine_tune", "temp_scale"])
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--d2_models_dir", type=str, required=True)
    parser.add_argument("--label_ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    # Method-specific parameters
    parser.add_argument("--adaptation_epochs", type=int, default=50)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"[INFO] Running Sim2Real: {args.model}, {args.transfer_method}, ratio={args.label_ratio}, seed={args.seed}")
    
    # Load benchmark dataset
    benchmark = BenchmarkCSIDataset(args.benchmark_path)
    X, y, subjects, rooms, metadata = benchmark.load_wifi_csi_benchmark()
    
    # Initialize Sim2Real evaluator
    evaluator = Sim2RealEvaluator(device=device)
    
    # Find pre-trained model from D2
    model_path = Path(args.d2_models_dir) / f"{args.model}_best.pth"
    if not model_path.exists():
        # Try alternative naming conventions
        alternative_paths = [
            Path(args.d2_models_dir) / f"{args.model}.pth",
            Path(args.d2_models_dir) / f"best_{args.model}.pth",
            Path(args.d2_models_dir) / f"{args.model}_seed{args.seed}.pth"
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"[WARNING] No pre-trained model found for {args.model}")
            print(f"[WARNING] Searched: {model_path} and alternatives")
            # Will train from scratch in this case
    
    # Load pre-trained model
    try:
        model = evaluator.load_pretrained_model(str(model_path), args.model)
        print(f"[INFO] Loaded pre-trained model from {model_path}")
    except Exception as e:
        print(f"[WARNING] Failed to load pre-trained model: {e}")
        print(f"[WARNING] Will train from scratch")
        from src.models import build_model
        model = build_model(args.model).to(device)
    
    # Create train/test splits based on label ratio
    from src.data_real import get_sim2real_loaders, RealCSIDataset
    from torch.utils.data import DataLoader
    
    if args.label_ratio < 1.0:
        train_loader, test_loader = get_sim2real_loaders(
            X, y, label_ratio=args.label_ratio, seed=args.seed, batch=64
        )
    else:
        # Use full data for 100% case
        full_ds = RealCSIDataset(X, y)
        test_loader = DataLoader(full_ds, batch_size=64, shuffle=False)
        train_loader = test_loader
    
    # Apply transfer learning method
    if args.transfer_method == "zero_shot":
        metrics = evaluator.zero_shot_evaluation(model, test_loader)
        
    elif args.transfer_method == "linear_probe":
        model, metrics = evaluator.linear_probe_adaptation(
            model, train_loader, test_loader, 
            epochs=args.adaptation_epochs, lr=1e-3
        )
        
    elif args.transfer_method == "fine_tune":
        model, metrics = evaluator.fine_tune_adaptation(
            model, train_loader, test_loader,
            epochs=args.adaptation_epochs, lr=args.fine_tune_lr
        )
        
    elif args.transfer_method == "temp_scale":
        cal_results = evaluator.temperature_calibration(model, train_loader)
        # Apply temperature and re-evaluate
        metrics = evaluator.zero_shot_evaluation(model, test_loader)
        metrics.update(cal_results)
    
    # Prepare results
    result_data = {
        'experiment': 'Sim2Real',
        'model': args.model,
        'transfer_method': args.transfer_method,
        'label_ratio': args.label_ratio,
        'seed': args.seed,
        'benchmark_metadata': metadata,
        'metrics': metrics,
        'training_config': {
            'adaptation_epochs': args.adaptation_epochs,
            'freeze_backbone': args.freeze_backbone,
            'fine_tune_lr': args.fine_tune_lr,
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"[OK] Results saved to {args.output_file}")
    print(f"[METRICS] Falling F1: {metrics.get('falling_f1', 0.0):.3f}, "
          f"Macro F1: {metrics.get('macro_f1', 0.0):.3f}, "
          f"ECE: {metrics.get('ece', 0.0):.3f}")

if __name__ == "__main__":
    main()