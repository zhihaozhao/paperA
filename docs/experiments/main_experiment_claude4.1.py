"""
Main experiment script integrating training and evaluation
Supports both Exp1 (Physics-Informed LSTM) and Exp2 (Mamba)
"""

import torch
import argparse
import json
from pathlib import Path
import sys
import os
from datetime import datetime
import numpy as np

# Add experiment directories to path
sys.path.append(str(Path(__file__).parent / 'exp1_multiscale_lstm_lite_attn_PINN'))
sys.path.append(str(Path(__file__).parent / 'exp2_mamba_replacement'))
sys.path.append(str(Path(__file__).parent / 'evaluation'))

from evaluation.benchmark_loader_claude4_1 import create_benchmark_dataloaders
from evaluation.cdae_stea_evaluation_claude4_1 import CDAEEvaluator, STEAEvaluator, save_evaluation_results


def get_model_config(experiment: str, dataset_info: Dict) -> Dict:
    """Get model configuration based on experiment and dataset"""
    
    base_config = {
        'num_subcarriers': 30,  # Default, will be overridden
        'num_antennas': 3,      # Default, will be overridden
        'num_classes': dataset_info['num_classes']
    }
    
    # Adjust input dimensions based on dataset
    input_shape = dataset_info['input_shape']
    
    if experiment == 'exp1':
        # Physics-Informed Multi-Scale LSTM
        if len(input_shape) == 3:
            if 'ntu' in dataset_info.get('name', '').lower():
                base_config['num_antennas'] = input_shape[0]
                base_config['num_subcarriers'] = input_shape[1]
            elif 'ut-har' in dataset_info.get('name', '').lower():
                base_config['num_antennas'] = input_shape[0]
                base_config['num_subcarriers'] = input_shape[1]
            elif 'widar' in dataset_info.get('name', '').lower():
                # Widar has different structure (22, 20, 20)
                base_config['num_antennas'] = 1
                base_config['num_subcarriers'] = input_shape[0]
        
        config = {
            **base_config,
            'hidden_dim': 128,
            'physics_config': {
                'lambda_fresnel': 0.1,
                'lambda_multipath': 0.05,
                'lambda_doppler': 0.05
            }
        }
        
    elif experiment == 'exp2':
        # Mamba State-Space Model
        config = {
            **base_config,
            'd_model': 256,
            'd_state': 16,
            'n_layers': 4,
            'dropout': 0.1,
            'hidden_dim': 256  # For simplified version
        }
    
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    return config


def create_model(experiment: str, config: Dict):
    """Create model based on experiment type"""
    
    if experiment == 'exp1':
        from exp1_multiscale_lstm_lite_attn_PINN.models_claude4_1 import create_model
        return create_model(config)
    
    elif experiment == 'exp2':
        from exp2_mamba_replacement.models_claude4_1 import SimplifiedMamba
        # Use simplified Mamba for now (doesn't require custom CUDA kernels)
        return SimplifiedMamba(config)
    
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def train_model(model, dataloaders, config: Dict, experiment: str):
    """Train the model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training configuration
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"\nTraining {experiment} on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(dataloaders['train']):
            data, labels = data.to(device), labels.to(device)
            
            # Reshape data if needed for the model
            if experiment == 'exp1' and len(data.shape) == 3:
                # Add antenna dimension if missing
                data = data.unsqueeze(-1)
            elif experiment == 'exp2' and len(data.shape) == 3:
                # Add antenna dimension if missing
                data = data.unsqueeze(-1)
            
            optimizer.zero_grad()
            output = model(data, labels)
            
            # Get loss
            if 'losses' in output:
                loss = output['losses']['total']
            elif 'loss' in output:
                loss = output['loss']
            else:
                loss = torch.nn.functional.cross_entropy(output['logits'], labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions = output['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")
        
        avg_train_loss = total_loss / len(dataloaders['train'])
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in dataloaders['val']:
                data, labels = data.to(device), labels.to(device)
                
                # Reshape if needed
                if experiment == 'exp1' and len(data.shape) == 3:
                    data = data.unsqueeze(-1)
                elif experiment == 'exp2' and len(data.shape) == 3:
                    data = data.unsqueeze(-1)
                
                output = model(data)
                predictions = output['predictions']
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, f'{experiment}_best_model.pth')
        
        scheduler.step()
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def evaluate_model(model, dataloaders, config: Dict, experiment: str):
    """Evaluate model with CDAE and STEA protocols"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Standard evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, labels in dataloaders['test']:
            data, labels = data.to(device), labels.to(device)
            
            # Reshape if needed
            if len(data.shape) == 3:
                data = data.unsqueeze(-1)
            
            output = model(data)
            predictions = output['predictions']
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100. * test_correct / test_total
    results['test_accuracy'] = test_acc
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # CDAE evaluation (if multiple domains available)
    if config.get('evaluate_cdae', False):
        print("\nPerforming CDAE evaluation...")
        cdae_evaluator = CDAEEvaluator(model, device)
        
        # Use test loader as target domain for now
        cdae_results = cdae_evaluator.evaluate_cross_domain(
            dataloaders['train'], 
            dataloaders['test'],
            'test_domain'
        )
        results['cdae'] = cdae_results
        
        cdae_evaluator.visualize_results(f'{experiment}_cdae_results.png')
    
    # STEA evaluation (few-shot)
    if config.get('evaluate_stea', False):
        print("\nPerforming STEA evaluation...")
        stea_evaluator = STEAEvaluator(model, device)
        
        # Evaluate with different shots
        stea_results = stea_evaluator.evaluate_multiple_shots(
            dataloaders['test'],
            n_way=min(5, config['model']['num_classes']),
            k_shots=[1, 5, 10],
            n_query=15,
            n_tasks=50
        )
        results['stea'] = stea_results
        
        stea_evaluator.visualize_few_shot_results(f'{experiment}_stea_results.png')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run WiFi CSI HAR experiments')
    parser.add_argument('--experiment', type=str, default='exp1', 
                       choices=['exp1', 'exp2'],
                       help='Experiment to run (exp1: Physics-LSTM, exp2: Mamba)')
    parser.add_argument('--dataset', type=str, default='ntu-fi-har',
                       choices=['ntu-fi-har', 'ntu-fi-id', 'ut-har', 'widar'],
                       help='Dataset to use')
    parser.add_argument('--data_path', type=str, default='./Data',
                       help='Path to benchmark data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--evaluate_cdae', action='store_true',
                       help='Perform CDAE evaluation')
    parser.add_argument('--evaluate_stea', action='store_true',
                       help='Perform STEA evaluation')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{args.experiment}_{args.dataset}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    dataloaders = create_benchmark_dataloaders(
        args.dataset,
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Get dataset info
    dataset_info = dataloaders['info']
    dataset_info['name'] = args.dataset
    
    print(f"Dataset info:")
    print(f"  Input shape: {dataset_info['input_shape']}")
    print(f"  Num classes: {dataset_info['num_classes']}")
    
    # Create model configuration
    model_config = get_model_config(args.experiment, dataset_info)
    
    # Full configuration
    config = {
        'experiment': args.experiment,
        'dataset': args.dataset,
        'model': model_config,
        'training': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        },
        'evaluate_cdae': args.evaluate_cdae,
        'evaluate_stea': args.evaluate_stea
    }
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    print(f"\nCreating {args.experiment} model...")
    model = create_model(args.experiment, model_config)
    
    # Train model
    print("\nTraining model...")
    model, train_results = train_model(model, dataloaders, config, args.experiment)
    
    # Save training results
    with open(save_dir / 'train_results.json', 'w') as f:
        json.dump(train_results, f, indent=2)
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = evaluate_model(model, dataloaders, config, args.experiment)
    
    # Save evaluation results
    save_evaluation_results(eval_results, save_dir / 'eval_results.json')
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Experiment: {args.experiment}")
    print(f"Dataset: {args.dataset}")
    print(f"Best Val Acc: {train_results['best_val_acc']:.2f}%")
    print(f"Test Acc: {eval_results['test_accuracy']:.2f}%")
    
    if 'cdae' in eval_results:
        cdae = eval_results['cdae']
        print(f"CDAE - Accuracy: {cdae['accuracy']:.4f}, F1: {cdae['f1_macro']:.4f}")
    
    if 'stea' in eval_results:
        for key, value in eval_results['stea'].items():
            if 'mean_accuracy' in value:
                print(f"STEA {key}: {value['mean_accuracy']:.4f} ± {value['std_accuracy']:.4f}")
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()