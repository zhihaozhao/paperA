"""
Unified Experiment Runner for All 4 Models
统一的实验运行和评估框架
Author: Claude 4.1
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import all experiment models
import sys
sys.path.append('docs/experiments')
from exp1_enhanced_sim2real.model_claude4 import EnhancedSim2RealModel
from exp2_enhanced_pinn_loss.model_claude4 import EnhancedModelWithPINNLoss
from exp3_pinn_lstm_causal.model_claude4 import PINNLSTMCausalModel
from exp4_mamba_efficiency.model_claude4 import MambaEfficiencyModel, LightweightMambaModel


class UnifiedExperimentRunner:
    """
    统一的实验运行器，支持所有4个模型的训练、评估和对比
    """
    
    def __init__(self,
                 experiment_name: str,
                 model_type: str,
                 data_config: Dict,
                 training_config: Dict,
                 device: str = 'cuda'):
        
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.data_config = data_config
        self.training_config = training_config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        self.results_dir = Path(f"results/{experiment_name}")
        self.checkpoint_dir = Path(f"checkpoints/{experiment_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model(model_type)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'physics_score': [],
            'efficiency_metrics': []
        }
        
    def _create_model(self, model_type: str) -> nn.Module:
        """Create model based on type"""
        input_shape = self.data_config['input_shape']
        num_classes = self.data_config['num_classes']
        
        if model_type == 'exp1_sim2real':
            return EnhancedSim2RealModel(
                input_shape=input_shape,
                num_classes=num_classes,
                use_domain_adaptation=True
            )
        elif model_type == 'exp2_pinn_loss':
            return EnhancedModelWithPINNLoss(
                input_shape=input_shape,
                num_classes=num_classes
            )
        elif model_type == 'exp3_pinn_lstm':
            return PINNLSTMCausalModel(
                input_shape=input_shape,
                num_classes=num_classes,
                use_physics_loss=True
            )
        elif model_type == 'exp4_mamba':
            return MambaEfficiencyModel(
                input_shape=input_shape,
                num_classes=num_classes
            )
        elif model_type == 'exp4_mamba_light':
            return LightweightMambaModel(
                input_shape=input_shape,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config.get('weight_decay', 0.0001)
        
        if self.training_config.get('optimizer', 'adam') == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.training_config['optimizer'] == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        physics_scores = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Model-specific forward pass
            if self.model_type == 'exp1_sim2real':
                # Sim2Real specific
                predictions = self.model(data, domain='sim')
                loss = nn.CrossEntropyLoss()(predictions, labels)
                
            elif self.model_type == 'exp2_pinn_loss':
                # PINN loss specific
                predictions = self.model(data)
                csi_data = {
                    'amplitude': data[:, 0, :, :],
                    'phase': data[:, 1, :, :]
                }
                loss, loss_dict = self.model.compute_loss(predictions, labels, csi_data, epoch)
                physics_scores.append(self.model.get_physics_consistency_score(csi_data))
                
            elif self.model_type == 'exp3_pinn_lstm':
                # PINN LSTM specific
                predictions, features = self.model(data)
                loss, loss_dict = self.model.compute_loss(predictions, labels, features)
                
            elif self.model_type in ['exp4_mamba', 'exp4_mamba_light']:
                # Mamba specific
                predictions, metrics = self.model(data)
                loss = nn.CrossEntropyLoss()(predictions, labels)
            else:
                predictions = self.model(data)
                loss = nn.CrossEntropyLoss()(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        avg_physics = np.mean(physics_scores) if physics_scores else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'physics_score': avg_physics
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Model-specific forward pass
                if self.model_type == 'exp1_sim2real':
                    predictions = self.model(data, domain='real')
                elif self.model_type == 'exp3_pinn_lstm':
                    predictions, _ = self.model(data)
                elif self.model_type in ['exp4_mamba', 'exp4_mamba_light']:
                    predictions, _ = self.model(data)
                else:
                    predictions = self.model(data)
                
                loss = nn.CrossEntropyLoss()(predictions, labels)
                
                total_loss += loss.item()
                _, predicted = predictions.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def run_experiment(self, 
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      num_epochs: int) -> Dict:
        """Run complete experiment"""
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {self.experiment_name}")
        print(f"Model Type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            if 'physics_score' in train_metrics:
                self.history['physics_score'].append(train_metrics['physics_score'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics['accuracy'])
        
        # Final evaluation
        final_results = self.evaluate_model(val_loader)
        
        # Save results
        self.save_results(final_results)
        
        return final_results
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                if self.model_type == 'exp1_sim2real':
                    predictions = self.model(data, domain='real')
                elif self.model_type == 'exp3_pinn_lstm':
                    predictions, _ = self.model(data)
                elif self.model_type in ['exp4_mamba', 'exp4_mamba_light']:
                    predictions, metrics = self.model(data)
                else:
                    predictions = self.model(data)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Compute metrics
        _, predicted_classes = all_predictions.max(1)
        accuracy = (predicted_classes == all_labels).float().mean().item() * 100
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(all_labels.numpy(), predicted_classes.numpy())
        report = classification_report(all_labels.numpy(), predicted_classes.numpy(), output_dict=True)
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Efficiency metrics
        avg_inference_time = np.mean(inference_times)
        throughput = test_loader.batch_size / avg_inference_time
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'model_type': self.model_type,
            'training_history': self.history
        }
        
        return results
    
    def save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'model_type': self.model_type,
            'config': {
                'data': self.data_config,
                'training': self.training_config
            }
        }
        
        path = self.checkpoint_dir / f'best_model.pth'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def save_results(self, results: Dict):
        """Save experiment results"""
        # Save JSON results
        results_path = self.results_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training history
        self.plot_training_history()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['confusion_matrix'])
        
        print(f"Results saved to: {self.results_dir}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png')
        plt.close()
    
    def plot_confusion_matrix(self, cm: List):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.savefig(self.results_dir / 'confusion_matrix.png')
        plt.close()


class ModelComparator:
    """
    Compare all 4 models
    """
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        
    def compare_models(self, model_names: List[str]) -> Dict:
        """Compare multiple models"""
        comparison = {}
        
        for model_name in model_names:
            results_path = self.results_dir / model_name / 'results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    comparison[model_name] = {
                        'accuracy': results['accuracy'],
                        'params': results['total_params'],
                        'inference_time': results['avg_inference_time_ms'],
                        'throughput': results['throughput_samples_per_sec']
                    }
        
        return comparison
    
    def plot_comparison(self, comparison: Dict):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(comparison.keys())
        
        # Accuracy comparison
        accuracies = [comparison[m]['accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies)
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Parameters comparison
        params = [comparison[m]['params'] for m in models]
        axes[0, 1].bar(models, params)
        axes[0, 1].set_ylabel('Parameters')
        axes[0, 1].set_title('Model Size Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        times = [comparison[m]['inference_time'] for m in models]
        axes[1, 0].bar(models, times)
        axes[1, 0].set_ylabel('Inference Time (ms)')
        axes[1, 0].set_title('Inference Speed Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughputs = [comparison[m]['throughput'] for m in models]
        axes[1, 1].bar(models, throughputs)
        axes[1, 1].set_ylabel('Samples/sec')
        axes[1, 1].set_title('Throughput Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png')
        plt.show()
    
    def generate_report(self, comparison: Dict) -> str:
        """Generate comparison report"""
        report = "="*60 + "\n"
        report += "Model Comparison Report\n"
        report += "="*60 + "\n\n"
        
        for model, metrics in comparison.items():
            report += f"{model}:\n"
            report += f"  Accuracy: {metrics['accuracy']:.2f}%\n"
            report += f"  Parameters: {metrics['params']:,}\n"
            report += f"  Inference Time: {metrics['inference_time']:.2f} ms\n"
            report += f"  Throughput: {metrics['throughput']:.1f} samples/sec\n"
            report += "\n"
        
        # Find best models
        best_accuracy = max(comparison.items(), key=lambda x: x[1]['accuracy'])
        best_speed = min(comparison.items(), key=lambda x: x[1]['inference_time'])
        smallest = min(comparison.items(), key=lambda x: x[1]['params'])
        
        report += "Best Models:\n"
        report += f"  Highest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.2f}%)\n"
        report += f"  Fastest Inference: {best_speed[0]} ({best_speed[1]['inference_time']:.2f} ms)\n"
        report += f"  Smallest Model: {smallest[0]} ({smallest[1]['params']:,} params)\n"
        
        return report


def generate_synthetic_data(num_samples: int = 1000,
                           input_shape: Tuple = (3, 114, 500),
                           num_classes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic CSI data for testing"""
    np.random.seed(42)
    
    # Generate random CSI data
    data = np.random.randn(num_samples, *input_shape).astype(np.float32)
    
    # Generate labels
    labels = np.random.randint(0, num_classes, num_samples)
    
    # Add class-specific patterns
    for i in range(num_samples):
        class_idx = labels[i]
        # Add frequency pattern
        data[i, :, class_idx*10:(class_idx+1)*10, :] += 1.5
        # Add temporal pattern
        data[i, :, :, ::class_idx+1] += 1.0
    
    return data, labels


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Unified Experiment Runner')
    parser.add_argument('--model', type=str, required=True,
                       choices=['exp1_sim2real', 'exp2_pinn_loss', 'exp3_pinn_lstm', 
                               'exp4_mamba', 'exp4_mamba_light'],
                       help='Model type to run')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison after training')
    
    args = parser.parse_args()
    
    # Data configuration
    data_config = {
        'input_shape': (3, 114, 500),
        'num_classes': 6
    }
    
    # Training configuration
    training_config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'optimizer': 'adam',
        'weight_decay': 0.0001
    }
    
    # Generate synthetic data
    print("Generating synthetic data...")
    train_data, train_labels = generate_synthetic_data(1000)
    val_data, val_labels = generate_synthetic_data(200)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_data),
        torch.from_numpy(train_labels)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_data),
        torch.from_numpy(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run experiment
    runner = UnifiedExperimentRunner(
        experiment_name=args.model,
        model_type=args.model,
        data_config=data_config,
        training_config=training_config
    )
    
    results = runner.run_experiment(train_loader, val_loader, args.epochs)
    
    # Print results
    print("\n" + "="*60)
    print("Experiment Results:")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Parameters: {results['total_params']:,}")
    print(f"Inference Time: {results['avg_inference_time_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    
    # Run comparison if requested
    if args.compare:
        print("\n" + "="*60)
        print("Running Model Comparison...")
        print("="*60)
        
        comparator = ModelComparator()
        all_models = ['exp1_sim2real', 'exp2_pinn_loss', 'exp3_pinn_lstm', 
                     'exp4_mamba', 'exp4_mamba_light']
        
        comparison = comparator.compare_models(all_models)
        report = comparator.generate_report(comparison)
        print(report)
        
        # Save report
        with open('results/comparison_report.txt', 'w') as f:
            f.write(report)
        
        # Plot comparison
        comparator.plot_comparison(comparison)


if __name__ == "__main__":
    main()