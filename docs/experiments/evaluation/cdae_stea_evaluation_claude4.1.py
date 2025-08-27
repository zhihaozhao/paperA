"""
Cross-Domain Activity Evaluation (CDAE) and Small-Target Environment Adaptation (STEA) protocols
For WiFi CSI-based HAR evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import random


class CDAEEvaluator:
    """Cross-Domain Activity Evaluation protocol"""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def evaluate_cross_domain(self, 
                             source_loader, 
                             target_loader,
                             domain_name: str) -> Dict:
        """
        Evaluate model trained on source domain on target domain
        
        Args:
            source_loader: DataLoader for source domain (for reference)
            target_loader: DataLoader for target domain
            domain_name: Name of target domain
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for data, labels in tqdm(target_loader, desc=f"Evaluating on {domain_name}"):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                output = self.model(data)
                predictions = output['predictions']
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store features for domain analysis
                if 'features' in output:
                    all_features.append(output['features'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Per-class accuracy
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        results = {
            'domain': domain_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_accuracy': per_class_acc.tolist()
        }
        
        # Domain shift analysis if features available
        if all_features:
            features = np.concatenate(all_features, axis=0)
            results['feature_stats'] = {
                'mean': features.mean(axis=0).tolist(),
                'std': features.std(axis=0).tolist(),
                'feature_dim': features.shape[-1]
            }
        
        self.results[domain_name] = results
        
        return results
    
    def evaluate_multiple_domains(self, 
                                 source_loader,
                                 target_loaders: Dict) -> Dict:
        """
        Evaluate on multiple target domains
        
        Args:
            source_loader: Source domain dataloader
            target_loaders: Dictionary of target domain dataloaders
        
        Returns:
            Aggregated results across all domains
        """
        all_results = {}
        
        for domain_name, loader in target_loaders.items():
            print(f"\nEvaluating cross-domain transfer to {domain_name}")
            results = self.evaluate_cross_domain(source_loader, loader, domain_name)
            all_results[domain_name] = results
            
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1 Macro: {results['f1_macro']:.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'mean_accuracy': np.mean([r['accuracy'] for r in all_results.values()]),
            'std_accuracy': np.std([r['accuracy'] for r in all_results.values()]),
            'mean_f1': np.mean([r['f1_macro'] for r in all_results.values()]),
            'std_f1': np.std([r['f1_macro'] for r in all_results.values()])
        }
        
        all_results['average'] = avg_metrics
        
        return all_results
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Visualize cross-domain evaluation results"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, len(self.results), figsize=(5*len(self.results), 10))
        
        if len(self.results) == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (domain, results) in enumerate(self.results.items()):
            # Confusion matrix
            conf_matrix = np.array(results['confusion_matrix'])
            sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[0, idx], cmap='Blues')
            axes[0, idx].set_title(f'{domain} - Confusion Matrix')
            axes[0, idx].set_xlabel('Predicted')
            axes[0, idx].set_ylabel('Actual')
            
            # Per-class accuracy bar plot
            per_class_acc = results['per_class_accuracy']
            axes[1, idx].bar(range(len(per_class_acc)), per_class_acc)
            axes[1, idx].set_title(f'{domain} - Per-Class Accuracy')
            axes[1, idx].set_xlabel('Class')
            axes[1, idx].set_ylabel('Accuracy')
            axes[1, idx].set_ylim([0, 1])
            
            # Add overall metrics as text
            text = f"Acc: {results['accuracy']:.3f}\nF1: {results['f1_macro']:.3f}"
            axes[1, idx].text(0.02, 0.98, text, transform=axes[1, idx].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class STEAEvaluator:
    """Small-Target Environment Adaptation protocol for few-shot learning"""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def create_few_shot_tasks(self, 
                              dataloader,
                              n_way: int = 5,
                              k_shot: int = 5,
                              n_query: int = 15,
                              n_tasks: int = 100) -> List[Dict]:
        """
        Create few-shot learning tasks from dataset
        
        Args:
            dataloader: DataLoader to sample from
            n_way: Number of classes per task
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            n_tasks: Number of tasks to create
        
        Returns:
            List of few-shot tasks
        """
        # Collect all data by class
        data_by_class = {}
        
        for data, labels in dataloader:
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in data_by_class:
                    data_by_class[label] = []
                data_by_class[label].append(data[i])
        
        # Create tasks
        tasks = []
        available_classes = list(data_by_class.keys())
        
        for _ in range(n_tasks):
            # Sample N classes
            if len(available_classes) < n_way:
                continue
                
            task_classes = random.sample(available_classes, n_way)
            
            support_set = []
            support_labels = []
            query_set = []
            query_labels = []
            
            for new_label, class_id in enumerate(task_classes):
                class_data = data_by_class[class_id]
                
                if len(class_data) < k_shot + n_query:
                    continue
                
                # Sample support and query
                indices = random.sample(range(len(class_data)), k_shot + n_query)
                
                for i in range(k_shot):
                    support_set.append(class_data[indices[i]])
                    support_labels.append(new_label)
                
                for i in range(k_shot, k_shot + n_query):
                    query_set.append(class_data[indices[i]])
                    query_labels.append(new_label)
            
            if len(support_set) == n_way * k_shot:
                tasks.append({
                    'support': torch.stack(support_set),
                    'support_labels': torch.tensor(support_labels),
                    'query': torch.stack(query_set),
                    'query_labels': torch.tensor(query_labels)
                })
        
        return tasks
    
    def adapt_and_evaluate(self, 
                           task: Dict,
                           adaptation_steps: int = 10,
                           adaptation_lr: float = 0.01) -> Dict:
        """
        Adapt model on support set and evaluate on query set
        
        Args:
            task: Few-shot task dictionary
            adaptation_steps: Number of gradient steps for adaptation
            adaptation_lr: Learning rate for adaptation
        
        Returns:
            Evaluation results
        """
        # Clone model for adaptation
        adapted_model = self.model
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=adaptation_lr)
        
        support_data = task['support'].to(self.device)
        support_labels = task['support_labels'].to(self.device)
        query_data = task['query'].to(self.device)
        query_labels = task['query_labels'].to(self.device)
        
        # Adaptation phase
        adapted_model.train()
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            output = adapted_model(support_data, support_labels)
            
            if 'loss' in output:
                loss = output['loss']
            else:
                loss = nn.CrossEntropyLoss()(output['logits'], support_labels)
            
            loss.backward()
            optimizer.step()
        
        # Evaluation phase
        adapted_model.eval()
        with torch.no_grad():
            output = adapted_model(query_data)
            predictions = output['predictions']
        
        # Calculate metrics
        accuracy = accuracy_score(query_labels.cpu(), predictions.cpu())
        f1 = f1_score(query_labels.cpu(), predictions.cpu(), average='macro')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'n_way': len(torch.unique(support_labels)),
            'k_shot': len(support_labels) // len(torch.unique(support_labels))
        }
    
    def evaluate_few_shot(self,
                         dataloader,
                         n_way: int = 5,
                         k_shot: int = 5,
                         n_query: int = 15,
                         n_tasks: int = 100,
                         adaptation_steps: int = 10) -> Dict:
        """
        Full few-shot evaluation
        
        Args:
            dataloader: DataLoader for target domain
            n_way: Number of classes per task
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            n_tasks: Number of tasks to evaluate
            adaptation_steps: Number of adaptation steps
        
        Returns:
            Aggregated results
        """
        # Create tasks
        tasks = self.create_few_shot_tasks(dataloader, n_way, k_shot, n_query, n_tasks)
        
        if not tasks:
            return {'error': 'Could not create tasks'}
        
        # Evaluate each task
        results = []
        for task in tqdm(tasks, desc=f"{n_way}-way {k_shot}-shot evaluation"):
            result = self.adapt_and_evaluate(task, adaptation_steps)
            results.append(result)
        
        # Aggregate results
        aggregated = {
            f'{n_way}_way_{k_shot}_shot': {
                'mean_accuracy': np.mean([r['accuracy'] for r in results]),
                'std_accuracy': np.std([r['accuracy'] for r in results]),
                'mean_f1': np.mean([r['f1_score'] for r in results]),
                'std_f1': np.std([r['f1_score'] for r in results]),
                'n_tasks': len(results)
            }
        }
        
        self.results.update(aggregated)
        
        return aggregated
    
    def evaluate_multiple_shots(self,
                               dataloader,
                               n_way: int = 5,
                               k_shots: List[int] = [1, 5, 10],
                               n_query: int = 15,
                               n_tasks: int = 100) -> Dict:
        """
        Evaluate with different number of shots
        
        Args:
            dataloader: Target domain dataloader
            n_way: Number of classes
            k_shots: List of shot numbers to evaluate
            n_query: Number of query samples
            n_tasks: Number of tasks per configuration
        
        Returns:
            Results for all configurations
        """
        all_results = {}
        
        for k_shot in k_shots:
            print(f"\nEvaluating {n_way}-way {k_shot}-shot...")
            results = self.evaluate_few_shot(dataloader, n_way, k_shot, n_query, n_tasks)
            all_results.update(results)
            
            key = f'{n_way}_way_{k_shot}_shot'
            if key in results:
                print(f"  Accuracy: {results[key]['mean_accuracy']:.4f} ± {results[key]['std_accuracy']:.4f}")
                print(f"  F1 Score: {results[key]['mean_f1']:.4f} ± {results[key]['std_f1']:.4f}")
        
        return all_results
    
    def visualize_few_shot_results(self, save_path: Optional[str] = None):
        """Visualize few-shot learning results"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Extract shot numbers and accuracies
        shots = []
        accuracies = []
        stds = []
        
        for key in sorted(self.results.keys()):
            if 'way' in key and 'shot' in key:
                parts = key.split('_')
                k_shot = int(parts[2])
                shots.append(k_shot)
                accuracies.append(self.results[key]['mean_accuracy'])
                stds.append(self.results[key]['std_accuracy'])
        
        if not shots:
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(shots, accuracies, yerr=stds, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Shots (K)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Few-Shot Learning Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Add value labels
        for i, (x, y, std) in enumerate(zip(shots, accuracies, stds)):
            plt.text(x, y + std + 0.02, f'{y:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def save_evaluation_results(results: Dict, save_path: str):
    """Save evaluation results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("CDAE and STEA evaluation protocols implemented")
    print("Use CDAEEvaluator for cross-domain evaluation")
    print("Use STEAEvaluator for few-shot adaptation evaluation")