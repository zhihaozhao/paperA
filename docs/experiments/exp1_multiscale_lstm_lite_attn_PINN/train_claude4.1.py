"""
Training script for Physics-Informed Multi-Scale LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Dict
import wandb
from datetime import datetime

from models_claude4.1 import create_model
from data_loader_claude4.1 import create_dataloaders


class Trainer:
    """Trainer class for Physics-Informed CSI Model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        
        # Create dataloaders
        self.dataloaders = create_dataloaders(config['data'])
        
        # Setup training
        self.setup_training()
        
        # Setup logging
        self.setup_logging()
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['lr'] * 0.01
        )
        
        # Metrics
        self.train_metrics = {'loss': [], 'acc': []}
        self.val_metrics = {'loss': [], 'acc': [], 'f1': []}
        
    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(self.config['training']['exp_dir']) / f"exp1_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize wandb if enabled
        if self.config['training'].get('use_wandb', False):
            wandb.init(
                project="wifi-har-exp1",
                name=f"exp1_{timestamp}",
                config=self.config
            )
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Train Epoch {epoch}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data, labels)
            
            # Get losses
            losses = output['losses']
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = output['predictions']
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
            
            # Log to wandb
            if self.config['training'].get('use_wandb', False) and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ce_loss': losses['ce'].item(),
                    'train/fresnel_loss': losses.get('fresnel', torch.tensor(0)).item(),
                    'train/multipath_loss': losses.get('multipath', torch.tensor(0)).item(),
                    'train/doppler_loss': losses.get('doppler', torch.tensor(0)).item(),
                })
        
        metrics = {
            'loss': total_loss / len(self.dataloaders['train']),
            'acc': 100. * correct / total
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.dataloaders['val'], desc=f'Val Epoch {epoch}')
            
            for data, labels in pbar:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                output = self.model(data, labels)
                
                # Get losses
                losses = output['losses']
                loss = losses['total']
                
                # Metrics
                total_loss += loss.item()
                predictions = output['predictions']
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Store for F1 calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        metrics = {
            'loss': total_loss / len(self.dataloaders['val']),
            'acc': 100. * correct / total,
            'f1': f1
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
        
        # Save periodic
        if epoch % self.config['training'].get('save_freq', 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch}.pth')
    
    def train(self):
        """Main training loop"""
        print(f"Training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check if best
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_metrics['acc']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log
            print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%, F1: {val_metrics['f1']:.4f}")
            print(f"Best Val F1: {self.best_val_f1:.4f}, Best Val Acc: {self.best_val_acc:.2f}%")
            
            # Log to wandb
            if self.config['training'].get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_acc': train_metrics['acc'],
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['acc'],
                    'val/f1': val_metrics['f1'],
                    'lr': self.scheduler.get_last_lr()[0]
                })
        
        print(f"\nTraining complete!")
        print(f"Best Val F1: {self.best_val_f1:.4f}, Best Val Acc: {self.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Physics-Informed CSI Model')
    parser.add_argument('--config', type=str, default='configs/exp1_config.json',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--exp_dir', type=str, default='./experiments',
                        help='Directory for experiments')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Default config
    config = {
        'model': {
            'num_subcarriers': 30,
            'num_antennas': 3,
            'num_classes': 6,
            'hidden_dim': 128,
            'physics_config': {
                'lambda_fresnel': 0.1,
                'lambda_multipath': 0.05,
                'lambda_doppler': 0.05
            }
        },
        'data': {
            'data_path': args.data_path,
            'batch_size': args.batch_size,
            'window_size': 100,
            'stride': 50,
            'num_workers': 4
        },
        'training': {
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': 1e-4,
            'exp_dir': args.exp_dir,
            'save_freq': 10,
            'use_wandb': args.use_wandb
        }
    }
    
    # Load config file if provided
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            # Merge configs
            for key in file_config:
                if key in config:
                    config[key].update(file_config[key])
                else:
                    config[key] = file_config[key]
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()