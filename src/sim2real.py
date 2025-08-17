import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
from datetime import datetime

from src.models import build_model
from src.metrics import compute_metrics
from src.calibration import ece, brier
from torch.utils.data import DataLoader

class Sim2RealEvaluator:
    """Sim2Real Transfer Learning Evaluation Framework"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def load_pretrained_model(self, model_path: str, model_name: str) -> nn.Module:
        """Load pre-trained model from D2 experiments"""
        model = build_model(model_name)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        model.to(self.device)
        return model
    
    def zero_shot_evaluation(self, model: nn.Module, real_loader: DataLoader) -> Dict:
        """Direct evaluation of synthetic-trained model on real data"""
        model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in real_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = model(x, y=None)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probs = torch.softmax(logits, dim=1)
        
        # Compute metrics
        metrics = compute_metrics(
            labels.numpy(), 
            probs.numpy(), 
            num_classes=4, 
            positive_class=3  # Falling class
        )
        
        # Add calibration metrics
        metrics.update({
            'ece': ece(probs, labels),
            'nll': nn.functional.cross_entropy(logits, labels).item(),
            'brier': brier(probs, labels)
        })
        
        return metrics
    
    def linear_probe_adaptation(self, model: nn.Module, train_loader: DataLoader, 
                               val_loader: DataLoader, epochs=50, lr=0.001) -> Tuple[nn.Module, Dict]:
        """Freeze backbone, train only classifier on real data"""
        model.train()
        
        # Freeze all parameters except the final classifier
        for name, param in model.named_parameters():
            if 'head' in name or 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Only optimize classifier parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_metrics = None
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                logits, loss = model(x, y)
                if loss is None:
                    loss = criterion(logits, y)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0 or epoch == epochs - 1:
                metrics = self.zero_shot_evaluation(model, val_loader)
                if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                    best_metrics = metrics
        
        return model, best_metrics
    
    def fine_tune_adaptation(self, model: nn.Module, train_loader: DataLoader,
                           val_loader: DataLoader, epochs=30, lr=1e-4) -> Tuple[nn.Module, Dict]:
        """End-to-end fine-tuning with low learning rate"""
        model.train()
        
        # Unfreeze all parameters for fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_metrics = None
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                logits, loss = model(x, y)
                if loss is None:
                    loss = criterion(logits, y)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if epoch % 5 == 0 or epoch == epochs - 1:
                metrics = self.zero_shot_evaluation(model, val_loader)
                if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1']:
                    best_metrics = metrics
        
        return model, best_metrics
    
    def temperature_calibration(self, model: nn.Module, real_val_loader: DataLoader) -> Dict:
        """Calibration-only adaptation using temperature scaling"""
        from src.train_eval import calibrate_temperature
        
        # Use existing calibration function from train_eval.py
        cal_results = calibrate_temperature(model, real_val_loader, self.device)
        return cal_results