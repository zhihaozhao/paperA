import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

def ece(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        if in_bin.sum() > 0:
            ece += np.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * in_bin.mean()
    return float(ece)

def brier(probs, labels, num_classes=4):
    onehot = np.eye(num_classes)[labels]
    return float(((probs - onehot)**2).mean())

def nll(logits, labels):
    """
    Negative log-likelihood
    """
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    nll_values = -log_probs[np.arange(len(labels)), labels]
    return float(np.mean(nll_values))

class TemperatureScaling:
    """
    Temperature scaling for calibration
    """
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels, method='minimize'):
        """
        Fit temperature parameter on validation set
        """
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        
        def nll_loss(t):
            if t <= 0:
                return 1e6
            scaled_logits = logits / t
            return nll(scaled_logits, labels)
        
        try:
            result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
            if result.success:
                self.temperature = float(result.x)
            else:
                self.temperature = 1.0
        except:
            self.temperature = 1.0
        
        return self.temperature
    
    def transform(self, logits):
        """
        Apply temperature scaling to logits
        """
        if torch.is_tensor(logits):
            return logits / self.temperature
        else:
            return logits / self.temperature

def temperature_scaling(probs_val, y_val):
    """
    Perform temperature scaling calibration
    Returns calibrated probabilities and optimal temperature
    """
    # Convert probabilities back to logits (approximate)
    probs_val = np.clip(probs_val, 1e-8, 1-1e-8)
    logits_val = np.log(probs_val)
    
    # Fit temperature
    ts = TemperatureScaling()
    try:
        t_opt = ts.fit(logits_val, y_val)
        
        # Apply temperature scaling
        logits_cal = logits_val / t_opt
        # Convert back to probabilities
        probs_cal = np.exp(logits_cal)
        probs_cal = probs_cal / np.sum(probs_cal, axis=1, keepdims=True)
        
        return probs_cal, t_opt
    except:
        return probs_val, None