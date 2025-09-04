
import torch
import torch.nn as nn
import numpy as np
from sklearn.calibration import calibration_curve
import json

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Calculate ECE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def temperature_scaling(logits, temperature):
    """Apply temperature scaling"""
    return logits / temperature

def find_optimal_temperature(val_logits, val_labels, temps=np.arange(0.1, 5.0, 0.1)):
    """Find optimal temperature on validation set"""
    best_temp = 1.0
    best_nll = float('inf')
    
    for temp in temps:
        scaled_logits = temperature_scaling(val_logits, temp)
        probs = torch.softmax(torch.tensor(scaled_logits), dim=-1).numpy()
        
        # Calculate NLL
        nll = -np.mean(np.log(probs[range(len(val_labels)), val_labels] + 1e-8))
        
        if nll < best_nll:
            best_nll = nll
            best_temp = temp
    
    return best_temp

def run_calibration_test():
    """Test calibration on your models"""
    
    # Load your test predictions (you need to generate these)
    # Format: logits shape (N, num_classes), labels shape (N,)
    
    # Placeholder - replace with actual predictions
    N = 1000
    num_classes = 6
    
    # Simulate predictions (replace with actual model outputs)
    np.random.seed(42)
    logits = np.random.randn(N, num_classes) * 2
    labels = np.random.randint(0, num_classes, N)
    
    # Split into val and test
    val_size = int(0.2 * N)
    val_logits = logits[:val_size]
    val_labels = labels[:val_size]
    test_logits = logits[val_size:]
    test_labels = labels[val_size:]
    
    # Calculate raw ECE
    raw_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    raw_confidences = raw_probs.max(axis=1)
    raw_predictions = raw_probs.argmax(axis=1)
    raw_accuracies = (raw_predictions == test_labels).astype(float)
    
    ece_raw = expected_calibration_error(raw_accuracies, raw_confidences)
    
    # Find optimal temperature
    optimal_temp = find_optimal_temperature(val_logits, val_labels)
    
    # Apply temperature scaling
    calibrated_logits = temperature_scaling(test_logits, optimal_temp)
    cal_probs = torch.softmax(torch.tensor(calibrated_logits), dim=-1).numpy()
    cal_confidences = cal_probs.max(axis=1)
    cal_predictions = cal_probs.argmax(axis=1)
    cal_accuracies = (cal_predictions == test_labels).astype(float)
    
    ece_cal = expected_calibration_error(cal_accuracies, cal_confidences)
    
    # Calculate improvement
    improvement = (ece_raw - ece_cal) / ece_raw * 100
    
    results = {
        'ece_raw': round(ece_raw, 4),
        'ece_calibrated': round(ece_cal, 4),
        'optimal_temperature': round(optimal_temp, 2),
        'improvement_percent': round(improvement, 1),
        'accuracy': round(np.mean(raw_accuracies), 3)
    }
    
    print(f"ECE Raw: {results['ece_raw']}")
    print(f"ECE Calibrated: {results['ece_calibrated']}")
    print(f"Optimal Temperature: {results['optimal_temperature']}")
    print(f"Improvement: {results['improvement_percent']}%")
    
    return results

if __name__ == "__main__":
    results = run_calibration_test()
    with open('xavier_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
