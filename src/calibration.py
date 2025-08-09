import numpy as np

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