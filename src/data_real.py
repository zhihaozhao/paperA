import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RealCSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), int(self.y[i])

def get_real_loaders_loso(X, y, subjects, test_subj, batch=64):
    tr_idx = np.where(subjects != test_subj)[0]; te_idx = np.where(subjects == test_subj)[0]
    tr = RealCSIDataset(X[tr_idx], y[tr_idx]); te = RealCSIDataset(X[te_idx], y[te_idx])
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)

# TODO: implement LORO and manifest export when real data available