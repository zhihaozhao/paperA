import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SynthCSIDataset(Dataset):
    def __init__(self, n=2000, T=128, F=30, difficulty="mid", seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(0, 1, size=(n, T, F)).astype(np.float32)
        self.y = rng.integers(0, 4, size=n, endpoint=False)
        # TODO: replace with your v19.2 generator; here is placeholder
        # You can inject overlap/noise/harmonics per 'difficulty'
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), int(self.y[i])

def get_synth_loaders(batch=64, difficulty="mid", seed=0):
    ds = SynthCSIDataset(difficulty=difficulty, seed=seed)
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    split = int(0.8*len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx); te = torch.utils.data.Subset(ds, te_idx)
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)
