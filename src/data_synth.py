import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SynthCSIDataset(Dataset):
    def __init__(self, n=2000, T=128, F=30, difficulty="mid", seed=0):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, 4, size=n, endpoint=False)

        t = np.linspace(0, 1, T, endpoint=False).astype(np.float32)  # time axis
        X = np.zeros((n, T, F), dtype=np.float32)

        # base frequencies per class
        base_freqs = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        # per-feature small offsets
        feat_delta = rng.normal(0, 0.2, size=(F,)).astype(np.float32)

        # difficulty settings
        if difficulty == "easy":
            noise_std = 0.2
            amp_min, amp_max = 0.9, 1.1
            drift_std = 0.0
            drop_feat_prob = 0.0
        elif difficulty == "mid":
            noise_std = 0.5
            amp_min, amp_max = 0.7, 1.3
            drift_std = 0.1
            drop_feat_prob = 0.1
        elif difficulty == "hard":
            noise_std = 0.8
            amp_min, amp_max = 0.6, 1.4
            drift_std = 0.2
            drop_feat_prob = 0.2
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        for i in range(n):
            cls = self.y[i]
            f0 = base_freqs[cls]
            amp = rng.uniform(amp_min, amp_max)
            phase = rng.uniform(0, 2*np.pi)

            # smooth drift
            if drift_std > 0:
                drift = rng.normal(0, drift_std, size=(T,)).astype(np.float32)
                drift = np.cumsum(drift) / max(1, T // 8)
            else:
                drift = np.zeros(T, dtype=np.float32)

            for f in range(F):
                freq = f0 + feat_delta[f]
                signal = np.sin(2*np.pi*freq*t + phase)  # (T,)
                scale = 0.8 + 0.4 * rng.random()
                x = amp * scale * signal
                x = x + drift + rng.normal(0, noise_std, size=(T,)).astype(np.float32)
                X[i, :, f] = x

            # random feature noising/drop (harder)
            if drop_feat_prob > 0:
                mask = rng.random(F) < drop_feat_prob  # (F,)
                idx = np.where(mask)[0]  # (K,)
                for fidx in idx:
                    X[i, :, fidx] += rng.normal(0, noise_std * 1.5, size=(T,)).astype(np.float32)

        self.X = X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])

def get_synth_loaders(batch=64, difficulty="mid", seed=0):
    ds = SynthCSIDataset(difficulty=difficulty, seed=seed)
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    split = int(0.8*len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx); te = torch.utils.data.Subset(ds, te_idx)
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)
