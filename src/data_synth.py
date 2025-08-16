import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import hashlib
import pickle
import os
from pathlib import Path

class SynthCSIDataset(Dataset):
    def __init__(self, n=2000, T=128, F=30, difficulty="mid", seed=0,
        sc_corr_rho=None,        # None/<=0 disables
        env_burst_rate=0.0,      # 0 disables
        gain_drift_std=0.0,      # 0 disables
        cache_dir="cache/synth_data"):     # 缓存目录
        
        # 生成缓存key
        cache_key = self._generate_cache_key(n, T, F, difficulty, seed, 
                                           sc_corr_rho, env_burst_rate, gain_drift_std)
        
        # 设置缓存路径
        cache_path = Path(cache_dir) / f"synth_data_{cache_key}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载缓存
        if cache_path.exists():
            print(f"[INFO] Loading cached dataset from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.X = cached_data['X']
                self.y = cached_data['y']
                return
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}, regenerating...")
        
        # 缓存不存在或加载失败，重新生成
        print(f"[INFO] Generating new dataset (n={n}, T={T}, F={F}, difficulty={difficulty}, seed={seed})")
        self._generate_data(n, T, F, difficulty, seed, sc_corr_rho, env_burst_rate, gain_drift_std)
        
        # 保存到缓存
        try:
            cache_data = {'X': self.X, 'y': self.y}
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[INFO] Dataset cached to {cache_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")
    
    def _generate_cache_key(self, n, T, F, difficulty, seed, sc_corr_rho, env_burst_rate, gain_drift_std):
        """生成基于参数的缓存键"""
        params_str = f"n={n}_T={T}_F={F}_diff={difficulty}_seed={seed}_rho={sc_corr_rho}_burst={env_burst_rate}_drift={gain_drift_std}"
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def _generate_data(self, n, T, F, difficulty, seed, sc_corr_rho, env_burst_rate, gain_drift_std):
        """原始的数据生成逻辑"""
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

        # ===== Optional perturbations (default OFF to preserve old behavior) =====
        # Precompute subcarrier correlation if enabled
        L_sc = None
        if sc_corr_rho is not None and sc_corr_rho > 0.0:
            idx = np.arange(F)
            toe_row = (sc_corr_rho ** np.abs(idx - idx[0])).astype(np.float32)
            Sigma_sc = toe_row[np.abs(idx[:, None] - idx[None, :])].astype(np.float32)
            Sigma_sc = Sigma_sc + 1e-6 * np.eye(F, dtype=np.float32)
            L_sc = np.linalg.cholesky(Sigma_sc).astype(np.float32)

        for i in range(n):
            x_i = X[i]  # (T, F)
            T_local, F_local = x_i.shape

            # 1) Subcarrier-correlated additive noise (time-smoothed)
            if L_sc is not None:
                z_t = rng.normal(0, 1, size=(T_local,)).astype(np.float32)
                k = max(3, T_local // 16);
                k = k if k % 2 == 1 else k + 1
                kernel = np.ones(k, dtype=np.float32) / k
                z_t = np.convolve(z_t, kernel, mode="same").astype(np.float32)
                base_scale = 0.1 if difficulty in ("mid", "hard") else 0.07
                add_noise = np.empty_like(x_i, dtype=np.float32)
                for t_idx in range(T_local):
                    e = rng.normal(0, 1, size=(F_local,)).astype(np.float32)
                    add_noise[t_idx] = (L_sc @ e) * (base_scale * z_t[t_idx])
                x_i = x_i + add_noise

            # 2) Slow multiplicative gain drift
            if gain_drift_std and gain_drift_std > 0.0:
                drift_steps_extra = rng.normal(0, gain_drift_std, size=(T_local,)).astype(np.float32)
                drift_curve = 1.0 + np.cumsum(drift_steps_extra) / max(1, T_local // 8)
                drift_curve = np.clip(drift_curve, 0.8, 1.2).astype(np.float32)
                x_i = x_i * drift_curve[:, None]

            # 3) Environmental bursts (narrow/wide-band)
            if env_burst_rate and env_burst_rate > 0.0:
                n_events = rng.poisson(max(0.0, env_burst_rate))
                for _ in range(n_events):
                    t0 = rng.integers(0, T_local)
                    width = rng.integers(max(2, T_local // 64), max(3, T_local // 16))
                    t1 = min(T_local, t0 + width)
                    if t1 <= t0:
                        continue
                    window = np.hanning(t1 - t0).astype(np.float32)
                    amp = rng.uniform(0.3, 0.8)
                    if rng.random() < 0.6:
                        # narrow-band
                        f_idx = rng.integers(0, F_local)
                        x_i[t0:t1, f_idx] = x_i[t0:t1, f_idx] + amp * window
                    else:
                        # wide-band with possible correlated shape across subcarriers
                        bump = amp * window[:, None]
                        if L_sc is not None:
                            e = rng.normal(0, 1, size=(F_local,)).astype(np.float32)
                            shape_f = (L_sc @ e)
                        else:
                            shape_f = rng.normal(0, 1, size=(F_local,)).astype(np.float32)
                        shape_f = shape_f / (np.linalg.norm(shape_f) + 1e-6)
                        x_i[t0:t1, :] = x_i[t0:t1, :] + bump * shape_f[None, :]

        self.X = X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i])

def get_synth_loaders(batch=64, difficulty="mid", seed=0,
                      n=2000, T=128, F=30,
                      sc_corr_rho=None, env_burst_rate=0.0, gain_drift_std=0.0,
                      cache_dir="cache/synth_data"):
    ds = SynthCSIDataset(n=n, T=T, F=F, difficulty=difficulty, seed=seed,
                         sc_corr_rho=sc_corr_rho,
                         env_burst_rate=env_burst_rate,
                         gain_drift_std=gain_drift_std,
                         cache_dir=cache_dir)
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    split = int(0.8*len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx); te = torch.utils.data.Subset(ds, te_idx)
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)
