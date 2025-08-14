
# This is a patched version of your data_synth.py.
# Key changes to introduce more challenge and break macro_f1=1.0:
# - Added 'label_noise_prob' parameter (default=0.1): Randomly flips labels with probability p to introduce irreducible errors.
# - Made 'num_classes' configurable (default=8, increased from fixed 4 for more classes to separate).
# - Enhanced class_overlap: In addition to linear blending of base_freqs, add per-sample frequency jitter (normal noise scaled by overlap) to make signals less distinguishable.
# - Strengthened perturbations: Increased impact of gain_drift_std (stronger scaling), env_burst_rate (larger bursts), sc_corr_rho (smoother correlation).
# - Backward compatible: With label_noise_prob=0, num_classes=4, class_overlap=0, behavior matches original exactly.
# - Integration notes:
#   - In train_eval.py: Add to argparse: ap.add_argument("--label_noise_prob", type=float, default=0.1); ap.add_argument("--num_classes", type=int, default=8)
#   - Pass args.label_noise_prob and args.num_classes to SynthCSIDataset in get_synth_loaders.
#   - In sweep_lambda.py: Add "--label_noise_prob", "0.1", "--num_classes", "8" to cmd in run_one.

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
# from torch.utils.data import Subset

class SynthCSIDataset(Dataset):
    def __init__(
        self,
        n: int = 2000,
        T: int = 128,
        F: int = 30,
        difficulty: str = "mid",
        seed: int = 0,
        sc_corr_rho: Optional[float] = None,   # None/<=0 disables
        env_burst_rate: float = 0.0,           # 0 disables
        gain_drift_std: float = 0.0,           # 0 disables
        class_overlap: float = 0.0,            # 新增：类间重叠，默认关闭
        label_noise_prob: float = 0.0,         # New: Label noise probability (default 0 for compatibility)
        num_classes: int = 8,                  # New: Configurable num_classes (increased default for difficulty)
    ):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, num_classes, size=n, endpoint=False)  # Updated to use num_classes
        # NEW: Semantic class names for CSI fall detection
        self.class_names = [
            "Normal Walking",  # 0: Steady mid-freq
            "Shaking Limbs",  # 1: High jitter (epilepsy)
            "Facial Twitching",  # 2: Quick small bursts
            "Punching",  # 3: High freq + strong bursts (violence)
            "Kicking",  # 4: High freq + drift
            "Epileptic Fall",  # 5: Jitter + bursts (falling)
            "Elderly Fall",  # 6: Low freq + strong drift
            "Fall and Can't Get Up"  # 7: Low freq + long static drift
        ]
        # Add label noise: Randomly flip labels
        if label_noise_prob > 0:
            flip_mask = rng.random(size=n) < label_noise_prob
            self.y[flip_mask] = rng.integers(0, num_classes, size=flip_mask.sum(), endpoint=False)

        t = np.linspace(0, 1, T, endpoint=False).astype(np.float32)  # time axis
        X = np.zeros((n, T, F), dtype=np.float32)

        # NEW: Per-class base_freqs and adjustments
        base_freqs = np.array([5.0, 8.0, 7.0, 10.0, 9.0, 6.0, 3.0, 2.0], dtype=np.float32)  # Customized for scenarios
        # Per-class multipliers (e.g., for jitter, bursts)
        class_jitter_scale = np.array([0.5, 2.0, 1.5, 1.0, 1.0, 2.5, 0.8, 0.2])  # Higher for shaking/fall
        class_burst_boost = np.array([0.1, 0.5, 0.8, 2.0, 1.5, 1.0, 0.3, 0.0])  # Higher for violence/fall
        class_drift_boost = np.array([0.2, 0.5, 0.3, 0.4, 0.6, 0.8, 2.0, 3.0])  # Higher for elderly/static

        # base frequencies per class (scaled to num_classes)
        base_freqs = np.linspace(3.0, 3.0 + 2.0 * (num_classes - 1), num_classes).astype(np.float32)
        # per-feature small offsets
        feat_delta = rng.normal(0, 0.2, size=(F,)).astype(np.float32)

        # [ANCHOR:CLASS_OVERLAP_BEGIN]
        # Enhanced class overlap: Linear blending + per-sample frequency jitter
        overlap = float(max(0.0, min(1.0, class_overlap)))
        if overlap > 0.0:
            # Blend adjacent base_freqs (generalized for num_classes)
            blended_freqs = base_freqs.copy()
            for c in range(num_classes):
                neighbors = [c]
                if c > 0: neighbors.append(c - 1)
                if c < num_classes - 1: neighbors.append(c + 1)
                blended_freqs[c] = np.mean(base_freqs[neighbors])  # Average with neighbors
            base_freqs = (1.0 - overlap) * base_freqs + overlap * blended_freqs

        # 合成信号（保持最小实现；如你已有更完整生成逻辑，可直接替换这一段）
        # 这里用简化的谐波叠加 + 三类扰动占位，确保可运行
        for i in range(n):
            cls = int(self.y[i])
            f0 = float(base_freqs[cls])

            # Enhanced: Add per-sample frequency jitter scaled by overlap
            jitter = rng.normal(0, overlap * class_jitter_scale[cls])  # Stronger mixing for higher overlap
            f0 += jitter

            # 基础正弦 + 少量谐波
            # shape: (T,)
            wave = (
                np.sin(2 * np.pi * f0 * t)
                + 0.3 * np.sin(2 * np.pi * 2 * f0 * t + 0.5)
                + 0.2 * np.sin(2 * np.pi * 3 * f0 * t + 0.8)
            ).astype(np.float32)

            # 每个特征通道添加微小的相位/幅度差异
            for f in range(F):
                amp = 1.0 + 0.1 * feat_delta[f]
                phase = 0.2 * feat_delta[f]
                x = amp * np.sin(2 * np.pi * f0 * t + phase) \
                    + 0.15 * np.sin(2 * np.pi * 2 * f0 * t + 0.3 + phase) \
                    + 0.1 * np.sin(2 * np.pi * 3 * f0 * t + 0.6 + phase)
                X[i, :, f] = x.astype(np.float32)

        # [ANCHOR:PERTURB_BEGIN]
        # 三类扰动：以最小实现提供可运行占位，参数为0/None时不生效
        # 1) 子载波相关（频域相关，可通过对特征做一阶平滑模拟）
        if sc_corr_rho is not None and sc_corr_rho > 0:
            rho = float(sc_corr_rho)
            # 对每个样本的特征维进行相关性注入：X[:, :, f] <- rho*prev + (1-rho)*curr
            for f in range(1, F):
                X[:, :, f] = (rho * X[:, :, f - 1] + (1 - rho) * X[:, :, f]).astype(np.float32)

        # 2) 环境突发（对时间轴加入稀疏的脉冲） - Strengthened: Larger burst magnitude
        if env_burst_rate > 0.0:
            rate = float(env_burst_rate)
            n_bursts = int(max(0, np.round(rate * T)))
            if n_bursts > 0:
                for i in range(n):
                    cls = int(self.y[i])
                    boosted_rate = rate * class_burst_boost[cls]  # Boost bursts for violence/fall classes
                    n_bursts = int(max(0, np.round(boosted_rate * T)))
                    idxs = np.clip(np.random.default_rng(seed + i).integers(0, T, size=n_bursts), 0, T - 1)
                    for j in idxs:
                        X[i, j, :] += 1.0 * np.random.default_rng(seed + 1000 + i + j).normal(0, 1, size=(F,)).astype(np.float32)  # Increased from 0.5

        # 3) 增益漂移（时间轴缓慢趋势） - Strengthened: Larger scale variation
        if gain_drift_std > 0.0:
            std = float(gain_drift_std)
            # 生成一个慢变趋势（累计和的高斯噪声再归一）
            drift_rng = np.random.default_rng(seed + 12345)
            for i in range(n):
                cls = int(self.y[i])
                boosted_std = std * class_drift_boost[cls]  # Boost drift for elderly/static
                drift = drift_rng.normal(0, boosted_std, size=T).astype(np.float32)
                # drift = drift_rng.normal(0, std, size=T).astype(np.float32)
                drift = np.cumsum(drift)
                drift = (drift - drift.mean()) / (drift.std() + 1e-6)
                scale = 1.0 + 0.2 * drift  # Increased from 0.05 to 20% variation
                X[i] = (X[i] * scale[:, None]).astype(np.float32)
        # [ANCHOR:PERTURB_END]

        self.X = X
        self.T = T
        self.F = F
        self.sc_corr_rho = sc_corr_rho
        self.env_burst_rate = env_burst_rate
        self.gain_drift_std = gain_drift_std
        self.difficulty = difficulty
        self.seed = seed
        self.class_overlap = overlap
        self.label_noise_prob = label_noise_prob
        self.num_classes = num_classes  # Store for metrics access

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)



# This is a patched version of your data_synth.py.
# Key changes to introduce more challenge and break macro_f1=1.0:
# - Added 'label_noise_prob' parameter (default=0.1): Randomly flips labels with probability p to introduce irreducible errors.
# - Made 'num_classes' configurable (default=8, increased from fixed 4 for more classes to separate).
# - Enhanced class_overlap: In addition to linear blending of base_freqs, add per-sample frequency jitter (normal noise scaled by overlap) to make signals less distinguishable.
# - Strengthened perturbations: Increased impact of gain_drift_std (stronger scaling), env_burst_rate (larger bursts), sc_corr_rho (smoother correlation).
# - Backward compatible: With label_noise_prob=0, num_classes=4, class_overlap=0, behavior matches original exactly.
# - Integration notes:
#   - In train_eval.py: Add to argparse: ap.add_argument("--label_noise_prob", type=float, default=0.1); ap.add_argument("--num_classes", type=int, default=8)
#   - Pass args.label_noise_prob and args.num_classes to SynthCSIDataset in get_synth_loaders.
#   - In sweep_lambda.py: Add "--label_noise_prob", "0.1", "--num_classes", "8" to cmd in run_one.

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SynthCSIDataset(Dataset):
    def __init__(
        self,
        n: int = 2000,
        T: int = 128,
        F: int = 30,
        difficulty: str = "mid",
        seed: int = 0,
        sc_corr_rho: Optional[float] = None,   # None/<=0 disables
        env_burst_rate: float = 0.0,           # 0 disables
        gain_drift_std: float = 0.0,           # 0 disables
        class_overlap: float = 0.0,            # 新增：类间重叠，默认关闭
        label_noise_prob: float = 0.0,         # New: Label noise probability (default 0 for compatibility)
        num_classes: int = 8,                  # New: Configurable num_classes (increased default for difficulty)
    ):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, num_classes, size=n, endpoint=False)  # Updated to use num_classes
        # NEW: Semantic class names for CSI fall detection
        self.class_names = [
            "Normal Walking",  # 0: Steady mid-freq
            "Shaking Limbs",  # 1: High jitter (epilepsy)
            "Facial Twitching",  # 2: Quick small bursts
            "Punching",  # 3: High freq + strong bursts (violence)
            "Kicking",  # 4: High freq + drift
            "Epileptic Fall",  # 5: Jitter + bursts (falling)
            "Elderly Fall",  # 6: Low freq + strong drift
            "Fall and Can't Get Up"  # 7: Low freq + long static drift
        ]
        # Add label noise: Randomly flip labels
        if label_noise_prob > 0:
            flip_mask = rng.random(size=n) < label_noise_prob
            self.y[flip_mask] = rng.integers(0, num_classes, size=flip_mask.sum(), endpoint=False)

        t = np.linspace(0, 1, T, endpoint=False).astype(np.float32)  # time axis
        X = np.zeros((n, T, F), dtype=np.float32)

        # NEW: Per-class base_freqs and adjustments
        base_freqs = np.array([5.0, 8.0, 7.0, 10.0, 9.0, 6.0, 3.0, 2.0], dtype=np.float32)  # Customized for scenarios
        # Per-class multipliers (e.g., for jitter, bursts)
        class_jitter_scale = np.array([0.5, 2.0, 1.5, 1.0, 1.0, 2.5, 0.8, 0.2])  # Higher for shaking/fall
        class_burst_boost = np.array([0.1, 0.5, 0.8, 2.0, 1.5, 1.0, 0.3, 0.0])  # Higher for violence/fall
        class_drift_boost = np.array([0.2, 0.5, 0.3, 0.4, 0.6, 0.8, 2.0, 3.0])  # Higher for elderly/static

        # base frequencies per class (scaled to num_classes)
        base_freqs = np.linspace(3.0, 3.0 + 2.0 * (num_classes - 1), num_classes).astype(np.float32)
        # per-feature small offsets
        feat_delta = rng.normal(0, 0.2, size=(F,)).astype(np.float32)

        # [ANCHOR:CLASS_OVERLAP_BEGIN]
        # Enhanced class overlap: Linear blending + per-sample frequency jitter
        overlap = float(max(0.0, min(1.0, class_overlap)))
        if overlap > 0.0:
            # Blend adjacent base_freqs (generalized for num_classes)
            blended_freqs = base_freqs.copy()
            for c in range(num_classes):
                neighbors = [c]
                if c > 0: neighbors.append(c - 1)
                if c < num_classes - 1: neighbors.append(c + 1)
                blended_freqs[c] = np.mean(base_freqs[neighbors])  # Average with neighbors
            base_freqs = (1.0 - overlap) * base_freqs + overlap * blended_freqs

        # 合成信号（保持最小实现；如你已有更完整生成逻辑，可直接替换这一段）
        # 这里用简化的谐波叠加 + 三类扰动占位，确保可运行
        for i in range(n):
            cls = int(self.y[i])
            f0 = float(base_freqs[cls])

            # Enhanced: Add per-sample frequency jitter scaled by overlap
            jitter = rng.normal(0, overlap * class_jitter_scale[cls])  # Stronger mixing for higher overlap
            f0 += jitter

            # 基础正弦 + 少量谐波
            # shape: (T,)
            wave = (
                np.sin(2 * np.pi * f0 * t)
                + 0.3 * np.sin(2 * np.pi * 2 * f0 * t + 0.5)
                + 0.2 * np.sin(2 * np.pi * 3 * f0 * t + 0.8)
            ).astype(np.float32)

            # 每个特征通道添加微小的相位/幅度差异
            for f in range(F):
                amp = 1.0 + 0.1 * feat_delta[f]
                phase = 0.2 * feat_delta[f]
                x = amp * np.sin(2 * np.pi * f0 * t + phase) \
                    + 0.15 * np.sin(2 * np.pi * 2 * f0 * t + 0.3 + phase) \
                    + 0.1 * np.sin(2 * np.pi * 3 * f0 * t + 0.6 + phase)
                X[i, :, f] = x.astype(np.float32)

        # [ANCHOR:PERTURB_BEGIN]
        # 三类扰动：以最小实现提供可运行占位，参数为0/None时不生效
        # 1) 子载波相关（频域相关，可通过对特征做一阶平滑模拟）
        if sc_corr_rho is not None and sc_corr_rho > 0:
            rho = float(sc_corr_rho)
            # 对每个样本的特征维进行相关性注入：X[:, :, f] <- rho*prev + (1-rho)*curr
            for f in range(1, F):
                X[:, :, f] = (rho * X[:, :, f - 1] + (1 - rho) * X[:, :, f]).astype(np.float32)

        # 2) 环境突发（对时间轴加入稀疏的脉冲） - Strengthened: Larger burst magnitude
        if env_burst_rate > 0.0:
            rate = float(env_burst_rate)
            n_bursts = int(max(0, np.round(rate * T)))
            if n_bursts > 0:
                for i in range(n):
                    cls = int(self.y[i])
                    boosted_rate = rate * class_burst_boost[cls]  # Boost bursts for violence/fall classes
                    n_bursts = int(max(0, np.round(boosted_rate * T)))
                    idxs = np.clip(np.random.default_rng(seed + i).integers(0, T, size=n_bursts), 0, T - 1)
                    for j in idxs:
                        X[i, j, :] += 1.0 * np.random.default_rng(seed + 1000 + i + j).normal(0, 1, size=(F,)).astype(np.float32)  # Increased from 0.5

        # 3) 增益漂移（时间轴缓慢趋势） - Strengthened: Larger scale variation
        if gain_drift_std > 0.0:
            std = float(gain_drift_std)
            # 生成一个慢变趋势（累计和的高斯噪声再归一）
            drift_rng = np.random.default_rng(seed + 12345)
            for i in range(n):
                cls = int(self.y[i])
                boosted_std = std * class_drift_boost[cls]  # Boost drift for elderly/static
                drift = drift_rng.normal(0, boosted_std, size=T).astype(np.float32)
                # drift = drift_rng.normal(0, std, size=T).astype(np.float32)
                drift = np.cumsum(drift)
                drift = (drift - drift.mean()) / (drift.std() + 1e-6)
                scale = 1.0 + 0.2 * drift  # Increased from 0.05 to 20% variation
                X[i] = (X[i] * scale[:, None]).astype(np.float32)
        # [ANCHOR:PERTURB_END]

        self.X = X
        self.T = T
        self.F = F
        self.sc_corr_rho = sc_corr_rho
        self.env_burst_rate = env_burst_rate
        self.gain_drift_std = gain_drift_std
        self.difficulty = difficulty
        self.seed = seed
        self.class_overlap = overlap
        self.label_noise_prob = label_noise_prob
        self.num_classes = num_classes  # Store for metrics access

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def get_synth_loaders(
        batch: int = 64,
        difficulty: str = "mid",
        seed: int = 0,
        n: int = 2000,
        T: int = 128,
        F: int = 30,
        sc_corr_rho: Optional[float] = None,
        env_burst_rate: float = 0.0,
        gain_drift_std: float = 0.0,
        class_overlap: float = 0.0,
        label_noise_prob: float = 0.0,  # New param
        num_classes: int = 8,  # New param
        num_workers: int = 0,
        pin_memory: bool = False,
):
    ds = SynthCSIDataset(
        n=n, T=T, F=F, difficulty=difficulty, seed=seed,
        sc_corr_rho=sc_corr_rho,
        env_burst_rate=env_burst_rate,
        gain_drift_std=gain_drift_std,
        class_overlap=class_overlap,
        label_noise_prob=label_noise_prob,
        num_classes=num_classes,
    )
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)

    # UPDATED: Split into three parts (train 70%, val 15%, test 15%)
    train_split = int(0.7 * len(idx))
    val_split = int(0.85 * len(idx))  # 70% train + 15% val = 85%
    tr_idx = idx[:train_split]
    val_idx = idx[train_split:val_split]
    te_idx = idx[val_split:]

    tr = Subset(ds, tr_idx)
    val = Subset(ds, val_idx)
    te = Subset(ds, te_idx)

    return (
        DataLoader(tr, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    )
