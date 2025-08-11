from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 可直接替换的对齐版
# 默认行为不变：class_overlap=0.0 时与旧版完全一致

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
    ):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, 4, size=n, endpoint=False)

        t = np.linspace(0, 1, T, endpoint=False).astype(np.float32)  # time axis
        X = np.zeros((n, T, F), dtype=np.float32)

        # base frequencies per class
        base_freqs = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        # per-feature small offsets
        feat_delta = rng.normal(0, 0.2, size=(F,)).astype(np.float32)

        # [ANCHOR:CLASS_OVERLAP_BEGIN]
        # 类间可控重叠：仅对 base_freqs 做线性拉近。overlap∈[0,1]，默认0不变
        overlap = float(max(0.0, min(1.0, class_overlap)))
        if overlap > 0.0:
            bf = base_freqs.copy()
            # 相邻/近邻混合（不引入新函数，不影响其它逻辑）
            bf0 = (bf[0] + bf[1]) * 0.5
            bf1 = (bf[0] + bf[1] + bf[2]) / 3.0
            bf2 = (bf[1] + bf[2] + bf[3]) / 3.0
            bf3 = (bf[2] + bf[3]) * 0.5
            base_freqs = (1.0 - overlap) * bf + overlap * np.array([bf0, bf1, bf2, bf3], dtype=np.float32)
        # [ANCHOR:CLASS_OVERLAP_END]

        # 合成信号（保持最小实现；如你已有更完整生成逻辑，可直接替换这一段）
        # 这里用简化的谐波叠加 + 三类扰动占位，确保可运行
        for i in range(n):
            cls = int(self.y[i])
            f0 = float(base_freqs[cls])
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

        # 2) 环境突发（对时间轴加入稀疏的脉冲）
        if env_burst_rate > 0.0:
            rate = float(env_burst_rate)
            n_bursts = int(max(0, np.round(rate * T)))
            if n_bursts > 0:
                for i in range(n):
                    idxs = np.clip(np.random.default_rng(seed + i).integers(0, T, size=n_bursts), 0, T - 1)
                    for j in idxs:
                        X[i, j, :] += 0.5 * np.random.default_rng(seed + 1000 + i + j).normal(0, 1, size=(F,)).astype(np.float32)

        # 3) 增益漂移（时间轴缓慢趋势）
        if gain_drift_std > 0.0:
            std = float(gain_drift_std)
            # 生成一个慢变趋势（累计和的高斯噪声再归一）
            drift_rng = np.random.default_rng(seed + 12345)
            for i in range(n):
                drift = drift_rng.normal(0, std, size=T).astype(np.float32)
                drift = np.cumsum(drift)
                drift = (drift - drift.mean()) / (drift.std() + 1e-6)
                scale = 1.0 + 0.05 * drift  # 5%量级的慢变
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
    class_overlap: float = 0.0,  # 新增参数：默认0，完全向后兼容
):
    ds = SynthCSIDataset(
        n=n, T=T, F=F, difficulty=difficulty, seed=seed,
        sc_corr_rho=sc_corr_rho,
        env_burst_rate=env_burst_rate,
        gain_drift_std=gain_drift_std,
        class_overlap=class_overlap,
    )
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx)
    te = torch.utils.data.Subset(ds, te_idx)
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)