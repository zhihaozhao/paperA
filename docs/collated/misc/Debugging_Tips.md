# Debugging & Efficiency Tips (D1–D6)

## 1) 冒烟测试（Smoke Test）
- 原理：先用最小数据/最少点验证端到端流程可跑通，尽快暴露路径、依赖、维度、命名问题。
- 做法：
  - D4 用 1% 标签、1–2 个 seed、单模型（enhanced）先跑；观察日志是否命中 ckpt 与输出是否生成。
  - 命令示例：见 `docs/Runbook_D1_D6.md` 中 D4 最小集合。

## 2) 早停（Early Stopping）
- 原理：当验证指标长期不提升时提前停止，降低过拟合与算力浪费。
- 本仓库：`--min_epochs` 与 `--patience` 协同使用（满足最小轮次后若无提升即停）。
- 建议：D4 低标注设 `--min_epochs 50 --patience 8`；D2 设更短的 `--val_every`（如在训练脚本中实现）。

## 3) AMP（混合精度）
- 原理：FP16/FP32 混合，提高吞吐降低显存占用；可能带来数值不稳定，小 batch 时一般无碍。
- 本仓库：加 `--amp` 即可（已在相关训练代码分支判断 GPU 可用时启用）。

## 4) Resume（断点/去重）
- 原理：对已完成的输出不重复计算，或中断后继续。
- 本仓库：
  - Python 侧：加 `--resume`，若 `--out` 存在则直接返回（已实现）。
  - Batch 侧：`scripts\run_d4_loro.bat` 自动 [SKIP] 已存在输出。

## 5) 多线程与固定内存（num_workers / pin_memory）
- 原理：DataLoader 并行解码与固定内存页可加速主机→GPU 传输。
- 本仓库：D4 路径已支持 `--num_workers` 与 `--pin_memory`，会在 `get_sim2real_loaders` 中生效。
- 示例：`--num_workers 4 --pin_memory`。

## 6) 缓存数据（Cache）
- 原理：磁盘缓存预处理好的数据，后续重复运行直接加载。
- 本仓库：
  - 真实基准：`cache/real_benchmark/bench_*.npz`（自动生成/加载）。
  - 合成数据：`src/data_synth.py` 内置缓存逻辑（可预生成，参见 D2 脚本）。

## 7) 归一化/尺度对齐（Normalization）
- 原理：源/目标域分布差异大时，zscore/中心化可稳定迁移。
- 本仓库：D4 可用 `--input_norm zscore|center|minmax|none`（默认 zscore）。

## 8) 测试集为空的回退（label_ratio=1.0）
- 原理：当全部样本都分到训练集时，保留每类最少一个样本到测试集，或回退到全量集评估。
- 本仓库：已在 `get_sim2real_loaders` 与 D4 评估路径加入防护。

## 9) 快速定位 ckpt（精确匹配 + 递归回退）
- 原理：优先用 `final_{model}_{seed}_hard.pth` 精确匹配；否则递归模式匹配最新文件。
- 本仓库：已在 D4 加强日志打印“期望路径/候选列表/最终路径”。

## 10) 典型故障与修复
- OrderedDict 没有 eval：ckpt 为 state_dict 时需先构建模型再 `load_state_dict`（已实现）。
- 训练步数不足：低标注时调小 `--batch_size`、增大 `--min_epochs`。
- 校准差：先稳定训练，再加 `--transfer_method temp_scale` 做后校准。
