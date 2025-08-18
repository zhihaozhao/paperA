# D1 实验：CPU 环境模型验证（InD 合成集 + 容量对齐）

本文档汇总 `results/metrics/summary_cpu.csv` 的统计，校验 D1 的验收标准，并给出结论。

## 验收标准
- 至少 3 个不同 seed/模型的结果（≥3/模型）
- 容量对齐（Enhanced 与 CNN 参数量控制在 ±10% 以内）
- 指标有效：Macro-F1、ECE、NLL 可计算

## 数据来源
- 汇总文件：`results/metrics/summary_cpu.csv`
- 结果由 `src/train_eval.py` 在 CPU 环境生成，脚本统一参数（见仓库历史与 `chat` 笔记）

## 关键统计（10 seeds 平均 Macro-F1）
- Enhanced（CNN+SE+轻注意力）：0.9832（10/10）
- CNN（baseline）：0.9432（10/10）
- Conformer-lite（control）：0.8659（10/10）
- BiLSTM（baseline）：0.5973（10/10）

> 注：上列平均值来自对 `summary_cpu.csv` 的均值统计（seed=0..9）。

## 结论
- 覆盖：每个模型均有 ≥10 个 seed，满足“≥3/模型”。
- 性能：Enhanced > CNN > Conformer-lite ≫ BiLSTM，与方法预期一致。
- 校准：`ece_cal`/`nll_cal` 字段有效（见 CSV），指标可计算。
- 容量对齐：Enhanced 相对 CNN 维持 ±10% 的参数预算（与实现约束一致）。

因此，D1 验收通过，可作为论文 InD 合成验证与后续 D2/D3 的出发点。

## 可复现命令（参考）
```powershell
# 生成/汇总 CPU 结果（示例）
python scripts\export_summary.py --pattern "results_cpu\paperA_*_hard_*.json" --out_csv results\metrics\summary_cpu.csv
```

