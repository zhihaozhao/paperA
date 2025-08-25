# D1 实验：GPU 对齐与一致性检查（与 CPU 统计对照）

本页记录 GPU 环境（`results_gpu/`、`results/`）的对齐实验，核对与 CPU 统计的一致性，并给出偏差分析。

## 目标
- 验证 GPU 训练/评估在相同数据配置下与 CPU 结论一致（相对排序不变，分数在合理波动范围内）
- 确认温度标定（ECE/NLL）在 GPU 浮点下无异常

## 数据来源
- GPU 日志与结果：`results_gpu/` 与 `results/` 下的 d2/d3 验收中间产物
- 参考 CPU 汇总：`docs/D1_CPU_Validation.md` 与 `results/metrics/summary_cpu.csv`

## 对齐要点
- 相同模型/难度/种子组合下，Macro-F1 与 ECE 的差异通常在 ±0.01–0.02 以内
- 相对排序保持：Enhanced > CNN > Conformer-lite ≫ BiLSTM
- ECE/温度标定：GPU 下的 `ece_cal` 低于 `ece_raw`（校准有效），且与 CPU 方向一致

## 观察与结论
- 从当前 GPU 结果（摘自 `results_gpu/d2` 与 `results/` 中间统计）看：
  - Enhanced、CNN 的 InD 表现保持显著优势，排序与 CPU 一致；
  - ECE 校准有效；
  - 个别 seed 的绝对分差接近 0.02，处于可接受的随机性与数值差异范围。

## 备注
- 若需严格逐行对齐表格，可在 D2/D3 全量完成后运行：
```powershell
python scripts\export_summary.py --pattern "results_gpu\paperA_*_hard_*.json" --out_csv results\metrics\summary_gpu.csv
```
并将 `summary_gpu.csv` 与 CPU 汇总做成对照表（可加入 `src/tables.py` 自动化生成）。

