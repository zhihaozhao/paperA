# D2实验验收标准 - 快速参考

## 🎯 核心验收标准

### ✅ 必达指标
| 指标 | 标准 | 说明 |
|------|------|------|
| **Falling F1** | < 0.99 | 必须未达完美，保持任务挑战性 |
| **Mutual Misclass** | > 0 | 必须存在类间混淆，验证数据集难度 |
| **ECE改善** | Enhanced < Baseline | Enhanced模型校准效果更好 |
| **实验完整性** | ≥ 90% | 540个实验中至少完成486个 |
| **种子覆盖** | ≥ 3 seeds/组合 | 每个(模型,参数组合)至少3个种子 |

### 📊 实验规模
- **总实验数**: 540个 (4模型 × 5种子 × 3×3×3网格参数)
- **网格参数**: Class Overlap × Label Noise × Env Burst Rate
- **预估耗时**: 预生成缓存10-15小时，实验执行3-4小时

## 🚀 快速执行命令

```bash
# 1. 预生成数据集 (一次性10-15小时投入)
python scripts/pregenerate_d2_datasets.py --spec scripts/d2_spec.json

# 2. 执行D2实验 (3-4小时，从缓存加载)
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json --resume

# 3. 验收检查
python scripts/validate_d2_acceptance.py --save-report docs/d2_acceptance_report.md
```

## 📈 预期结果基准

```
正常范围:
- Macro F1: 0.88 ± 0.05
- Falling F1: 0.92 ± 0.04 (< 0.99)
- ECE (Enhanced): 0.08 ± 0.02  
- ECE (Baselines): 0.12 ± 0.03
- Mutual Misclass: 0.05 ± 0.02 (> 0)
```

## 🎉 验收通过标志

验收脚本输出: `🎉 D2实验验收通过！所有标准均已达成`

---
**详细文档**: [docs/D2_Experiment_Guide.md](D2_Experiment_Guide.md)