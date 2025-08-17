# Route-A Pipeline (Quick Start)
- conda env create -f env.yml && conda activate csi-fall-route-a
- python -m pip install torch --index-url https://download.pytorch.org/whl/cu121  # 根据GPU情况调整
- bash scripts/make_all.sh
Outputs:
- plots/*.pdf
- tables/*.tex
- paper/main.pdf (compiled manuscript)

## 运行约定
- CPU=冒烟测试（验证链路与产物，不做长时间训练）；GPU=产出研究结果。

### CPU 冒烟（约1–3分钟）
```bash
REM Windows CMD
sweep_local.bat
```
该脚本仅运行一次轻量模型：生成 `results_cpu/smoke_cnn.json` 与对应 `.log`，用于验证：
- 训练/验证/测试流程可跑通
- 独立日志与结果 JSON 生成
- 后续汇总脚本能正确读取

### GPU 实验
使用 `sweep.bat`（已默认启用 AMP、保存 final 模型、降低验证频率）。所有 GPU 结果输出到 `results_gpu/`，便于与 CPU 冒烟区分。