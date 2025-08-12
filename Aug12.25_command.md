### 运行命令：
TEMP_MODE=logspace bash scripts/run_lambda_sweep.sh mid 0 0,0.02,0.05,0.08,0.12,0.18 --force --temp_mode logspace --temp_min 0.5 --temp_max 5 --temp_steps 60

推荐的命令串
- 先用 λ=0 求一个固定温度（可选）。例如你刚才结果显示 T≈0.5，你也可以直接指定：
  python scripts/sweep_lambda.py --only mid:0 --lambdas 0 --force --temp_mode logspace --temp_min 0.5 --temp_max 5 --temp_steps 60
  记下输出 JSON 中的 temperature，比如 0.5
- 用固定温度跑全 sweep（避免每个 λ 重复搜索）：
  python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --force --fixed_temp 0.5
- 画图：
  python scripts/plot_lambda_curves.py --csv results/synth_lambda/metrics_lambda_full.csv --out_pdf figs/fig_lambda_curves.pdf

如果你打算用同一个温度跑整条 λ 曲线（避免为每个 λ 重复搜索），可以直接固定：
--fixed_temp 0.5 --temp_mode none
比如：
python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --force --fixed_temp 0.5 --temp_mode none --val_split 0.5
绘图脚本注意参数名：
python scripts/plot_lambda_curves.py --csv results/synth_lambda/metrics_lambda_full.csv --out_pdf figs/fig_lambda_curves.pdf

注意：
- --force 会覆盖现有 JSON
- --temp_min 0.5 把最小温度抬高，避免 T=0.01 贴边
- 如果你的 sweep 脚本会把环境变量 TEMP_MODE 读取并传下去，也可以用 TEMP_MODE=logspace，但最稳妥是显式加上命令行参数 --temp_mode logspace

这条命令是在跑一次“λ 扫描”的单例实验，用验证集搜索温度标定 T。各参数含义与规则如下：

- python scripts/sweep_lambda.py
  运行 sweep 脚本，内部会调用你指定的训练/评估脚本（如 src/train_eval.py），并整理结果到 CSV/JSON。

- --only mid:0
  只跑一个难度-种子组合：difficulty=mid，seed=0。格式为 难度:种子，多组时可用逗号分隔（如 mid:0,hard:1）。

- --lambdas 0
  只评估 λ=0 这一档。可传逗号分隔的列表（如 0,0.02,0.05,…）。

- --force
  强制重跑，忽略现有缓存或已有结果文件（若脚本支持缓存/跳过机制）。

- --temp_mode logspace
  启用温度标定搜索，采用对数等分的搜索方式（在 [temp_min, temp_max] 区间内按 logspace 探索）。

- --temp_min 0.5
  温度搜索下界 T_min=0.5。

- --temp_max 5
  温度搜索上界 T_max=5。

- --temp_steps 60
  搜索步数为 60（在 logspace 上取 60 个候选 T 评估 NLL/ECE 以择优）。

运行效果
- 只会训练/评估 difficulty=mid、seed=0、λ=0 的配置；
- 使用验证集在 T∈[0.5, 5] 的对数刻度上搜索 60 个候选，选出最优温度 T_cal（通常以最小 NLL 或你实现里的目标为准）；
- 输出 JSON/CSV 里会同时包含未校准（raw）与校准（cal）的指标；若已实现 plot/汇总，则也会更新对应表格或图。

python scripts/plot_lambda_curves.py --csv results/synth_lambda/metrics_lambda_full.csv --out_pdf plots/fig_lambda_curves.pdf --difficulty mid --seed 0 --use_ece_cal

可选参数说明

--csv: 你的汇总 CSV 路径
--out_pdf: 输出的 PDF 图路径。若目录不存在，脚本会自动创建
--difficulty/--seed/--model: 仅在 CSV 包含这些列时才会应用过滤；若没有这些列，直接去掉参数
--use_ece_cal: 若加上该开关且 CSV 中存在 ece_cal 列，就会优先画 ece_cal；否则退回 ece 列
常见变体

如果 CSV 没有 difficulty 或 seed 列：
python scripts/plot_lambda_curves.py
--csv results/synth_lambda/metrics_lambda_full.csv
--out_pdf plots/fig_lambda_curves.pdf
--use_ece_cal

如果你希望画原始 ECE（不用校准）：
python scripts/plot_lambda_curves.py
--csv results/synth_lambda/metrics_lambda_full.csv
--out_pdf plots/fig_lambda_curves.pdf

成功运行的输出

生成文件 plots/fig_lambda_curves.pdf
终端会打印 Wrote plots/fig_lambda_curves.pdf


明白，你已经改好代码，现在需要“一条命令链”把数据重评估、聚合成 CSV，并重新出图。下面给你两套可直接粘贴的命令：快速版与标定版。任选其一整段执行即可。

快速版（固定温度，最快出结果）
- 适用：先验证 mutual_misclass 已成功写入，并快速完成图
- 命令（整段依次执行）:
  bash -lc '
  set -e
  # 1) 重评估 mid:0 指定 λ（覆盖旧结果）
  python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --force --fixed_temp 0.5

  # 2) 聚合为 CSV（按你项目的输出目录替换 in_dir，如有差异）
  IN_DIR=results/synth_lambda/mid_s0
  OUT_CSV=results/synth_lambda/metrics_lambda_full.csv
  python scripts/aggregate_metrics.py --in_dir "$IN_DIR" --out_csv "$OUT_CSV"

  # 3) 绘图（使用校准列名，如有 ece_cal 列则启用）
  OUT_PDF=plots/fig_lambda_curves_mid_s0.pdf
  mkdir -p "$(dirname "$OUT_PDF")"
  python scripts/plot_lambda_curves.py --csv "$OUT_CSV" --out_pdf "$OUT_PDF" --difficulty mid --seed 0 --use_ece_cal

  echo "Done. Wrote $OUT_PDF"
  '

标定版（为每个 λ 搜索最优温度，ECE 曲线更可信）
- 适用：报告用或需要更稳的 ECE
- 命令（把温度搜索参数替换为你项目默认值，如果不同）:
  bash -lc '
  set -e
  # 1) 重评估并对每个 λ 搜索温度（去掉 --fixed_temp）
  python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --force

  # 2) 聚合
  IN_DIR=results/synth_lambda/mid_s0
  OUT_CSV=results/synth_lambda/metrics_lambda_full.csv
  python scripts/aggregate_metrics.py --in_dir "$IN_DIR" --out_csv "$OUT_CSV"

  # 3) 绘图
  OUT_PDF=plots/fig_lambda_curves_mid_s0.pdf
  mkdir -p "$(dirname "$OUT_PDF")"
  python scripts/plot_lambda_curves.py --csv "$OUT_CSV" --out_pdf "$OUT_PDF" --difficulty mid --seed 0 --use_ece_cal

  echo "Done. Wrote $OUT_PDF"
  '

运行后快速自检
- 随机打开一个 JSON（如 results/synth_lambda/mid_s0/enhanced_l0.12.json），确认 metrics.mutual_misclass 是数值
- 简单检查 CSV：
  python - << "PY"
import pandas as pd
df=pd.read_csv("results/synth_lambda/metrics_lambda_full.csv")
print("rows:", len(df))
print("has mutual_misclass:", "mutual_misclass" in df.columns)
if "mutual_misclass" in df.columns:
    print("NaN ratio:", df["mutual_misclass"].isna().mean())
    print(df[["lambda","macro_f1","ece_cal" if "ece_cal" in df.columns else "ece","mutual_misclass"]].head())
PY

如果你的聚合脚本的 in_dir 路径不同，告诉我目录结构，我帮你改成精确路径。