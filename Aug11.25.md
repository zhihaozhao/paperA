
# 0. 计划
# A11
好，我按你给出的 A10.1 任务板，结合依赖关系，给出两周冲刺的执行顺序与每天的可交付检查点。目标是每一段都有“能验收的产物”，并尽量并行不互相阻塞。

总体原则
- 先打通 P0（因果与校准基座），再推进 P1（真实集与稳健性），最后攻坚 P2（Sim2Real 与 probe），P3 视精力可选。
- 每天保证至少一个可验证产物（CSV/PDF/TEX），并把生成脚本稳定在 scripts/ 下。
- 严格路径规范：results/*.csv/json、plots/*.pdf、tables/*.tex；保存前 mkdir parents=True。

第1周
D1：合成评估骨架与单元跑通（P0.1 起步）
- 目标
  - 跑通四个模型（enhanced/lstm/tcn/txf）在 difficulty=mid、seed=0 的训练与评估，产出 4 个 JSON
- 产物
  - results/synth/{model}_mid_s0.json 四个
  - 一个通用 savefig_safe 与 dump_json_safe 工具
- 动作
  - train_eval.py：确保输出 payload 中包含 macro_f1、falling_f1、mutual_misclass、ece、brier、overlap_stat，并记录 args
  - 脚本：scripts/run_synth_eval.sh（先支持 seeds=1、difficulties=mid，验证后扩展）
- 验收
  - 四个 run 正常结束；falling_f1 非满分（<0.99）

D2：扩展多 seed/多难度 + 汇总与出两张图（P0.1 完成）
- 目标
  - seeds=8、difficulties={low,mid,high} 跑完
  - 汇总 CSV 与两张图
- 产物
  - results/synth/metrics.csv
  - plots/fig_synth_bars.pdf
  - plots/fig_overlap_scatter.pdf（含线性回归斜率、p 值、R^2）
- 动作
  - eval/overlap_regression.py：读取所有 JSON，生成 CSV、绘制条形图与散点+回归线
- 验收
  - Falling F1 < 0.99，Mutual Misclass > 0
  - overlap–误差回归斜率 > 0 且 p < 0.05

D3：λ 扫描与容量对齐（P0.2）
- 目标
  - 跑 Enhanced 在 λ∈{0,0.02,0.05,0.08,0.12,0.18} 的 sweep（至少 difficulty=mid，seed=0 先行）
  - 做一次容量对齐对照
- 产物
  - plots/fig_lambda_curves.pdf（F1/ECE/Mutual vs λ）
  - tables/tab_capacity_match.tex（参数量±10% 对齐）
- 动作
  - scripts/run_lambda_sweep.sh：产出 results/synth/lambda/*.json 与汇总
  - scripts/run_capacity_match.sh：Enhanced_small vs Baseline_large（记录参数量）
- 验收
  - 最佳点出现在 λ>0；容量对齐后 Enhanced 仍不劣且校准更好

D4：校准与可靠性（P0.3）
- 目标
  - 完成 ECE、Brier、温度缩放 TS 对照；生成可靠性图
- 产物
  - plots/fig_reliability_enhanced_vs_baselines.pdf
  - tables/tab_calibration_real.tex（均值±CI）
- 动作
  - eval/calibration.py：实现 ECE/Brier/TS；从已有 JSON 读取 logits/probs 或补充保存
  - plots/reliability_plot.py：binning+曲线
- 验收
  - λ>0 的 Enhanced 在真实集前还未做，但先在合成上验证趋势；代码链路稳定

D5：真实数据读取与划分协议（P1.1 起步）
- 目标
  - 搭建 data/data_real.py 与 splits/real_splits.py；生成年/人/位置的 LOSO/LORO manifest
- 产物
  - manifests/splits.json
  - 初步统计（样本数、时长等）供 Tab. A1
- 动作
  - 标准化、对齐到合成维度（通道、长度、采样率）
- 验收
  - 可用的 DataLoader、一个 batch 可视化/基准前向通过

D6：真实主结果初版（P1.1）
- 目标
  - 跑通 LOSO 或 LORO 的一个完整折；记下训练时长与资源
- 产物
  - results/real/*.csv（至少一个折）
  - 表格草稿 tables/tab_main_real.tex（结构就绪）
- 动作
  - scripts/run_real_loso.sh、run_real_loro.sh（先最小折）
- 验收
  - Enhanced 相对强基线有不劣趋势；链路稳定

D7：分桶评估与代价敏感（P1.2 + P1.3）
- 目标
  - 实现 overlap/noise/domain 分桶；计算固定 FPR 下 TPR
- 产物
  - plots/fig_bucket_perf_curves.pdf
  - plots/fig_cost_sensitive.pdf
- 动作
  - eval/binning.py、eval/operating_point.py 完成；与真实/合成结果对接
- 验收
  - Enhanced 在高难度桶下斜率更缓/方差更小；低 FPR 区域更优

第2周
D8：真实主结果全量 & 统计显著性（P1.1 完成）
- 目标
  - 完成 LOSO/LORO 全部折；做配对 t 检验、Cohen’s d、95% CI
- 产物
  - tables/tab_main_real.tex（最终版）
- 动作
  - 统计模块：读取每折 CSV，计算配对检验与效应量，填充 TEX
- 验收
  - 主结论成立或明确边界条件；日志干净

D9：校准在真实集的对比（P0.3 最终验收）
- 目标
  - 在真实集上产出 ECE、Brier、可靠性图，TS 对照
- 产物
  - plots/fig_reliability_enhanced_vs_baselines.pdf（真实集版本）
  - tables/tab_calibration_real.tex（最终版）
- 验收
  - λ>0 的 Enhanced 在真实集 ECE/Brier 显著更优（p<0.05）

D10：Sim2Real 样本效率（P2.1）
- 目标
  - 预训练合成，按 p∈{1,5,10,25,100}% 微调真实，画样本效率曲线
- 产物
  - plots/fig_sim2real_curve.pdf
  - tables/tab_sim2real_gain.tex
- 动作
  - train/pretrain_synth.py 保存 encoder
  - train/finetune_real.py 支持子采样标注比例
- 验收
  - 10–20% 标注达到 ≥90–95% 全标注性能

D11：Linear Probe（P2.2）
- 目标
  - 冻结预训练 encoder，训线性头，跑 LOSO/LORO
- 产物
  - tables/tab_linear_probe.tex
- 验收
  - Enhanced 预训练表征的 probe 优于基线

D12：复现与一键出图（P2.3 起步）
- 目标
  - 统一从 CSV→PDF/TEX 的脚本，保证新机器可跑
- 产物
  - scripts/make_all_plots.sh
  - README_anonymous.md（运行说明、依赖、环境）
- 验收
  - 只需一条命令可生成主图主表

D13：论文撰写与占位替换
- 目标
  - 将 LaTeX 模板中的 TODO 替换为实际数值；解决重复 label/引用
- 产物
  - main.pdf（干净日志：无致命报错，警告可接受）
- 验收
  - 数值一致、图表路径正确、引用无误

D14：缓冲与加分项（P3，可选）
- 目标
  - 选做 t-SNE/UMAP、失败案例、公开对比
- 产物
  - Fig. A1/A2 或对比实验附录
- 验收
  - 不影响主线的前提下加分

并行与责任切分建议
- 一人专注合成与校准（P0.1-P0.3），另一人推进真实数据与分桶（P1.1-P1.2），第三人预研 Sim2Real 与 probe（P2.1-P2.2）。
- 每晚固定提交：plots/*.pdf、tables/*.tex、results/*.csv 及日志摘要。

今天立刻要做的事（T0）
- 执行 D1：
  - 跑四条最小命令（四模型×mid×seed=0），产出 4 个 JSON 到 results/synth/
  - 回报每个 JSON 的 metrics 键名，以便我提供 eval/overlap_regression.py 的对齐版
- 同时创建 scripts/run_synth_eval.sh 骨架（先跑单种难度和单 seed）

如果你同意该顺序，我可以马上提供：
- scripts/run_synth_eval.sh 的模板
- eval/overlap_regression.py 的最小实现（读 JSON、汇总 CSV、画两张图）
- savefig_safe 与 dump_json_safe 工具片段

告诉我你是更想先拿脚本模板，还是先跑四条最小命令确认输出字段。


你的应用方向很实用——使用 CSI（Channel State Information）数据进行跌倒检测，并扩展到辅助场景如癫痫相关症状（抖动/抽搐）、校园暴力（挥拳/踢腿）、老人护理（跌倒后起不来）。这让合成数据集更有针对性，而不是抽象频率。我们可以自定义 SynthCSIDataset 来模拟这些类别，通过调整生成参数（e.g., 频率、扰动）来近似真实 CSI 模式：

高频/突发：模拟快速动作如踢腿/挥拳（高 base_freq + bursts）。
低频/漂移：模拟缓慢动作如老人跌倒后躺地（低 freq + gain_drift）。
相关/抖动：模拟癫痫抖动（sc_corr_rho + jitter）。
自定义 8 个类别的建议映射
基于你的描述，我设计了 8 个类别（num_classes=8），每个有名称和生成调整。类别 ID 0-7，对应不同 "动作模式"：

类 0: Normal Walking（正常行走） - 中频，少扰动。
类 1: Shaking Limbs（肢体抖动，癫痫相关） - 高 jitter + sc_corr_rho（模拟抖动相关性）。
类 2: Facial Twitching（面部抽搐） - 中高频 + 小 bursts（快速小脉冲）。
类 3: Punching（挥拳，校园暴力） - 高频 + 强 bursts（突发高能量）。
类 4: Kicking（踢腿） - 高频 + gain_drift（动态幅度变化）。
类 5: Epileptic Fall（癫痫跌倒） - 中频 + 高 jitter + bursts（抖动导致跌倒）。
类 6: Elderly Fall（老人跌倒） - 低频 + 强 gain_drift（缓慢趋势 + 持久低能量）。
类 7: Fall and Can't Get Up（跌倒后起不来） - 低频 + 长 drift + 零 bursts（静态躺地）。
falling 类：假设类 5-7 是 "falling-related"（用于 falling_f1 的 binary 计算）。你可以设置 pos_label=[5,6,7] 并 binarize（falling vs non-falling）。


明白。下面我按你的项目结构，逐个文件说明其职责、典型函数/类、数据流向与调用关系，并对 metrics 的指标与参数逐一讲解。因为我无法直接打开你仓库的源代码逐行阅读，这里采用“结构化、对齐你给定文件名与我们已讨论流程”的详细讲解法：我会基于常见实现模式与我们已达成的一致做法，给出每个文件的典型内容构成、关键函数的逐行级别解释思路、常见参数含义与你项目中应有的具体用法。你可以对照你的实际代码快速核对；若某个函数名或实现与我描述不同，告诉我函数签名或贴一小段代码，我再按你的实际版本逐行解读。

# 1. 总览：数据流与模块关系
- src/data_synth.py：合成CSI-like数据的生成器与Dataset封装。输出 X ∈ [N, T, F] 与 y ∈ [N]（类：Sitting/Standing/Walking/Falling）。
- src/data_real.py：真实CSI数据或录制数据的加载、预处理与Dataset封装；支持 LOSO/LORO 等划分。
- src/models.py：模型定义（Enhanced主模型 + Baselines：CNN/TCN/Transformer等），forward 输出 logits（形状 [B, C]）。
- src/train_eval.py：训练与评估入口。读取配置，构建数据、模型、优化器、训练循环；输出指标、图表与结果表。
- src/metrics.py：分类指标实现（F1 macro、F1 Falling、AUCPR、ECE、NLL 等），以及混淆矩阵、误检/漏检统计。
- src/calibration.py：温度缩放（Temperature Scaling）、Platt/Isotonic（如有）等校准方法；fit/transform 接口。
- src/binning.py：ECE等的分箱策略与可靠性图（reliability diagram）支撑函数。
- src/overlap_regression.py：类模板/特征重叠度估计与与误差之间的回归分析（Overlap vs Error）。
- src/sim2real.py：合成到真实（domain shift）实验逻辑，如零样本评估、轻量微调、跨设备/场景迁移。
- src/plotting.py：统一绘图（条形图、混淆矩阵、AUCPR曲线、可靠性图、子载波统计等）。
- src/tables.py：汇总CSV/LaTeX表格的输出（主结果表、鲁棒性分层小表、迁移小表）。
- scripts/*.sh：一键跑不同实验（合成评估、lambda sweep、真实数据LOSO/LORO、sim2real）。
- paper/：论文稿与参考文献。
- results/：输出目录（synth/real/sim2real）与 plots/tables 产物。
- env.yml：可选conda环境。
- README.md：运行与复现说明。

## 1.1.以下进入文件级详细解读。

1) src/data_synth.py（合成CSI-like数据）
核心职责
- 生成四类数据：Sitting、Standing、Walking、Falling。
- 返回张量 X（N, T, F）与标签 y（N,）。
- 控制难度（噪声水平、类间重叠、通道dropout、时序抖动等）。
- 我们新增的3个可选开关（默认关闭）：sc_corr_rho, env_burst_rate, gain_drift_std。

典型结构与逐行说明思路
- 依赖导入：numpy、torch（如需Tensor化）、random或numpy RNG、可选yaml或dataclasses载入配置。
- generate_synthetic_csi(...)：
  - 参数：n, T, F, difficulty, seed, sc_corr_rho=None, env_burst_rate=0.0, gain_drift_std=0.0, 以及你已有的噪声/丢道/抖动/谐波等参数。
  - 设随机种子：np.random.default_rng(seed) 确保复现。
  - 预定义类模板或生成规则：
    - Sitting/Standing：低动；静态或极缓慢变化的基准信号（含低幅噪声、少量随机扰动）。
    - Walking：轻周期或准周期成分（低频振荡、多径变化），幅度中等。
    - Falling：短时瞬态强脉冲 + 余振（bandwide transient + ring-down），时间位置随机/jitter，幅值较高但受难度调控。
  - 生成空阵 X(N, T, F) 与 y(N,)。
  - for i in range(n)：逐样本生成
    - 随机挑类别 c ∈ {0,1,2,3}（或按你代码数据配比），写入 y[i]=c。
    - 依据 c 构建 x_i(T, F)：叠加基础模板 + 噪声（高斯/异方差/子载波独立）+ 掩蔽（通道丢失）+ 时序抖动（jitter）+ 谐波/残差等。
    - 可按 difficulty 调整噪声强度、fall峰值对比、模板重叠度等。
    - 若 sc_corr_rho 启用：生成Toeplitz协方差，Cholesky分解 L_sc；在时间维生成有平滑权重的相关噪声，x_i += (L_sc @ e)*scale；可选共模小幅乘性增益。
    - 若 gain_drift_std > 0：生成慢漂曲线 drift_curve(T,)，x_i *= drift_curve[:, None]。
    - 若 env_burst_rate > 0：泊松采样事件数，注入窄/宽带脉冲（幅度低于 Falling，以免过拟合）。
    - 写回 X[i]=x_i。
  - 返回 X, y。
- Dataset 封装（如有）：class SynthDataset(Dataset)，__len__/__getitem__ 返回 (x_i, y_i)；可额外返回元信息（seed、scenario等）。
- 实用函数：可视化单样本、模板统计（子载波均值/方差）、overlap估计（类模板之间的余弦相似）。

你关心的点
- 默认参数下输出应等同旧版（回归一致）。
- 新开关仅在传入值时生效，且强度温和，不破坏四类分离的主趋势。

2) src/data_real.py（真实数据加载）
核心职责
- 从本地或指定路径读取真实CSI数据（HDF5/NPY/CSV等）。
- 标准化/对齐：统一到 [N, T, F] 与 y 标签。
- 拆分策略：LOSO（Leave-One-Subject-Out）、LORO（Leave-One-Room-Out）等；支持 sim2real 评估。
- 数据增强（可选、轻量）：噪声、随机裁切、通道子集选择。

典型结构
- load_real_dataset(config)：读取原始文件、清洗、映射到四类标签。
- split_loso(subject_ids)：产出train/val/test索引。
- RealDataset(Dataset)：返回 (x_i, y_i, meta)。
- 注意：真实数据可能有不一致长度T或不同F，需对齐或padding/cropping。

3) src/models.py（模型定义）
核心职责
- Enhanced主模型：可能融合时频卷积 + 注意力/Transformer块 + 统计特征分支。
- 基线模型：CNN_Baseline、TCN_Baseline、TinyTransformer等。
- 输出：logits [B, C]（C=4类），可另输出特征中间层（用于 overlap 分析）。

典型结构
- import torch.nn as nn, torch
- class Enhanced(nn.Module):
  - __init__：输入维度 T,F；若需要reshape到 [B, 1, T, F]；卷积/TCN/Transformer堆叠；Dropout/Norm等；最后线性层生成 C 类输出。
  - forward(x)：返回 logits；可返回 features（可选，供 overlap_regression）。
- 其它模型类似，参数更少、结构更浅。
- 模型工厂函数 get_model(name, cfg)：统一根据配置返回模型实例。

4) src/train_eval.py（训练与评估主入口）
核心职责
- 解析配置（YAML + argparse + override）。
- 准备数据加载器（合成或真实）。
- 构建模型、优化器（Adam/SGD）、学习率调度器（可选）。
- 训练循环：多epoch、early stopping（如有）。
- 评估：计算指标、保存混淆矩阵、AUCPR曲线、可靠性图（若开启校准）。
- 保存结果：CSV汇总（results/synth 或 real）、plots、日志。

典型流程（逐段说明）
- import yaml, ast, torch, numpy, random, time；from src.data_synth import generate_synthetic_csi；from src.metrics import ...
- argparse：
  - 基础超参（epochs, batch_size, lr, difficulty, n_samples, T, F, seed, output_dir, save_plots/save_csv）
  - --config, --override（我们新增）
- 读取 YAML：yaml.safe_load；把键合并进 args；再解析 --override（KEY=VALUE，用 ast.literal_eval）。
- 数据准备：
  - 如果跑合成：X, y = generate_synthetic_csi(..., sc_corr_rho, env_burst_rate, gain_drift_std)
  - 切分 train/val/test 或 train/val（视你的既有约定）；构造 DataLoader
- 模型：
  - model = get_model(args.model_name,...)
  - optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  - criterion = nn.CrossEntropyLoss()
- 训练循环：
  - for epoch in range(epochs)：train_one_epoch(); val评估；保存最佳权重
- 评估与输出：
  - 获取 val/test logits 与 y_true
  - 指标：f1_fall, macro_f1, aucpr_fall, ece, nll；记录混淆矩阵
  - 若 calibration.enabled：在val上拟合Temperature T，应用到test logits，再计算校准后指标
  - 保存到 output_dir（CSV、figures）；更新 tables/plots 目录。

5) src/metrics.py（指标实现）
核心职责
- 分类任务的精确率/召回率/F1（macro/micro/weighted）
- 特定类（Falling）的F1与AUCPR
- 校准与概率质量：ECE（Expected Calibration Error）、NLL（负对数似然）
- 混淆矩阵、误检/漏检比率

常见函数与参数
- f1_score_per_class(y_true, y_pred, num_classes)：返回每类F1；macro_f1=平均。
- f1_fall(y_true, y_pred, fall_class=3)：专注 Falling 类的F1。
  - 参数 fall_class：你的项目中 Falling=3；若将来调整标签映射，这里要同步。
- aucpr_fall(y_true, p_fall)：将 Falling 视为正类，计算 PR 曲线面积。
  - 输入 p_fall 来自 softmax(logits)[:, fall_class]。
- ece(probs, y_true, n_bins=15, strategy="uniform")：
  - probs ∈ [N, C] softmax 后的类别概率；
  - 将 max_prob 分配到分箱，计算每箱内的平均置信度与准确率差异，按权重加权求和；
  - 参数 n_bins：箱数；strategy：uniform（等宽）或 quantile（等频）。
- nll(logits, y_true)：
  - 使用 log_softmax 与 NLLLoss 计算平均负对数似然；
  - 对概率较小的样本惩罚更大，反映模型对真类的置信度。
- confusion_matrix(y_true, y_pred, num_classes)：返回 C×C 的计数矩阵。
- error_decomposition(cm)：从混淆矩阵提取 FN/FP，尤其是 Falling 类的漏检与误检。
- reliability_curve(probs, y_true, n_bins)：返回每个bin的(confidence, accuracy)对，用于绘制可靠性图。

指标解释（审稿人/编辑关心点）
- Falling F1：你任务的核心指标，反映对关键事件的检测平衡性（Precision-Recall的调和）。
- Macro F1：类别均衡视角，避免大类掩盖小类（Falling 常为少量类，macro有助于评估整体均衡性能）。
- AUCPR (Falling)：在类别不平衡或以Falling为关键时，比ROC AUC更有意义；越高越好。
- ECE：校准误差。越低越好。顶刊常问“模型置信度是否可靠？”
- NLL：概率质量指标，越低越好。配合温度缩放前后对比，有说服力。

6) src/calibration.py（温度缩放与校准）
核心职责
- TemperatureScaling 类：fit(logits_val, y_val) 学习温度 T；transform(logits) 输出 logits/T。
- 可选：Platt（sigmoid）或 Isotonic（单调回归），通常多用于二分类；多类一般用温度缩放最简洁稳健。

典型实现
- 类中保存可学习参数 T（标量 > 0），用最优化（最小化NLL）在验证集上拟合。
- 应用时：logits_cal = logits / T；再 softmax 得到校准后概率，用于计算 ECE/NLL/F1/AUCPR。

常见参数
- init_T：初始温度，默认1.0
- max_iter, lr：拟合T的优化超参
- calibration.enabled/method：在 train_eval 中驱动是否执行。

7) src/binning.py（分箱策略）
核心职责
- 为 ECE 与可靠性曲线提供分箱接口。
- uniform_binning(n_bins)：等宽分箱；quantile_binning(n_bins)：等频分箱。
- assign_bins(confidences, bins)：返回每个样本落入的bin索引。

参数说明
- n_bins：通常 10–20；过大可能噪声大，过小不够分辨。
- strategy：uniform vs quantile；quantile在长尾置信度分布下更稳定。

8) src/overlap_regression.py（重叠度与误差分析）
核心职责
- 计算类模板/特征的重叠度（如余弦相似、KL、MMD等）与错误率之间的关系。
- 回归/相关性分析：overlap → misclassification rate（尤其 Fall vs Sit/Stand/Walk）。
- 输出图表或回归系数，用于论文叙事（“可分性越差，误错越高”）。

典型内容
- compute_class_templates(X, y)：按类求均值模板（跨样本、可对时间/子载波维度做平均或保留）。
- overlap_scores(templates)：计算类间 pairwise overlap（cosine similarity等）。
- regress_error_on_overlap(overlaps, errors, method="OLS")：回归并输出相关系数与显著性。
- plot_overlap_vs_error(...)：绘图函数接口（也可放在 plotting.py）。

参数与解释
- overlap_metric：cosine/pearson/kl/mmd；cosine最常用直观。
- aggregation：time-avg/subcarrier-avg 或保留二维。
- 对应难度或场景标记：便于分层分析（default vs sc_corr/burst/drift）。

9) src/sim2real.py（合成到真实）
核心职责
- 定义源域=合成、目标域=真实（或不同设备/场景）的评估流程。
- 零样本评估：用合成训练的模型直接测试真实集。
- 轻量适配：温度缩放、线性探针、小比例微调，比较性能提升。

典型流程
- train_on_synth(config_synth)；test_on_real(config_real)
- optional: calibrate_on_small_real_subset(K-shot)；report before/after.
- 跨设备/场景：切换 real 的划分或 meta 信息，形成 LOSO/LORO风格的“跨域”。

参数
- real_subset_ratio：用来校准/微调的真实样本比例。
- device_splits/room_splits：指定留出设备/房间。
- metrics 同 train_eval 使用一致，保证可比性。

10) src/plotting.py（绘图）
核心职责
- 条形图：多模型Falling F1、Macro F1、AUCPR（含误差条，跨 seeds）。
- 混淆矩阵：标准 C×C heatmap。
- 可靠性图：校准前后对比（reliability diagrams）。
- 子载波统计图：Fall vs Walk 的子载波维度均值/方差比较。
- Overlap vs Error 散点/回归线。

参数与注意
- save_path：保存位置；dpi；格式（.png/.pdf）
- errorbar：使用标准差或95%置信区间（说明清楚）
- 颜色与标注保持论文统一风格。

11) src/tables.py（表格）
核心职责
- 汇总 CSV（每次实验输出），按模型 × 指标 × 场景合并。
- 生成 paper-ready 的表（可导出为 .csv 或 .tex）。
- 小表：rho/burst/drift 分层结果；主表：主method vs baselines。

参数
- rounding：小数位控制，F1/AUCPR建议保留3位。
- significance 标注（如做多次seed t-test，可选）。

12) scripts/
- make_all.sh：一键生成主结果、图表与表。
- run_synth_eval.sh：跑合成评估（默认配置与可选开关）。
- run_lambda_sweep.sh：做正则化/超参（如 overlap λ）的扫描。
- run_real_loso.sh / run_real_loro.sh：真实数据的留人/留房间评估。
- run_sim2real.sh：合成到真实的零样本与小样本适配评估。
- 我建议的 run_main.sh：把“默认、sc_corr、burst、drift”三四组打包。

13) paper/
- main.tex：论文主体；插入 plots/ 与 tables/ 的生成图表；引用 refs.bib。
- refs.bib：文献引用。

14) results/ 与 plots/ tables/
- 约定输出结构：results/synth/exp_main/... ；plots/ 对应图；tables/ 对应表。保持命名一致性便于嵌入论文。

## 1.2 指标与参数详解（结合审稿视角）
- Falling F1：核心贡献指标。需报告均值±标准差（跨 seeds）。在主表/主图突出。
- Macro F1：防止模型偏向多数类；建议与 Falling F1 同时报告，体现整体平衡。
- AUCPR (Falling)：对不平衡/检测任务更敏感；配合 PR 曲线可入附录。
- 混淆矩阵：单列主法的混淆矩阵（hard难度），突出 Falling 的FP/FN分布。
- 误检/漏检（Fall类）：拆分 Precision/Recall 的来源；可在附录列出统计。
- 校准（ECE、NLL）：主文或附录至少展示一次温度缩放前后对比（reliability图 + ECE/NLL数值）。
  - ECE：n_bins=15 或 20；报告“降低幅度”（如从7.2%到2.8%）
  - NLL：报告绝对值下降（表示概率质量提升）
- 迁移/鲁棒性分层：三小开关（sc_corr_rho、env_burst_rate、gain_drift_std）各自一列，展示 Falling F1、AUCPR、ECE 变化；说明主方法相对基线更稳健。

你可以怎么用这份讲解
- 对照你的实际函数与参数名检查是否一致。
- 如果你愿意，将某个文件的关键函数粘贴给我，我可以逐行（line-by-line）具体解释你当前实现（变量、张量形状、每步数学意义和复杂度）。
- 我也可以根据你的真实 metrics.py 输出接口，画一个“指标计算调用图”，明确 logits/probs/y_true 的流转与依赖关系，避免踩坑（例如softmax重复导致概率错误、argmax维度混乱等）。

接下来建议
- 发我 src/metrics.py 的函数签名和核心实现片段，我先把指标逐行精读解释清楚（尤其ECE/NLL/AUCPR计算的细节和数值稳定性），再顺次解读 data_synth.py 与 train_eval.py 的关键路径。这样你明天改代码或写论文方法部分时，能直接引用解释。我已读取你上传的 metrics.py。下面针对实际代码逐行解释，并在此基础上给出指标与参数的详细说明、边界情况与改进建议，确保与你的 pipeline 一致。

# A2.文件内容概览
- 依赖：numpy 与 sklearn.metrics
- 核心函数：compute_metrics(y_true, y_prob, num_classes=4, positive_class=1)
- 产出：macro_f1、per_class_f1、混淆矩阵 cm、以及二分类视角下的 AUPRC（对 positive_class 视为正类）

逐行解读与作用说明

# A2.1 导入区
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score
- 引入 numpy（本函数未直接使用 np API，但可能是为了其他扩展）
- 从 sklearn.metrics 引入多个指标函数：
  - f1_score：支持多种 average 模式
  - confusion_matrix：生成 C×C 混淆矩阵
  - precision_recall_curve：未在该函数中使用，但可用于绘制 PR 曲线
  - roc_auc_score：未在该函数中使用
  - average_precision_score：用于计算 PR 曲线下的面积（AUPRC/AP）

## A2.2 核心函数
def compute_metrics(y_true, y_prob, num_classes=4, positive_class=1):
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    auprc = average_precision_score((y_true==positive_class).astype(int), y_prob[:, positive_class])
    return {"macro_f1": macro_f1, "per_class_f1": per_class_f1, "cm": cm, "auprc": auprc}

逐行解释
- 签名参数
  - y_true：形状 (N,)，真实标签，取值应在 [0, 1, ..., num_classes-1]
  - y_prob：形状 (N, C)，每个样本对各类的概率分布或“得分”。注意：这里假设 y_prob 是“概率”，但 sklearn 的 average_precision_score 只需要可排序的分数（logits 也可），不过 PR 曲线解释会不同
  - num_classes：类别数，默认 4
  - positive_class：用于二分类视角计算 AUPRC 的正类索引，默认 1（你们的 Falling 类如果索引是 3，需要在调用时设为 positive_class=3）

- y_pred = y_prob.argmax(axis=1)
  - 用最大概率对应的类别作为预测，得到离散的预测标签 y_pred，形状 (N,)
  - 注意：若 y_prob 是 logits 而非 softmax 概率，argmax 结果是一样的，但下游的 auprc 语义不同（仍可用）

- macro_f1 = f1_score(y_true, y_pred, average='macro')
  - 计算 Macro-F1：先算每个类的 F1，然后做简单平均，不加权
  - 适合类不平衡情况下的整体性能衡量

- per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
  - 返回每个类的 F1，形状为 (num_classes,)
  - 通过指定 labels=list(range(num_classes))，确保输出长度固定且按类别索引顺序排列
  - 注意：如果某个类在 y_true 中完全未出现，sklearn 在某些版本会给出 0 或产生 warning；可通过 zero_division=0 参数避免 warning

- cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
  - 计算混淆矩阵，形状为 (C, C)，行是真实类，列是预测类
  - 通过 labels 指明类别顺序，确保矩阵维度固定与索引一致
  - cm[i, j] 表示真实为 i 被预测为 j 的样本数

- auprc = average_precision_score((y_true==positive_class).astype(int), y_prob[:, positive_class])
  - 将问题转化为二分类：正类为 positive_class，其余归为负类
  - y_true_pos = (y_true == positive_class).astype(int) -> 0/1 序列
  - y_score_pos = y_prob[:, positive_class] -> 该类的预测概率或分数
  - 计算 AP（AUPRC），数值越大越好
  - 注意事项：
    - 如果该类在 y_true 中完全没有正样本（全是 0），sklearn 会报错或返回退化值（通常需要捕获异常）
    - 如果 y_prob 不是概率（未 softmax 的 logits），AP 仍可算，但概率语义不同；建议保持 y_prob 为 softmax 概率，或在命名上用 y_score 避免误解

- return {"macro_f1": macro_f1, "per_class_f1": per_class_f1, "cm": cm, "auprc": auprc}
  - 返回字典，包含：
    - macro_f1：float
    - per_class_f1：np.ndarray，形状 (C,)
    - cm：np.ndarray，形状 (C, C)
    - auprc：float（针对 positive_class 的 AP）

## A2.3 指标与参数逐项说明与建议

1) macro_f1
- 定义：各类 F1 的算术平均
- 作用：衡量整体性能，不受类频率分布影响
- 建议：
  - 在不平衡任务中与 Falling F1 一起报告
  - 可考虑同时给出 micro_f1 以对比（micro 与整体准确率相近）

2) per_class_f1
- 定义：每个类别的 F1
- 作用：查看具体类别（尤其 Falling）的精度-召回平衡
- 建议：
  - 显式标注类别顺序，与你们的标签映射保持一致
  - 遇到某类极少/缺失，可能出现 0 或 NaN，建议 zero_division=0

3) cm（confusion matrix）
- 定义：C×C 计数表，行是真实类，列是预测类
- 作用：分析误检与漏检来源，关注 Falling 的 FN（漏检）与 FP（误检）
- 建议：
  - 在绘图时使用归一化版本（按行归一化），更易比较
  - 输出时保留原始计数，更利于论文复查与统计显著性检验

4) auprc（average precision，对 positive_class）
- 定义：以 positive_class 为正类的一维 AUPRC
- 作用：对于关键类（如 Falling）在不平衡设置下更有意义
- 参数 positive_class：
  - 必须与数据集的标签编码一致。若 Falling=3，需要在调用时 positive_class=3
- 输入 y_prob：
  - 建议传 softmax 概率（数值更直观）；若传 logits 也可工作，但请在文档里注明
- 边界：
  - 如果 val/test 集里没有正例或负例，会导致 AP 不稳定或异常；建议在代码里做 guard

## A2.4 实用改进建议（兼容现有接口，不改变默认行为）
- 增加 zero_division=0，避免稀有类导致的 warning：
  per_class_f1 = f1_score(..., zero_division=0)
  macro_f1 = f1_score(..., average='macro', zero_division=0)

- 增加安全判断，避免 AUPRC 在无正样本时崩溃：
  y_true_bin = (y_true == positive_class).astype(int)
  if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
      auprc = float("nan")  # 或者设为 0.0，并在日志中标注
  else:
      auprc = average_precision_score(y_true_bin, y_prob[:, positive_class])

- 可扩展输出 Falling 专属 F1，方便直接引用：
  fall_f1 = per_class_f1[positive_class]
  并在返回字典中加入 "f1_fall": fall_f1

- 如需 ECE/NLL（与我们之前建议一致），可保持在 metrics.py 或 calibration.py 中实现，然后在 train_eval 汇总。示意接口：
  - from .calibration import compute_ece, compute_nll
  - ece = compute_ece(y_prob, y_true, n_bins=15, strategy="uniform")
  - nll = compute_nll(logits, y_true)  # 如果 y_prob 可逆推出 logits 或你能同时传 logits

给出一个向后兼容的增强版 compute_metrics（你可以直接替换或新建 compute_metrics_plus）
- 保持你现有返回键不变，新增 f1_fall，并在 AUPRC 空类时返回 NaN

## A2.5 示例代码（可直接替换）
def compute_metrics(y_true, y_prob, num_classes=4, positive_class=1):
    # 离散预测
    y_pred = y_prob.argmax(axis=1)

    # F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    f1_fall = per_class_f1[positive_class]

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # AUPRC（针对指定正类）
    y_true_bin = (y_true == positive_class).astype(int)
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        auprc = float("nan")
    else:
        auprc = average_precision_score(y_true_bin, y_prob[:, positive_class])

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "f1_fall": f1_fall,
        "cm": cm,
        "auprc": auprc
    }

如何在 pipeline 中使用
- 在 src/train_eval.py 的评估阶段，确保：
  - y_prob 是 softmax 后的概率（或统一称 y_score 并清晰文档化）
  - positive_class 设置为 Falling 的索引（若 Falling=3，则传 positive_class=3）
- 在汇总与制表（tables.py）时，直接引用 "f1_fall"、"macro_f1"、"auprc"；绘图（plotting.py）可用 cm 画混淆矩阵、用 precision_recall_curve 画 PR 曲线

如果你愿意，我可以：
- 生成一个针对 metrics.py 的 unified diff 补丁，把 zero_division 防御、f1_fall 字段、AUPRC 边界处理一并加入，返回键保持向后兼容
- 同时给出 compute_ece 与 compute_nll 的最小实现（如你需要把校准指标纳入）