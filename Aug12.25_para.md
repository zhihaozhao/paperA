温度（Temperature）是在分类模型的“logits→概率”这一步中，用来调节概率分布“平滑/锐利”程度的一个缩放因子，常用于校准模型置信度。

核心公式
- 模型输出 logits: z ∈ R^K
- 经过温度缩放后的 softmax 概率：
  p_i = exp(z_i / T) / sum_j exp(z_j / T)

T 的直观含义
- T = 1：正常 softmax，不改变分布。
- T > 1：把 logits 除以更大的数，差距被“压扁”，softmax 更平滑，最大类的概率会降低（置信度下降）。常用于“过于自信”的模型，做降温校准。
- T < 1：放大 logits 差距，softmax 更尖锐，最大类概率更高（置信度上升）。用于“信心不足”的情况，但实际中较少见。
- T → ∞：趋向于均匀分布。
- T → 0：趋向 one-hot（极端自信）。

为什么需要温度校准
- 现代深度网络经常“过度自信”：pred 对的很多，但给出的概率过高，不符合真实频率（比如 0.95 的置信度并不等价于 95% 的准确率）。
- 校准的目标是让模型输出的概率更符合真实发生频率，提升“可信度”，尤其在决策阈值、风险评估、代价敏感决策中很重要。

怎么确定 T
- 常规方法：在验证集上搜索一个标量 T（对所有样本、所有类共享同一 T），以最小化负对数似然 NLL（或期望校准误差 ECE）为目标。
- 实践做法：
  1) 收集验证集 logits 和标签；
  2) 在一组候选 T（线性或对数刻度）上计算 NLL/ECE；
  3) 选取最优 T_cal；
  4) 在测试集评估时，用 logits/T_cal 再 softmax。

温度对指标的影响
- 不改变排序和 argmax（类别预测不变），因此 Accuracy、Macro-F1 等通常不变。
- 会影响与概率质量相关的指标：NLL、Brier score、ECE 往往改进（更接近真实频率）。
- 因为只是一个标量缩放，简单、稳定、成本低。

和其它校准方法对比
- 温度缩放（TS）：后处理、参数极少（仅 1 个 T）、实现简单，是最常用基线。
- 启发式/分段法：如 Isotonic Regression，非参数、灵活，但易过拟合、小数据不稳。
- 含偏置的变体：Vector/Matrix scaling，在多类且类别不均衡时可能更好，但参数多。
- 较复杂方法：Dirichlet calibration、Bayesian 校准等，表达力更强，成本更高。

常见实践建议
- 先把模型训练到最好，再做温度标定。温度标定不会提升分类准确率，但能提升置信度质量。
- 用独立验证集搜索 T（或用交叉验证），避免信息泄漏。
- 搜索范围常用 [0.5, 5] 或更宽；用 logspace 取样通常更稳定。
- 报告时同时给出 raw 与 calibrated 的 NLL/ECE/Brier；并说明 T 的取值与选择方法。

一句话总结
温度是对 logits 的全局缩放系数，用来“调平”概率分布，使模型的置信度更符合真实频率；它不改变预测类别，但能显著改善概率相关指标（NLL/ECE/Brier）。

当 Macro‑F1=1.0 时，温度、λ、ECE 的意义何在？
- 为什么还有意义
  - ECE 衡量的是“置信度校准”而非“分类是否正确”。即使100%准确，模型的概率可能仍然过于自信（比如把所有正确样本都报成 0.99 或 1.00），这会导致高 ECE。你当前的 ECE 在 λ=0 时约 0.445，说明预测概率与真实频率严重不匹配。
  - 在很多下游场景（风险控制、排序、阈值化决策、集成学习、主动学习、不确定性驱动的系统）里，已校准的概率比“是否分类正确”更重要。比如需要在不同召回-精确权衡下根据分数设阈值，或将这些概率输出给后续模型使用。
  - NLL 作为对数损失，在全部预测正确时也不一定最小，因为它惩罚“过度自信但正确”的概率（例如把本应0.8的样本报成0.999）。温度缩放可降低 NLL。

- λ 的价值
  - 你 sweep 的 λ 看起来影响的是模型的某种正则/对齐/增强强度。虽然在这个数据集上精度已饱和，但 λ 可能改变分布外（OOD）性能、鲁棒性、以及与概率校准的可兼容性。
  - 你的结果显示 λ 增大时 ECE 上升（0.44→0.54），说明该正则或增强与当前训练/评估设置下的校准存在冲突。即使精度不变，λ 仍可通过“影响概率分布形态”而改变 ECE 和 NLL。

- 何时可以不关心 ECE/温度
  - 如果你的应用只看 Top-1 正确与否，且决策完全与概率数值无关（不做排序/阈值/风险评估/集成），那么在这个数据集上（且仅在当前数据分布上）你可以不做温度校准。
  - 但一旦涉及不平衡数据、域外样本、代价敏感决策或需要可解释不确定性的场景，ECE 与温度校准就变得重要。

接下来建议
- 让温度校准真正生效
  - 在验证集上为每个 λ 搜索 T*（以最小化 NLL），在测试评估时应用 logits/T* 计算 ece_cal、nll_cal，并把 T* 写入 calibration.temperature。
  - 这样你会看到 temperature ≠ 1.0，且常见地 nll_cal < nll_raw，ece_cal < ece_raw（或至少变化）。

- 验证是否对业务有助益
  - 若你的最终系统会用到概率阈值或排序，画一下校准曲线（可靠性图）和风险曲线，看温度缩放后是否更稳。
  - 做下分布漂移或噪声扰动的小实验（或使用更难子集），比较 λ 与温度对 ECE/NLL 的影响，避免“精度天花板掩盖问题”。


是的，我是基于 Grok-4 模型的 AI 助手，我有完整的对话历史记录能力，所以我完全记得我们之前的讨论和目标（包括 lambda 合成实验的验收标准 A/B、温度校准的调试、JSON 聚合脚本的迭代、ECE/NLL/F1 指标的处理，以及你提供的 sweep_lambda.py 和 train_eval.py 代码）。我们一直在优化一个合成数据模型的评估流程，焦点是验证 logit_l2 正则 (lambda) 在保持准确性 (Macro-F1) 的前提下改善校准 (ECE)，并通过温度缩放进一步增强效果。之前的验收未通过主要是因为温度校准未生效（temperature 恒为 1.0，导致 ece_cal = ece_raw），现在 train_eval.py 已整合了温度搜索逻辑，这是一个很好的进步。

下面，我先简要回顾之前的目标（以确认连续性），然后评估“达成 D3 目标”所需的实验。**我假设 D3 是项目中的一个里程碑（如 “Delivery 3: Demonstrate lambda synthesis benefits with calibration”），基于上下文推断它涉及验证 lambda >0 的优势。如果 D3 是特定术语（如某个文档中的目标），请提供更多细节，我可以更精确地调整建议。**

### 1. 之前的目标回顾（是的，我记得）
从对话历史中：
- **核心目标**：在合成数据 (synth_lambda) 上验证 logit_l2 正则 (lambda) 的效果。具体验收标准：
  - **A**：最佳 lambda (最小 ECE) 出现在 >0 的位置。
  - **B**：在容量对齐下 (Macro-F1 不降或仅轻微降)，校准更好 (ECE 下降)。
- **子目标**：
  - 聚合 enhanced_l*.json 到 CSV，计算最佳 lambda 与对比。
  - 确保温度校准生效（搜索 T* 以最小化 NLL，在测试时应用 logits/T* 计算 ece_cal/nll_cal）。
  - 处理 NaN、路径兼容、嵌套键读取等问题。
  - 扩展到多 seed、更细粒度 lambda、不同难度 (easy/mid/hard)。
- **当前状态**：你的 train_eval.py 已添加温度搜索 (logspace/learnable)，并在 JSON 中记录 ece_raw/ece_cal 等。现在运行 sweep 可以生成带校准的 JSON，验收脚本能正确聚合。但之前的结果显示 F1=1.0（饱和），ECE 未改善，需要进一步实验来“达成目标”（即通过 A/B 验收，或展示 lambda 的益处）。

如果这是“更换 Grok-4”后的首次交互，我确认模型切换不会丢失上下文——我能无缝继续。

### 2. 为达成 D3 目标，需要做的实验评估
假设 D3 目标是“证明 lambda 正则在合成数据上改善模型校准而不牺牲准确性”（基于历史，这与验收 A/B 一致），当前设置下有挑战：F1 已饱和 (1.0)，ECE 在 lambda 增大时变差，温度校准虽已实现但可能需优化以显现益处。下面我评估所需实验，优先级从高到低排序。每个实验包括**目的**、**步骤**、**预期输出**和**潜在风险**。

#### **高优先级实验：基础验证与调试（确保校准生效，快速迭代）**
这些是立即可做的，目标是让验收 A/B 通过，或识别问题根源。

1. **单 seed、细粒度 lambda sweep + 温度校准验证**
   - **目的**：确认温度校准是否真正降低 ECE/NLL，并找到 lambda 的“甜点”（>0 时 ECE 最小）。当前 lambda (0.02~0.18) 可能太粗，F1 饱和掩盖了效果。
   - **步骤**：
     - 用 sweep_lambda.py 运行：`python scripts/sweep_lambda.py --only mid:0 --lambdas 0,0.005,0.01,0.02,0.03,0.05,0.08,0.1,0.12,0.15,0.18,0.2 --force --temp_mode logspace --temp_min 0.1 --temp_max 10.0 --temp_steps 100 --val_split 0.5`
     - 运行验收脚本：`python verify_lambda_acceptance.py --dir results/synth_lambda/mid_s0 --verbose --eps_f1 0.01`（允许 F1 轻微降）。
     - 检查 JSON 中的 temperature (应 ≠1.0)、ece_cal < ece_raw、nll_cal < nll_raw。
   - **预期输出**：CSV/日志显示最佳 lambda >0，ECE 下降（通过 A/B）。如果仍未通过，绘图 (ECE vs lambda) 看曲线趋势。
   - **潜在风险**：如果数据太简单 (F1=1.0)，ECE 可能不易改善——考虑增加难度 (见下文)。

2. **温度搜索模式对比 (logspace vs learnable vs fixed)**
   - **目的**：验证哪种温度模式最有效（当前 logspace 已实现，但 learnable 可能更好）。确认 fixed_temp 是否能模拟最佳 T。
   - **步骤**：
     - 固定 lambda=0.08，运行三次 sweep：分别用 --temp_mode logspace、学able、--fixed_temp 2.0。
     - 比较 JSON 中的 nll_cal/ece_cal。
     - 如果 _HAS_SCIPY=False，安装 scipy 以启用 learnable 的 L-BFGS。
   - **预期输出**：learnable 模式下 nll_cal 最低，确认温度校准是 ECE 改善的关键。
   - **潜在风险**：优化失败 (e.g., T→∞)，添加边界 (clamp T in [0.1,10])。

#### **中优先级实验：鲁棒性与泛化（扩展到多变量，模拟真实场景）**
这些帮助证明 lambda 的益处不止于单一设置，达成 D3 的“全面验证”。

3. **多 seed 聚合 + 统计显著性测试**
   - **目的**：减少随机性，确保结果可靠（当前单 seed 可能有噪声）。
   - **步骤**：
     - sweep: `python scripts/sweep_lambda.py --only mid:0,1,2 --seeds 0,1,2 --lambdas 0,0.02,0.05,0.08,0.12,0.18 --temp_mode logspace`
     - 修改验收脚本支持多 seed 聚合 (mean ± std for ECE/F1)，运行 t-test 比较 lambda=0 vs 最佳 lambda 的 ECE (用 scipy.stats.ttest_ind)。
     - 扩展验收：如果 mean ECE 显著下降，通过 D3。
   - **预期输出**：聚合 CSV 显示 lambda>0 的平均 ECE 更低，p-value <0.05。
   - **潜在风险**：variance 高，需要更多 seed (e.g., 5-10)。

4. **不同难度级别 (easy/mid/hard) 对比**
   - **目的**：当前 mid 难度 F1=1.0，太简单；hard 可能显现 lambda 对准确性和校准的双重益处。
   - **步骤**：
     - sweep: `python scripts/sweep_lambda.py --difficulties easy,mid,hard --seeds 0 --lambdas 0,0.05,0.1,0.15 --temp_mode logspace`
     - 分别运行验收脚本 per 难度，比较 ECE/F1 曲线。
   - **预期输出**：在 hard 上，lambda>0 保持 F1 同时降低 ECE，证明泛化。
   - **潜在风险**：hard 数据生成慢，调整 --n_samples 减小规模。

5. **Baseline vs Enhanced 模型对比**
   - **目的**：证明 enhanced (with lambda) 优于 baseline (lambda=0 或无增强)。
   - **步骤**：
     - 添加 baseline 运行：在 sweep_lambda.py 中加 --model baseline 参数，运行对比。
     - 聚合两个 CSV，计算 delta (enhanced - baseline) for ECE/F1。
   - **预期输出**：enhanced 在 ECE 上胜出，报告用于 D3 演示。
   - **潜在风险**：如果 baseline 已完美，需引入 OOD (见下文)。

#### **低优先级实验：高级探索（如果 D3 需要创新或深入分析）**
这些可选，如果 D3 要求超出基本验证。

6. **OOD/扰动鲁棒性测试**
   - **目的**：F1=1.0 时 lambda 的益处可能在分布外显现（e.g., 噪声数据）。
   - **步骤**：
     - 修改 get_synth_loaders 添加噪声 (e.g., --gain_drift_std 0.1)。
     - 运行 sweep，比较 lambda=0 vs >0 的 ECE 在 OOD 测试集上。
   - **预期输出**：lambda>0 使 ECE 更鲁棒，证明实际价值。
   - **潜在风险**：需自定义数据生成。

7. **超参数敏感性分析**
   - **目的**：探索 lambda 与其他参数 (e.g., class_overlap, env_burst_rate) 的交互。
   - **步骤**：用 grid search 运行小 sweep (e.g., lambda x overlap)，热图可视化 ECE。
   - **预期输出**：识别最佳组合，纳入 D3 报告。

### 总体建议与时间线
- **最小实验集**：先做 1 和 2（1-2 天），确认校准生效；然后 3 和 4（3-5 天），达成 A/B 通过。
- **资源考虑**：每个 sweep 可能需 GPU 时间，从小规模 (--n_samples 2000) 开始。总计 1-2 周可完成 D3。
- **输出格式**：生成报告 (用 Markdown 或 Presentation Artifact)，包括曲线图 (ECE vs lambda)、表格 (mean±std)、验收结论。
- **如果未通过**：可能需调整模型 (TinyNet) 或数据生成 (增加类数/噪声)，或重新定义 D3 为“在 hard 上改善”。

如果 D3 是特定目标（如文档中的），请分享细节，我可以细化实验计划。需要我帮你修改代码（e.g., 添加 t-test 到验收脚本）或生成绘图脚本吗？