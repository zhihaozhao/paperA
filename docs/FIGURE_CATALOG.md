# 图表与脚本总览（Figure Catalog）

说明：本目录汇总论文中每个图的脚本、生成文件、在 `paper/main.tex` 的位置（章节与引用标签）、核心内容与作用（原因-结果链），方便审阅与复现。

## Figure 1 — 系统总览（System Overview）
- 脚本：`paper/figures/figure1-system-architecture.py`
- 产出：`paper/figures/figure1_system overview.pdf`
- 位置：引言后的系统框图，标签：`fig:system_overview`
- 内容：物理指导的合成数据生成 + 增强模型（CNN+SE+Temporal Attention）+ 可信评估流水线。
- 原因/结果：因为 WiFi CSI 受传播物理与人体交互影响，故采用“物理→合成→迁移→可信评估”的闭环设计，以保证可解释与可落地。

## Figure 2 — 协议概览（Protocols Overview）
- 脚本：`paper/figures/figure2-experimental-protocols.py`
- 产出：`paper/figures/figure4_experimental_overview.pdf`（命名历史沿用）
- 位置：Experimental Evaluation 开头，标签：`fig:protocols`
- 内容：SRD（合成鲁棒性验证）、CDAE（LOSO/LORO跨域）、STEA（Sim2Real标注效率）。
- 原因/结果：因为单一指标无法覆盖真实部署挑战，故设计三套互补协议，以量化“在合成/真实/迁移成本”三个维度的效果。

## Figure 5 — 跨域泛化（CDAE Cross-Domain）
- 脚本：`paper/figures/figure5-performance-heatmap.py`
- 产出：`paper/figures/figure-5.pdf`
- 位置：CDAE 小节，标签：`fig:cross_domain`
- 内容：LOSO/LORO 主结果、协议差距、稳定性（CV）、显著性。Enhanced 在两协议下表现一致且稳定。
- 原因/结果：因为真实落地存在“人/环境”域移，故用 LOSO/LORO 分离评测其因果效应；结果显示 Enhanced 学到域无关表征。

## Figure 6 — PCA 七视图（Feature Space Analysis）
- 脚本：`paper/figures/figure6-pca-analysis.py`
- 产出：`paper/figures/figure6_pca_analysis.pdf`
- 位置：CDAE 小节后续，标签：`fig:pca_analysis`
- 内容：PCA 主双图、协议一致性距离、载荷矩阵、方差解释率、三维投影、特征贡献。
- 原因/结果：因为需要解释“为何 Enhanced 稳定”，故以多视角揭示其在时/频特征上的均衡利用与簇结构稳定。

## Figure 7 — 标注效率（STEA Label Efficiency）
- 脚本：`paper/figures/figure6_pca_analysis.py`（同目录生成不同图，历史沿用）
- 产出：`paper/figures/figure-7.pdf`
- 位置：STEA 小节，标签：`fig:label_efficiency`
- 内容：标注比例 vs 性能曲线、成本收益、方法对比（零样本/线探/微调/校准）。
- 原因/结果：因为真实标注昂贵，故度量“标签→性能”的边际收益；结果给出 20% 标注的性价比拐点。

## D5/D6 — 压力测试与稳定性（Stress + Stability）
- 脚本（汇总）：
  - 统计与绘图：`scripts/plot_d5_d6.py`
  - 便捷运行入口：`paper/figures/plot_d5_d6.py`
- 产出：
  - 斜率图：`paper/figures/d5_d6_results_slope.pdf`
  - 表格：已内联进 `paper/main.tex`（`tab:d5d6`）
- 位置：D5/D6 小节（Experimental Evaluation 内），标签：`fig:d5d6_results`，表格 `tab:d5d6`
- 内容：D5 固定 hard 制度下的逐步增压（重叠、噪声、漂移）与多种子训练；D6 复核 GPU 多种子稳定性。
- 原因/结果：因为需要验证“在更难分布下的韧性”和“实现层随机性的可复现”，故设置 D5 压力与 D6 复核；结果显示 Enhanced 在均值/方差上最稳健。

## Figure 3 — 备选与盒线图（Comparative Options）
- 脚本：`paper/figures/figure3-enhanced-barplot.py`、`figure3-simple-boxplot.py`、`figure3-final-recommendation.py` 等
- 产出：多种中间可视化（供选型与附录/补充材料使用）
- 位置：可在附录/补充材料引用
- 内容：对比不同绘图与统计呈现方式（条形/小提琴/盒线），用于支撑主文图的设计选择。
- 原因/结果：因为单一呈现方式可能误导决策，故对比多种视觉编码，以选择既稳健又易读的表达。

## 运行说明（Reproducibility）
- 生成全部高级图：`paper/figures/generate_all_advanced_figures.py`
- SVG 导出：`paper/figures/generate_svg_figures.py`
- D5/D6：
  - 运行：`python3 paper/figures/plot_d5_d6.py`
  - 输出：`paper/figures/d5_d6_results_slope.pdf` 与内联表格数据
- 依赖：`requirements.txt`（或使用系统包安装 numpy/matplotlib）

## 与主文的映射（Map to Sections）
- 系统图 → 引言 `fig:system_overview`
- 协议图 → 实验评估开头 `fig:protocols`
- 跨域图 → CDAE `fig:cross_domain`
- PCA 七视图 → CDAE 深入解释 `fig:pca_analysis`
- 标注效率 → STEA `fig:label_efficiency`
- D5/D6 → 压力与稳定性 `fig:d5d6_results`，表 `tab:d5d6`

---
如需补充某图的细粒度生成步骤（参数/随机种子/字体等），可在对应脚本顶部补充注释，或在本目录附加子节。

