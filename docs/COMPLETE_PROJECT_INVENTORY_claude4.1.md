# 📚 Paper A 项目完整目录清单与使用说明

**更新日期**: 2024年12月28日  
**项目名称**: WiFi CSI HAR with Physics-Guided Synthetic Data  
**仓库地址**: https://github.com/zhihaozhao/paperA  
**文档版本**: v2.0 (包含最新LaTeX文档和Docker基础设施)

---

## 一、🗂️ 项目总体结构

```
/workspace/ (paperA根目录)
├── 📄 paper/                    # 论文相关文件
├── 📊 results/                   # 实验结果
├── 🔬 scripts/                   # 实验脚本
├── 💻 src/                       # 源代码
├── 📚 docs/                      # 文档目录
├── 📖 references/                # 参考文献
├── 🗃️ benchmark_data_claude4.1/  # 基准数据集
└── 🧪 experiments/               # Claude 4.1新增实验
```

---

## 二、📁 核心目录详细说明

### 1. 📄 **paper/** - 论文目录
```
paper/
├── main.tex                      # 主论文文件 (8页会议论文)
├── main_backup.tex               # 备份版本
├── refs.bib                      # 参考文献库 (60+篇高质量文献)
├── enhanced/                     # 增强版论文
│   ├── enhanced.tex              # 增强版论文
│   ├── enhanced_claude4.1opus.tex # Claude扩展版 (58,691字符)
│   └── plots/                    # 绘图脚本
├── zero/                         # Zero-shot论文
│   ├── zeroshot.tex              # Zero-shot版本
│   └── zeroshot_claude4.1opus.tex # Claude扩展版 (55,895字符)
├── figures/                      # 图表文件
│   ├── *.py                      # Python绘图脚本
│   ├── *.csv                     # 数据文件
│   └── *.tex                     # LaTeX图表
└── PlotPy/                       # 高级绘图脚本
```

**用途**: 包含所有论文版本、图表生成脚本和参考文献

### 2. 📊 **results/** - 实验结果
```
results/
├── d2/                           # 实验D2: 校准分析
├── d3/                           # 实验D3: 跨域评估
│   ├── loro/                    # Leave-One-Room-Out
│   └── loso/                    # Leave-One-Subject-Out
├── d4/                           # 实验D4: Sim2Real迁移
│   └── sim2real/                 # 仿真到真实迁移结果
├── d5/                           # 实验D5: 消融研究
├── d6/                           # 实验D6: 可信度分析
├── metrics/                      # 汇总指标
│   ├── summary_*.csv             # 各实验汇总
│   └── acceptance_report.txt    # 验收报告
└── README.md                     # 结果说明文档
```

**用途**: 存储所有实验结果JSON文件和性能指标

### 3. 🔬 **scripts/** - 实验脚本
```
scripts/
├── 核心训练脚本
│   ├── run_main.sh              # 主训练脚本
│   ├── run_train.sh             # 训练启动器
│   ├── run_infer.sh             # 推理脚本
│   └── run_sweep_from_json.py   # 参数扫描
├── 数据分析脚本
│   ├── analyze_d2_results.py    # D2结果分析
│   ├── analyze_d3_d4_for_figures.py # D3/D4图表生成
│   ├── generate_paper_figures.py # 论文图表生成
│   └── create_results_summary.py # 结果汇总
├── 验收脚本
│   ├── accept_d2.py             # D2验收
│   ├── accept_d3_d4.py          # D3/D4验收
│   └── validate_d*_acceptance.py # 各实验验收
└── 环境配置
    ├── env.sh                    # 环境变量
    └── make_all.sh              # 一键运行
```

**用途**: 自动化实验执行、结果分析和验收

### 4. 💻 **src/** - 源代码
```
src/
├── 核心模型
│   ├── models.py                # 基础模型 (CNN, BiLSTM, Conformer)
│   ├── models_pinn.py           # 物理信息神经网络
│   └── models_back.py           # 备份模型
├── 数据处理
│   ├── data_synth.py            # 合成数据生成器
│   ├── data_real.py             # 真实数据加载器
│   └── data_cache.py            # 数据缓存
├── 训练评估
│   ├── train_eval.py            # 训练评估主函数
│   ├── train_cross_domain.py    # 跨域训练
│   ├── evaluate.py              # 评估函数
│   └── infer.py                 # 推理接口
├── 特殊功能
│   ├── pinn_losses.py           # 物理损失函数
│   ├── calibration.py           # 模型校准
│   ├── reliability.py           # 可靠性分析
│   └── sim2real.py              # Sim2Real迁移
└── utils/                        # 工具函数
    ├── logger.py                 # 日志系统
    ├── registry.py               # 模型注册
    └── exp_recorder.py           # 实验记录
```

**用途**: 核心算法实现、模型定义、训练框架

### 5. 📚 **docs/** - 文档目录（12月28日更新）
```
docs/
├── experiments/                  # Claude 4.1实验文档
│   ├── exp1_enhanced_sim2real/  # Exp1: Enhanced Sim2Real模型
│   ├── exp2_enhanced_pinn_loss/ # Exp2: Enhanced + PINN损失
│   ├── exp3_pinn_lstm_causal/   # Exp3: PINN LSTM + 因果注意力
│   ├── exp4_mamba_efficiency/   # Exp4: Mamba效率模型
│   ├── thesis/                   # 博士论文章节
│   │   ├── exp1_comprehensive_analysis_claude4.1.tex # 100K+字符Exp1分析
│   │   ├── EXPERIMENTAL_ARCHITECTURE_CN_claude4.1.md # 中文架构文档
│   │   ├── exp[1-4]_*.bib       # 各实验BibTeX参考文献
│   │   ├── validate_latex_claude4.1.py # LaTeX验证脚本
│   │   └── LATEX_COMPILATION_GUIDE_claude4.1.md # 编译指南
│   ├── docker/                   # Docker部署
│   │   ├── Dockerfile_claude4.1  # Docker镜像定义
│   │   ├── docker-compose.yml    # 7个服务编排
│   │   ├── DOCKER_DEPLOYMENT_GUIDE_claude4.1.md # 部署指南
│   │   └── build_docker.sh       # 构建脚本
│   ├── evaluation/               # 评估框架
│   │   ├── benchmark_loader_claude4.1.py # 统一数据加载器
│   │   └── cdae_stea_evaluation_claude4.1.py # CDAE/STEA评估
│   ├── new_directions/           # 5个新研究方向
│   │   ├── direction[1-5]_*.md  # 各方向详细文档
│   │   └── paper_drafts/         # 40K+字符论文草稿
│   ├── paper_drafts/             # 实验论文草稿
│   │   ├── exp1_extended_claude4.1.tex # 73K字符
│   │   └── exp2_extended_claude4.1.tex # 77K字符
│   ├── unified_experiment_runner_claude4.1.py # 统一实验运行器
│   ├── STEP_BY_STEP_GUIDE_claude4.1.md # 详细操作指南
│   └── HOW_TO_RUN_claude4.1.md  # 快速使用指南
└── daily/                        # 工作日志
    ├── work_report_Dec27_claude4.1.md # 12月27日工作报告
    ├── GIT_COMMITS_SUMMARY_claude4.1.md # Git提交总结(40+commits)
    ├── VALUABLE_COMMITS_ANALYSIS_claude4.1.md # 价值提交分析
    └── FILE_STRUCTURE_MAP_claude4.1.md # 文件结构图
```

**用途**: 项目文档、实验说明、工作记录

---

## 三、🚀 快速开始指南

### 环境配置
```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量
source scripts/env.sh
```

### 运行实验
```bash
# 基础实验
cd scripts
./run_main.sh

# 参数扫描
python run_sweep_from_json.py --spec ../specs/D2_pinn_lstm_ms.json

# Claude 4.1新实验
python ../docs/experiments/main_experiment_claude4.1.py --experiment exp1
```

### 生成图表
```bash
# 生成论文所有图表
python scripts/generate_paper_figures.py

# 生成特定实验图表
python paper/figures/plot_d3_cross_domain.py
```

### 结果分析
```bash
# 分析D2实验
python scripts/analyze_d2_results.py

# 生成汇总报告
python scripts/create_results_summary.py
```

---

## 四、📊 重要文件说明

### 配置文件
| 文件 | 用途 |
|------|------|
| `requirements.txt` | Python依赖包 |
| `specs/*.json` | 实验配置规范 |
| `scripts/env.sh` | 环境变量设置 |
| `Makefile` | 自动化构建 |

### 数据文件
| 目录 | 内容 |
|------|------|
| `benchmark_data_claude4.1/` | WiFi-CSI-Sensing-Benchmark数据 |
| `Data/` | 预处理数据目录 |
| `results/*.json` | 实验结果JSON |
| `paper/figures/*.csv` | 图表数据 |

### 核心脚本
| 脚本 | 功能 |
|------|------|
| `src/train_eval.py` | 主训练循环 |
| `src/models_pinn.py` | PINN模型实现 |
| `src/data_synth.py` | 合成数据生成 |
| `scripts/run_main.sh` | 一键运行脚本 |

---

## 五、🔧 常用命令

### 训练模型
```bash
# 训练Enhanced模型
python src/train_eval.py --model enhanced --epochs 100

# 训练PINN-LSTM
python src/train_eval.py --model pinn_lstm --physics_weight 0.1
```

### 评估模型
```bash
# LOSO评估
python src/evaluate.py --protocol loso --model enhanced

# LORO评估  
python src/evaluate.py --protocol loro --model enhanced
```

### 生成论文
```bash
# 编译主论文
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 六、📈 实验协议说明

### D1: 合成数据验证
- 验证物理引导合成数据的有效性
- 文件: `results/d1/`

### D2: 校准分析
- 模型可信度和校准性能评估
- 文件: `results/d2/`, `scripts/analyze_d2_*.py`

### D3: 跨域评估 (CDAE)
- LOSO: Leave-One-Subject-Out
- LORO: Leave-One-Room-Out
- 文件: `results/d3/`

### D4: Sim2Real迁移 (STEA)
- 少样本学习和零样本迁移
- 文件: `results/d4/sim2real/`

### D5: 消融研究
- 组件贡献度分析
- 文件: `results/d5/`

### D6: 可信AI评估
- 不确定性量化
- 文件: `results/d6/`

---

## 七、🎯 Claude 4.1 新增内容（12月28日更新）

### 新实验模型（4个完整实现）
1. **Exp1: Enhanced Sim2Real Model** - 增强模型+领域自适应
2. **Exp2: Enhanced + PINN Loss** - 增强模型+自适应物理损失
3. **Exp3: PINN LSTM + Causal Attention** - 物理信息LSTM+因果注意力
4. **Exp4: Mamba Efficiency Model** - 线性复杂度状态空间模型

### 新评估协议
1. **CDAE**: Cross-Domain Activity Evaluation - 跨域活动评估
2. **STEA**: Small-Target Environment Adaptation - 小目标环境适应

### 新研究方向（5个完整框架）
1. **多模态融合** - WiFi+视觉+雷达融合（40K字符论文草稿）
2. **联邦学习** - 隐私保护分布式训练（40K字符论文草稿）
3. **神经架构搜索** - 自动化模型设计（40K字符论文草稿）
4. **因果推理** - 可解释性增强（40K字符论文草稿）
5. **持续学习** - 终身学习框架（40K字符论文草稿）

### 基础设施改进
1. **Docker容器化** - 7个服务的完整部署方案
2. **LaTeX文档系统** - 自动验证和编译指南
3. **统一实验框架** - 一键运行所有实验
4. **BibTeX文献库** - 60+篇高影响力参考文献

---

## 八、📝 注意事项

1. **数据路径**: 确保数据集放在正确目录
2. **GPU内存**: Enhanced模型需要至少8GB显存
3. **Python版本**: 推荐使用Python 3.8+
4. **依赖冲突**: 使用虚拟环境避免包冲突

---

## 九、🚀 快速导航（新增文件）

### 最重要的新文件
| 文件路径 | 描述 | 大小 |
|---------|------|------|
| `docs/experiments/thesis/exp1_comprehensive_analysis_claude4.1.tex` | Exp1完整分析（可编译） | 103K字符 |
| `docs/experiments/unified_experiment_runner_claude4.1.py` | 统一实验运行器 | 611行 |
| `docs/experiments/docker/docker-compose.yml` | Docker服务编排 | 7个服务 |
| `docs/experiments/STEP_BY_STEP_GUIDE_claude4.1.md` | 详细操作指南 | 507行 |

### 模型实现文件
| 实验 | 模型文件 | 行数 |
|------|---------|------|
| Exp1 | `exp1_enhanced_sim2real/model_claude4.1.py` | 945行 |
| Exp2 | `exp2_enhanced_pinn_loss/model_claude4.1.py` | 611行 |
| Exp3 | `exp3_pinn_lstm_causal/model_claude4.1.py` | 556行 |
| Exp4 | `exp4_mamba_efficiency/model_claude4.1.py` | 582行 |

### 参考文献文件
| 实验 | BibTeX文件 | 文献数 |
|------|-----------|--------|
| Exp1 | `thesis/exp1_sim2real.bib` | 15篇 |
| Exp2 | `thesis/exp2_pinn.bib` | 15篇 |
| Exp3 | `thesis/exp3_causal.bib` | 15篇 |
| Exp4 | `thesis/exp4_mamba.bib` | 15篇 |

### 工作报告文件
| 文件 | 内容 | 更新时间 |
|------|------|----------|
| `daily/GIT_COMMITS_SUMMARY_claude4.1.md` | 40+提交记录 | 12月28日 |
| `daily/VALUABLE_COMMITS_ANALYSIS_claude4.1.md` | TOP10价值提交 | 12月27日 |
| `daily/FILE_STRUCTURE_MAP_claude4.1.md` | 项目结构图 | 12月27日 |

---

## 九、🔗 相关链接

- GitHub仓库: https://github.com/zhihaozhao/paperA
- 数据集: https://github.com/zhihaozhao/WiFi-CSI-Sensing-Benchmark
- 论文预印本: [待发布]

---

## 十、📧 联系方式

如有问题，请查看:
- `docs/experiments/HOW_TO_RUN_claude4.1.md`
- `docs/daily/work_report_Dec27_claude4.1.md`

---

**文档版本**: v1.0  
**最后更新**: 2024-12-27  
**编制人**: Claude 4.1