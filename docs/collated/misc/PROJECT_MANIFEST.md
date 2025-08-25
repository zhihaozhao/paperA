# 📁 Project Manifest - WiFi CSI Physics-Guided Sim2Real HAR

## 🎯 **项目概述**

**论文标题**: Physics-Guided Synthetic WiFi CSI Data Generation for Trustworthy Human Activity Recognition: A Sim2Real Approach

**核心贡献**:
1. 物理引导合成数据生成器
2. 系统性Sim2Real验证
3. 少样本学习效率分析
4. Trustworthy评估协议

**当前状态**: D2实验完成(540配置)，准备Sim2Real实验

---

## 📚 **核心文档目录**

### **🔬 实验设计与验证**
| 文档 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **D2实验指南** | `docs/D2_Experiment_Guide.md` | D2协议完整说明，540配置实验设计 | ✅ |
| **D2验收标准** | `docs/D2_Experiment_Summary.md` | 验收参数、快速执行命令 | ✅ |
| **D2验证脚本** | `scripts/validate_d2_acceptance.py` | 自动化结果验证 | ✅ |

### **📊 SenseFi Benchmark集成**
| 文档 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **Benchmark分析** | `benchmarks/WiFi_CSI_Sensing_Benchmark_Analysis.md` | SenseFi详细分析，集成价值 | ✅ |
| **集成策略** | `docs/Selective_Benchmark_Integration_Strategy.md` | 选择性集成策略 | ✅ |
| **设置指南** | `docs/SenseFi_Benchmark_Setup_Guide.md` | 环境搭建与数据集获取 | ✅ |
| **快速开始** | `scripts/benchmark_quick_start.md` | 5分钟设置指南 | ✅ |

### **💡 核心原则与方法论**
| 文档 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **生成原则** | `docs/Generation_Principles.md` | 核心研究原则：基于真实材料 | ✅ |
| **环境访问** | `references/Windows_to_Remote_Workspace_Guide.md` | Windows→Linux环境切换 | ✅ |

### **📖 文献分析与引用**
| 文档 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **文献分析** | `references/WiFi_CSI_HAR_Literature_Analysis_2020-2024.md` | 9篇核心论文详细分析 | ✅ |
| **对比表格** | `references/Baseline_Comparison_Table.md` | LaTeX表格模板 | ✅ |
| **搜索总结** | `references/Literature_Search_Summary.md` | 文献搜索过程记录 | ✅ |
| **论文链接** | `references/literature_links.md` | 直接arXiv下载链接 | ✅ |
| **实验分析** | `references/Current_Experiment_Analysis.md` | 当前实验vs文献对比 | ✅ |

---

## 🔧 **核心代码与脚本**

### **🧪 实验执行脚本**
| 脚本 | 路径 | 功能 | 状态 |
|------|------|------|------|
| **D2执行** | `scripts/run_sweep_from_json.py` | D2协议实验执行主脚本 | ✅ |
| **D2配置** | `scripts/d2_spec.json` | 540配置参数(4模型×5种子×3×3×3) | ✅ |
| **优化配置** | `scripts/d2_spec_optimized.json` | 小规模测试配置 | ✅ |
| **数据预生成** | `scripts/pregenerate_d2_datasets.py` | D2数据集预生成脚本 | ✅ |
| **D3+D4总控(Win)** | `scripts/run_d3_d4_windows.bat` | Windows一键运行D3/D4 | ✅ |
| **D3 LORO运行(Win)** | `scripts/run_d3_loro.bat` | D3跨房间实验 | ✅ |
| **D3 LOSO运行(Win)** | `scripts/run_d3_loso.bat` | D3跨被试实验 | ✅ |
| **D4 Sim2Real(Win)** | `scripts/run_d4_loro.bat` | D4标签效率扫 | ✅ |

### **🔄 Sim2Real实验框架**
| 脚本 | 路径 | 功能 | 状态 |
|------|------|------|------|
| **数据适配器** | `docs/Optimized_Benchmark_Integration_Plan.md` | 真实数据适配器(Python代码) | ✅ |
| **Sim2Real实验** | `scripts/optimized_sim2real_experiments.py` | 优化的Sim2Real实验框架 | ✅ |
| **Benchmark集成** | `scripts/integrate_wifi_csi_benchmark.py` | 自动化benchmark集成工具 | ✅ |

### **⚙️ 核心模块**
| 模块 | 路径 | 功能 | 状态 |
|------|------|------|------|
| **训练评估** | `src/train_eval.py` | 主训练循环、模型构建、评估 | ✅ |
| **跨域训练** | `src/train_cross_domain.py` | LOSO/LORO/Sim2Real 入口 | ✅ |
| **数据生成** | `src/data_synth.py` | 物理引导合成数据生成器 | ✅ |
| **真实数据** | `src/data_real.py` | 基准数据加载/标准化/LOSO/LORO | ✅ |
| **数据缓存** | `src/data_cache.py` | 多级缓存系统(内存+磁盘) | ✅ |
| **模型定义** | `src/models.py` | Enhanced, CNN, BiLSTM等模型 | ✅ |
| **推理脚本** | `src/infer.py` | 模型推理与测试 | ✅ |

### **📈 分析工具**
| 脚本 | 路径 | 功能 | 状态 |
|------|------|------|------|
| **重叠回归** | `src/overlap_regression.py` | 类重叠与误差相关性分析 | ✅ |
| **结果验证** | `scripts/validate_d2_acceptance.py` | D2结果自动验证 | ✅ |
| **D3结果验证** | `scripts/validate_d3_acceptance.py` | D3 LOSO/LORO 验收脚本 | ✅ |
| **数据预生成** | `scripts/pregenerate_datasets.py` | 通用数据集预生成 | ✅ |

---

## 📝 **论文文档**

### **📄 主要论文文件**
| 文件 | 路径 | 内容 | 状态 |
|------|------|------|------|
| **论文正文** | `paper/main.tex` | LaTeX主文件，已更新结构和引用 | ✅ |
| **参考文献** | `paper/refs.bib` | 21个权威引用，包含SenseFi等 | ✅ |

### **📊 论文优化文档**
| 文档 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **框架优化** | `references/Optimized_Paper_Framework.md` | 论文结构优化总结 | ✅ |
| **集成总结** | `docs/WiFi_CSI_Benchmark_Integration_Summary.md` | Benchmark集成效果 | ✅ |

---

## ⚙️ **配置与环境文件**

### **🌐 环境配置**
| 文件 | 路径 | 功能 | 状态 |
|------|------|------|------|
| **Git忽略** | `.gitignore` | 排除大文件、缓存、benchmark | ✅ |
| **依赖说明** | `requirements.txt` | Python依赖包(如存在) | - |
| **环境指南** | `docs/SenseFi_Benchmark_Setup_Guide.md` | 完整环境搭建 | ✅ |

### **📁 目录结构**
```
项目根目录/
├── src/                    # 核心源代码
├── scripts/                # 实验执行脚本
├── docs/                   # 文档集合
├── references/             # 文献资料
├── benchmarks/             # Benchmark集成
├── paper/                  # 论文LaTeX文件
├── results/                # 实验结果(待添加)
├── plots/                  # 图表输出(待添加)
├── cache/                  # 数据缓存(被忽略)
└── logs/                   # 日志文件(待添加)
```

---

## 🚀 **Git管理策略**

### **📋 分支结构**
| 分支 | 用途 | 状态 |
|------|------|------|
| **main** | 主分支，稳定版本 | 🟢 |
| **feat/enhanced-model-and-sweep** | 当前开发分支 | 🟡 |
| **experiment/d2-results** | D2实验结果专用分支 | 🟡 待创建 |
| **experiment/sim2real-results** | Sim2Real结果分支 | ⏳ 未来 |

### **🏷️ 标签计划**
| 标签 | 描述 | 状态 |
|------|------|------|
| **v1.0-d2-complete** | D2实验完成里程碑 | ⏳ 待创建 |
| **v1.1-sim2real-complete** | Sim2Real实验完成 | ⏳ 未来 |
| **v2.0-paper-submission** | 论文提交版本 | ⏳ 未来 |

### **📋 管理命令**
详见: `docs/Git_Management_Commands.md`

---

## 🎯 **当前任务状态**

### **✅ 已完成任务**
- [x] D2实验设计与执行(540配置)
- [x] 物理引导数据生成器
- [x] 多级数据缓存系统
- [x] SenseFi benchmark分析与集成策略
- [x] 论文结构优化与引用更新
- [x] 文献综述(9篇核心论文)
- [x] Git管理策略设计

### **🟡 进行中任务**
- [ ] 论文引用更新Git提交 (遇到技术问题)
- [ ] D2实验结果从GPU服务器传输
- [ ] 实验结果Git分支管理

### **⏳ 待完成任务**
- [ ] Sim2Real benchmark实验
- [ ] 少样本学习效率分析
- [ ] 论文图表生成
- [ ] 最终论文完善
- [ ] 期刊投稿准备

---

## 🔧 **技术栈总结**

### **🐍 Python核心**
- PyTorch: 深度学习框架
- NumPy/SciPy: 数值计算
- Matplotlib: 图表生成
- Pandas: 数据分析
- Pickle: 数据序列化

### **📊 实验工具**
- JSON: 配置管理
- MD5: 缓存键生成
- Logging: 实验记录
- Bootstrap: 统计验证

### **📝 文档工具**
- LaTeX: 论文撰写
- Markdown: 文档系统
- BibTeX: 引用管理

### **🔧 开发工具**
- Git: 版本控制
- Python argparse: 命令行接口
- Cache系统: 性能优化

---

## 📞 **快速导航**

### **🆘 紧急问题解决**
1. **Git问题**: `docs/Git_Management_Commands.md` → "紧急情况处理"
2. **实验失败**: `docs/D2_Experiment_Guide.md` → "故障排除"
3. **环境问题**: `docs/SenseFi_Benchmark_Setup_Guide.md`

### **📋 常用命令**
```bash
# D2实验
python scripts/run_sweep_from_json.py --spec scripts/d2_spec.json

# 验证结果  
python scripts/validate_d2_acceptance.py results/d2/

# Git管理
git checkout -b experiment/d2-results
git tag -a v1.0-d2-complete -m "D2 Complete"
```

### **📖 文档索引**
- **实验**: `docs/D2_*`
- **Benchmark**: `benchmarks/` + `docs/SenseFi_*`
- **文献**: `references/`
- **论文**: `paper/`
- **Git**: `docs/Git_*`

---

## 🎉 **项目里程碑**

1. **🏗️ 框架搭建** (已完成) - 物理引导生成器 + 缓存系统
2. **🧪 D2验证** (已完成) - 540配置系统验证
3. **📚 文献调研** (已完成) - 9篇核心论文分析
4. **🔗 Benchmark集成** (已完成) - SenseFi集成策略
5. **📝 论文优化** (进行中) - 结构重构 + 引用更新
6. **🔄 Sim2Real实验** (即将开始) - 跨域验证
7. **🎯 论文投稿** (未来) - TMC/IoTJ提交

**当前位置: 里程碑5 → 里程碑6 过渡期**

---

*最后更新: 2025-01-16*  
*维护者: AI Assistant*  
*版本: v1.0*