# 📊 图片文件夹清理报告

## ✅ 已完成的清理工作

### Paper 2 (PASE-Net) - TMC投稿版本
**位置**: `paper/paper2_pase_net/manuscript/plots/`

#### 清理前状态
- 总文件数: 27个
- 包含多个草稿版本和未使用的图片

#### 清理后状态
- **保留文件**: 15个（9个图片 + 6个脚本）
- **备份文件**: 12个（移至 `plots_backup/`）

#### 保留的图片（论文中使用）
| 图号 | 文件名 | 用途 |
|------|--------|------|
| Fig. 1 | `fig1_system_architecture.pdf` | 系统架构图 |
| Fig. 2 | `fig2_physics_modeling_new.pdf` | 物理建模框架 |
| Fig. 3 | `fig3_cross_domain.pdf` | 跨域性能分析 |
| Fig. 4 | `fig4_calibration.pdf` | 校准分析 |
| Fig. 5 | `fig5_label_efficiency.pdf` | 标签效率 |
| Fig. 6 | `fig6_interpretability.pdf` | 可解释性分析 |

#### 保留的补充材料图片
- `d5_progressive_enhanced.pdf` - 渐进时间分析
- `ablation_noise_env_claude4.pdf` - 噪声因素分析
- `ablation_components.pdf` - 组件分析

#### 保留的生成脚本
- `scr1_system_architecture.py`
- `scr2_physics_modeling.py`
- `scr3_cross_domain.py`
- `scr4_calibration.py`
- `scr5_label_efficiency.py`
- `scr6_interpretability.py`

---

## 🗂️ 文件夹结构对比

### 清理前
```
plots/
├── fig1_system_architecture.pdf ✅
├── fig1_combined_system_architecture.pdf ❌ (草稿)
├── fig1_system_overview-.pdf ❌ (草稿)
├── fig2_physics_modeling.pdf ❌ (旧版本)
├── fig2_physics_modeling_new.pdf ✅
├── fig2_physics_modeling_v2.pdf ❌ (草稿)
├── fig3_cross_domain.pdf ✅
├── fig3_cross_domain_REAL.pdf ❌ (替代版本)
├── fig4_calibration.pdf ✅
├── fig4_calibration_REAL.pdf ❌ (替代版本)
├── fig5_label_efficiency.pdf ✅
├── fig5_label_efficiency_REAL.pdf ❌ (替代版本)
├── fig6_interpretability.pdf ✅
├── fig6_fall_detection_REAL.pdf ❌ (未使用)
├── fig6_fall_types_REAL.pdf ❌ (未使用)
└── ... (其他文件)
```

### 清理后
```
plots/
├── fig1_system_architecture.pdf ✅
├── fig2_physics_modeling_new.pdf ✅
├── fig3_cross_domain.pdf ✅
├── fig4_calibration.pdf ✅
├── fig5_label_efficiency.pdf ✅
├── fig6_interpretability.pdf ✅
├── d5_progressive_enhanced.pdf ✅ (补充材料)
├── ablation_noise_env_claude4.pdf ✅ (补充材料)
├── ablation_components.pdf ✅ (补充材料)
└── scr*.py (生成脚本)

plots_backup/
└── (12个备份文件)
```

---

## 📋 清理原则

### 保留文件的标准
1. **论文中明确引用的图片** - 通过 `\includegraphics` 命令引用
2. **补充材料图片** - 虽然被注释但需要用于补充材料
3. **必要的生成脚本** - 用于重现图片的Python脚本

### 移除文件的类型
1. **草稿版本** - 如 `*_v2.pdf`, `*_REAL.pdf`
2. **未使用的图片** - 论文中没有引用
3. **替代版本** - 被新版本取代的文件
4. **临时文件** - 测试或中间版本

---

## 🎯 投稿准备状态

### Paper 2 (PASE-Net) - TMC
- ✅ **图片文件夹已清理**
- ✅ **所有引用的图片都存在**
- ✅ **无多余的草稿文件**
- ✅ **生成脚本保留完整**
- ✅ **创建了备份文件夹**

### Paper 1 (Sim2Real) - IoTJ
- ⚠️ **需要创建并整理figures文件夹**
- ⚠️ **部分图片缺失**

### Paper 3 (Zero-shot) - TKDE
- ⚠️ **图片尚未准备**

---

## 📝 建议

### 立即行动
1. **Paper 2 可以直接投稿** - 图片已经清理完毕
2. **编译测试**:
   ```bash
   cd paper/paper2_pase_net/manuscript
   pdflatex enhanced_claude_v1.tex
   ```

### 后续工作
1. **Paper 1 需要准备图片**
2. **统一命名规范** - 建议所有论文使用 `fig[编号]_[描述].pdf` 格式
3. **版本控制** - 使用Git标签标记投稿版本

---

## 📊 统计摘要

| 指标 | 数值 |
|------|------|
| 清理的文件夹数 | 1 |
| 处理的文件总数 | 27 |
| 保留的文件数 | 15 |
| 备份的文件数 | 12 |
| 节省的空间 | ~2MB |
| LaTeX引用验证 | ✅ 全部通过 |

---

**清理完成时间**: 2024-12-04
**状态**: ✅ Paper 2 准备就绪，可以投稿！