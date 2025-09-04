# 📋 图片上传检查清单

## ⚠️ 重要提醒
**上传前必须检查.gitignore文件！**

## 🔍 .gitignore检查

### 当前忽略规则
```
plots/                # 忽略所有plots文件夹
paper/*.pdf          # 忽略paper目录下的PDF
```

### 已添加的例外规则
```
!paper/paper2_pase_net/manuscript/plots/*.pdf  # 允许我们的图片
!paper/paper2_pase_net/manuscript/*.tex        # 允许LaTeX文件
!paper/paper2_pase_net/manuscript/*.bib        # 允许bib文件
```

## ✅ 图片上传步骤

### 1. 检查文件是否被忽略
```bash
git check-ignore paper/paper2_pase_net/manuscript/plots/*.pdf
```

### 2. 强制添加被忽略的文件
```bash
git add -f paper/paper2_pase_net/manuscript/plots/*.pdf
```

### 3. 验证文件已添加
```bash
git status
```

## 📊 当前图片状态（已上传）

### 主文图片 (Fig 1-6)
| 文件名 | 状态 | 大小 | Git状态 |
|--------|------|------|---------|
| fig1_system_architecture.pdf | ✅ | 32KB | 已上传 |
| fig2_physics_modeling_new.pdf | ✅ | 56KB | 已上传 |
| fig3_calibration.pdf | ✅ | 48KB | 已上传 |
| fig4_cross_domain.pdf | ✅ | 34KB | 已上传 |
| fig5_label_efficiency.pdf | ✅ | 28KB | 已上传 |
| fig6_interpretability.pdf | ✅ | 250KB | 已上传 |

### 补充材料图片 (S1-S5)
| 文件名 | 状态 | 大小 | Git状态 |
|--------|------|------|---------|
| s1_cross_domain_multisubplot.pdf | ✅ | 46KB | 已上传 |
| s2_label_efficiency_multisubplot.pdf | ✅ | 40KB | 已上传 |
| s3_progressive_temporal.pdf | ✅ | 17KB | 已上传 |
| s4_ablation_noise_env.pdf | ✅ | 48KB | 已上传 |
| s5_ablation_components.pdf | ✅ | 15KB | 已上传 |

## 🛠️ 常见问题解决

### 问题1: 文件被gitignore
**解决方案**: 使用 `git add -f` 强制添加

### 问题2: PDF文件损坏
**解决方案**: 
1. 重新生成PDF
2. 检查文件大小（应该>10KB）
3. 尝试用PDF阅读器打开验证

### 问题3: 文件未显示在Git
**解决方案**:
1. 检查 `git status`
2. 确认文件路径正确
3. 使用 `git add -f` 强制添加

## 📝 最终验证命令

```bash
# 列出所有已跟踪的PDF文件
git ls-files | grep "paper2_pase_net.*\.pdf"

# 应该看到：
# paper/paper2_pase_net/manuscript/plots/fig1_system_architecture.pdf
# paper/paper2_pase_net/manuscript/plots/fig2_physics_modeling_new.pdf
# paper/paper2_pase_net/manuscript/plots/fig3_calibration.pdf
# paper/paper2_pase_net/manuscript/plots/fig4_cross_domain.pdf
# paper/paper2_pase_net/manuscript/plots/fig5_label_efficiency.pdf
# paper/paper2_pase_net/manuscript/plots/fig6_interpretability.pdf
# paper/paper2_pase_net/manuscript/plots/s1_cross_domain_multisubplot.pdf
# paper/paper2_pase_net/manuscript/plots/s2_label_efficiency_multisubplot.pdf
# paper/paper2_pase_net/manuscript/plots/s3_progressive_temporal.pdf
# paper/paper2_pase_net/manuscript/plots/s4_ablation_noise_env.pdf
# paper/paper2_pase_net/manuscript/plots/s5_ablation_components.pdf
```

## 🚀 快速上传脚本

```bash
#!/bin/bash
# 保存为 upload_figures.sh

echo "=== 检查并上传图片文件 ==="

# 1. 更新.gitignore（如果需要）
echo "检查.gitignore..."

# 2. 强制添加所有图片
echo "添加主文图片..."
git add -f paper/paper2_pase_net/manuscript/plots/fig*.pdf

echo "添加补充材料图片..."
git add -f paper/paper2_pase_net/manuscript/plots/s*.pdf

# 3. 添加LaTeX文件
echo "添加LaTeX文件..."
git add -f paper/paper2_pase_net/manuscript/*.tex
git add -f paper/paper2_pase_net/manuscript/*.bib

# 4. 检查状态
echo "当前Git状态："
git status --short

# 5. 提交
echo "准备提交..."
git commit -m "feat: Upload all figures with gitignore override"

# 6. 推送
echo "推送到远程..."
git push origin feat/enhanced-model-and-sweep

echo "=== 完成！==="
```

## ⚠️ 重要提醒

1. **每次上传前检查.gitignore**
2. **使用 `git add -f` 强制添加被忽略的文件**
3. **验证文件大小和完整性**
4. **确认所有文件都在git status中显示**
5. **推送后在GitHub上验证文件存在**

---

**最后更新**: 2024-12-04
**状态**: ✅ 所有图片已成功上传