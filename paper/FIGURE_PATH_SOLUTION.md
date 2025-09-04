# 图片路径管理方案

## 🎯 核心原则
**保持论文原有的文件夹结构，不改变LaTeX中的引用路径**

## 📁 当前结构（推荐保持）

### Paper 1 (Sim2Real)
```
paper/paper1_sim2real/manuscript/
├── main.tex (引用: figures/fig*.pdf)
├── figures/  ← 需要创建并放置图片
│   ├── fig1_system_architecture.pdf
│   ├── fig2_physics_guided_framework.pdf
│   └── ...
└── refs.bib
```

### Paper 2 (PASE-Net) ✅ 
```
paper/paper2_pase_net/manuscript/
├── enhanced_claude_v1.tex (引用: plots/fig*.pdf)
├── plots/  ← 已存在，保持不变
│   ├── fig1_system_architecture.pdf
│   ├── fig2_physics_modeling_new.pdf
│   └── ...
└── enhanced_refs.bib
```

### Paper 3 (Zero-shot)
```
paper/paper3_zero_shot/manuscript/
├── zeroshot.tex
├── figures/  ← 根据需要创建
└── zero_refs.bib
```

## 🔧 实施步骤

### 1. Paper 2 (PASE-Net) - 无需改动
- ✅ plots文件夹已在正确位置
- ✅ 所有图片都已存在
- ✅ LaTeX可以正常编译

### 2. Paper 1 (Sim2Real) - 需要创建figures文件夹
```bash
# 创建figures文件夹
mkdir -p paper/paper1_sim2real/manuscript/figures

# 从原始位置复制或生成图片
cp paper/figures/*.pdf paper/paper1_sim2real/manuscript/figures/
```

### 3. 图片生成脚本管理

#### 方案A：脚本与图片同位置（推荐）
```
manuscript/
├── plots/
│   ├── scr1_system_architecture.py  # 生成脚本
│   └── fig1_system_architecture.pdf  # 输出图片
```

#### 方案B：脚本集中管理
```
supplementary/scripts/figure_generation/
├── generate_all_figures.py
└── scr*.py  # 所有生成脚本

# 脚本输出路径指向：
# ../../manuscript/plots/fig*.pdf
```

## 📝 脚本更新示例

### 原始脚本（保持在manuscript/plots/）
```python
# scr2_physics_modeling.py
if __name__ == "__main__":
    fig = create_combined_figure()
    output_path = "fig2_physics_modeling_new.pdf"  # 当前目录
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

### 集中管理脚本（在supplementary/scripts/）
```python
# scr2_physics_modeling.py
if __name__ == "__main__":
    fig = create_combined_figure()
    # 输出到manuscript/plots/
    output_path = "../../manuscript/plots/fig2_physics_modeling_new.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

## ✅ 优势

1. **无需修改LaTeX文件** - 保持原有引用路径
2. **编译无障碍** - pdflatex可以直接找到图片
3. **版本控制友好** - 图片与论文在同一目录
4. **便于提交** - 打包时结构清晰

## 🚀 立即行动

### For Paper 2 (PASE-Net) - TMC提交
```bash
cd paper/paper2_pase_net/manuscript
pdflatex enhanced_claude_v1.tex
# 图片路径正确，可以直接编译
```

### For Paper 1 (Sim2Real) - 需要先设置
```bash
# 1. 创建figures文件夹
mkdir -p paper/paper1_sim2real/manuscript/figures

# 2. 复制或生成图片到figures/

# 3. 编译
cd paper/paper1_sim2real/manuscript
pdflatex main.tex
```

## 📊 状态总结

| Paper | LaTeX引用 | 实际位置 | 状态 | 行动 |
|-------|----------|---------|------|------|
| Paper 2 (PASE-Net) | plots/ | ✅ plots/ | ✅ Ready | 无需改动 |
| Paper 1 (Sim2Real) | figures/ | ❌ 不存在 | 需要修复 | 创建并放置图片 |
| Paper 3 (Zero-shot) | TBD | TBD | 待定 | 根据需要设置 |

## 🎯 建议

1. **保持Paper 2现状** - 已经正确配置
2. **修复Paper 1** - 创建figures文件夹
3. **统一命名约定**：
   - 使用`figures/`作为默认图片文件夹
   - 或使用`plots/`（如Paper 2）
   - 保持一致性

## 📝 备注

- 原始的`paper/enhanced/`文件夹可以保留作为备份
- 新的图片生成脚本可以放在supplementary中，但输出应该指向manuscript的图片文件夹
- 提交期刊时，只需要manuscript文件夹的内容