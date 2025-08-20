# WiFi-CSI项目多仓库迁移方案

## 目标架构

### 1. 主仓库：WiFi-CSI-Sensing-Core
- **用途**：核心算法、实验脚本、开发工具
- **保留内容**：
  - `src/` - 算法实现
  - `scripts/` - 实验和数据处理脚本
  - `eval/` - 评估工具
  - `benchmarks/` - 基准测试框架
  - 配置文件：`requirements.txt`, `env.yml`, `Makefile`, `sitecustomize.py`
  - 文档：`README.md`, `doc/`
  - Git配置：`.gitignore`

### 2. 实验结果仓库：WiFi-CSI-Sensing-Results
- **用途**：存储实验数据、日志、生成图表
- **迁移内容**：
  - `results/`, `results_cpu/`, `results_gpu/`
  - `logs/`
  - `cache/`
  - `plots/`, `figs/`, `tables/`
- **特点**：大文件友好，使用Git LFS

### 3. 期刊论文仓库：WiFi-CSI-Journal-Paper
- **用途**：期刊投稿相关文件
- **迁移内容**：
  - `paper/` 全部内容
  - `references/` 参考文献PDF
- **特点**：协作友好，审稿轮次版本控制

### 4. 博士论文仓库：WiFi-CSI-PhD-Thesis
- **用途**：博士学位论文
- **迁移内容**：
  - `论文/` 全部内容
- **特点**：独立章节管理

## 实施步骤

### 阶段1：创建新仓库结构
```bash
# 1. 创建实验结果仓库
mkdir ../WiFi-CSI-Sensing-Results
cd ../WiFi-CSI-Sensing-Results
git init
git lfs track "*.json" "*.npz" "*.pkl" "*.log"

# 2. 创建期刊论文仓库  
mkdir ../WiFi-CSI-Journal-Paper
cd ../WiFi-CSI-Journal-Paper
git init

# 3. 创建博士论文仓库
mkdir ../WiFi-CSI-PhD-Thesis  
cd ../WiFi-CSI-PhD-Thesis
git init
```

### 阶段2：迁移文件内容
```bash
# 从主仓库复制文件到对应仓库
cp -r /workspace/results* ../WiFi-CSI-Sensing-Results/
cp -r /workspace/logs ../WiFi-CSI-Sensing-Results/
cp -r /workspace/plots ../WiFi-CSI-Sensing-Results/
cp -r /workspace/figs ../WiFi-CSI-Sensing-Results/
cp -r /workspace/tables ../WiFi-CSI-Sensing-Results/
cp -r /workspace/cache ../WiFi-CSI-Sensing-Results/

cp -r /workspace/paper ../WiFi-CSI-Journal-Paper/
cp -r /workspace/references ../WiFi-CSI-Journal-Paper/

cp -r /workspace/论文 ../WiFi-CSI-PhD-Thesis/
```

### 阶段3：清理主仓库
```bash
# 从主仓库移除已迁移的目录
git rm -r results/ results_cpu/ results_gpu/
git rm -r logs/ plots/ figs/ tables/ cache/
git rm -r paper/ references/ 论文/
```

### 阶段4：建立仓库关联（可选）
可通过以下方式关联：
1. **Git Submodules**：在主仓库中添加子模块
2. **文档说明**：在README中说明相关仓库
3. **脚本自动化**：创建脚本自动同步/拉取

## 优势分析

### 1. 避免冲突耦合
- 不同类型文件独立版本控制
- 分支操作不会影响其他内容
- 减少merge conflict

### 2. 协作友好
- 论文仓库可以独立分享给导师/合作者
- 实验结果仓库可以选择性分享
- 代码仓库保持轻量

### 3. 存储优化
- 主仓库只包含源代码，clone速度快
- 大文件集中在results仓库，可用Git LFS
- 按需拉取，节省空间

### 4. 权限管理
- 不同仓库可以设置不同访问权限
- 敏感数据和公开代码分离

## .gitignore更新

主仓库的.gitignore需要移除以下规则（因为对应目录已迁移）：
```gitignore
# 移除这些行（目录已迁移到其他仓库）
# results/
# results_cpu/  
# plots/
# tables/
# logs/
# figs/
# chat/
# cache/
```

## 注意事项

1. **备份重要**：迁移前务必备份当前完整仓库
2. **分步执行**：逐个仓库迁移，确保每步正确
3. **测试验证**：迁移后测试各仓库功能
4. **文档更新**：更新README和相关文档
5. **团队通知**：如有协作者，提前通知变更

## 回滚方案

如果迁移过程中出现问题，可通过以下方式回滚：
```bash
# 恢复主仓库到迁移前状态
git reset --hard <migration_start_commit>
```

保留完整备份直到确认迁移成功。