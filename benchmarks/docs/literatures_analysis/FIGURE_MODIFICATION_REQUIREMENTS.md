# Figure 4、9、10 修改要求详细分析

## 总体要求
1. **补充完整论文列表**: 所有提到的数量（如"25 papers"）必须列出完整的论文清单
2. **添加完整引用信息**: 每篇论文都需要添加`\cite{}`信息和完整标题
3. **确保数据完整性**: 所有数据必须来源于`prisma_data.csv`，零编造

## Figure 4: Vision Algorithm Performance Meta-Analysis

### 当前问题
- ✅ 已有46篇论文的总体统计数据
- ❌ 缺少YOLO家族16篇论文的完整列表
- ❌ 缺少R-CNN家族7篇论文的完整列表  
- ❌ 缺少Hybrid方法17篇论文的完整列表
- ❌ 缺少Traditional方法16篇论文的完整列表

### 需要补充的内容
1. **YOLO家族16篇论文完整列表**:
   - 每篇论文需要: `\cite{author_year}` + 完整标题 + 性能指标
   - 按子类分组: YOLOv3(12), YOLOv4(8), YOLOv5(7), YOLOv8+(8)
   
2. **R-CNN家族7篇论文完整列表**:
   - Faster R-CNN: 10篇 → 需要完整列表
   - Mask R-CNN: 6篇 → 需要完整列表
   - Others: 2篇 → 需要完整列表

3. **性能分类详细数据**:
   - Fast High-Accuracy (9篇) → 需要9篇论文完整信息
   - Fast Moderate-Accuracy (3篇) → 需要3篇论文完整信息
   - Slow High-Accuracy (13篇) → 需要13篇论文完整信息
   - Slow Moderate-Accuracy (21篇) → 需要21篇论文完整信息

## Figure 9: Robotics Control Performance Meta-Analysis

### 当前问题
- ✅ 已有总体框架和5篇核心研究
- ❌ **Critical**: "Deep Reinforcement Learning: 25 papers" → 需要25篇论文完整列表
- ❌ **Critical**: "Classical Geometric Methods: 28 papers" → 需要28篇论文完整列表
- ❌ 缺少Vision-Guided Methods 15篇论文详细信息
- ❌ 缺少Hybrid Systems 9篇论文详细信息

### 需要补充的内容
1. **Deep RL 25篇论文完整列表** (最重要):
   - DDPG: 8篇 → 需要8篇论文的`\cite{}`和标题
   - A3C: 6篇 → 需要6篇论文的`\cite{}`和标题  
   - PPO: 5篇 → 需要5篇论文的`\cite{}`和标题
   - SAC: 4篇 → 需要4篇论文的`\cite{}`和标题
   - Others: 2篇 → 需要2篇论文的`\cite{}`和标题

2. **Classical Geometric Methods 28篇论文完整列表**:
   - RRT*: 12篇 → 需要完整列表
   - A*: 8篇 → 需要完整列表
   - Bi-RRT: 5篇 → 需要完整列表
   - Others: 3篇 → 需要完整列表

3. **环境性能分析详细数据**:
   - Laboratory/Controlled (15篇) → 需要论文列表
   - Greenhouse (24篇) → 需要论文列表  
   - Field/Orchard (38篇) → 需要论文列表

## Figure 10: Critical Analysis Data

### 当前问题
- ✅ 已有6篇TRL核心研究
- ❌ 缺少Critical Severity Problems (8篇) 的完整引用
- ❌ 缺少High Severity Problems (12篇) 的完整引用
- ❌ TRL评估中的支持研究数量需要详细论文列表

### 需要补充的内容
1. **Critical Severity Problems 8篇**:
   - 每篇需要完整的`\cite{}`信息和问题描述
   
2. **High Severity Problems 12篇**:
   - 每篇需要完整的`\cite{}`信息和限制因素描述

3. **TRL支持研究详细列表**:
   - Computer Vision: 12 studies → 需要12篇论文列表
   - Motion Planning: 10 studies → 需要10篇论文列表
   - End-Effector: 8 studies → 需要8篇论文列表
   - AI/ML Integration: 14 studies → 需要14篇论文列表

## 数据来源要求

### 必须遵循的原则
1. **只能使用prisma_data.csv中的论文**
2. **每篇论文必须有完整的bibtex key**
3. **所有性能数据必须来源于真实论文**
4. **不能编造任何论文或数据**

### 数据提取策略
1. **从prisma_data.csv提取**:
   - 搜索相关算法关键词
   - 匹配论文标题和摘要
   - 提取真实性能指标

2. **引用格式要求**:
   ```
   \cite{sa2016deepfruits} - DeepFruits: A Fruit Detection System Using Deep Neural Networks
   ```

3. **性能数据格式**:
   ```
   Sa et al. (2016) \cite{sa2016deepfruits} - R-CNN DeepFruits: 84.8% accuracy, 393ms, n=450
   ```

## 修改优先级

### 高优先级 (必须完成)
1. **Figure 9: Deep RL 25篇论文完整列表** - 这是用户特别强调的
2. **Figure 4: 算法家族完整论文列表** - YOLO(16), R-CNN(7), Hybrid(17), Traditional(16)
3. **所有论文的\cite{}信息补充**

### 中优先级 (重要)
1. **Figure 10: Critical和High severity问题论文列表**
2. **环境性能分析的详细论文分组**

### 低优先级 (可选)
1. **图表可视化优化**
2. **LaTeX格式微调**

## 执行计划

1. **Phase 1**: 从prisma_data.csv提取所有相关论文
2. **Phase 2**: 按算法家族和问题类型分类整理  
3. **Phase 3**: 补充完整的引用信息和标题
4. **Phase 4**: 验证数据完整性和准确性
5. **Phase 5**: 更新所有模板文件
6. **Phase 6**: 提交到git服务器

---
**创建时间**: 2024-08-25
**数据来源**: benchmarks/docs/prisma_data.csv (只读)
**学术诚信**: 严格遵守 - 零编造数据