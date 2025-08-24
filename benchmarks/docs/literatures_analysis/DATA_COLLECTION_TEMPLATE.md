# 真实数据收集模板

## 使用说明

这个模板用于从PDF论文中手动提取真实的性能数据。请严格按照格式填写，确保数据的准确性和可追溯性。

## 数据收集标准

### 必需字段
- **Paper ID**: 对应的BibTeX引用ID
- **PDF File**: 对应的PDF文件名
- **Performance Metrics**: 从论文实验结果中提取的真实数据
- **Data Source**: 数据在论文中的具体位置（页码、表格、图表）

### 性能指标定义
- **Accuracy**: 检测/识别准确率 (%)
- **Processing Time**: 单次处理时间 (ms)
- **Success Rate**: 成功率 (%)，适用于机器人操作
- **Dataset Size**: 测试数据集大小 (n=?)
- **Environment**: 测试环境（greenhouse, orchard, field, laboratory）

## 数据收集表格

### 高置信度映射 (需要优先填写)

#### 1. Apple Harvesting Papers

| PDF File | BibTeX ID | Accuracy (%) | Processing Time (ms) | Success Rate (%) | Dataset Size | Environment | Data Source | Verified |
|----------|-----------|--------------|---------------------|------------------|--------------|-------------|-------------|----------|
| 1_2016_THEDEVELOPMENTOFMECHANICALAPPLEHARVESTINGTECHNOLOGYAREVIEW.pdf | li2016characterizing | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| Fruit detection and segmentation for apple harvesting using visual sensor in orchards.pdf | jia2020detection | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| Real-Time Fruit Recognition and Grasping Estimation for Robotic Apple Harvesting.pdf | jia2020apple | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| Faster R–CNN–based apple detection in dense-foliage fruiting-wall trees using RGB and depth features for robotic harvesting.pdf | fu2020faster | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| A novel green apple segmentation algorithm based on ensemble U-Net under complex orchard environment.pdf | li2021novel | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |

#### 2. Strawberry Harvesting Papers

| PDF File | BibTeX ID | Accuracy (%) | Processing Time (ms) | Success Rate (%) | Dataset Size | Environment | Data Source | Verified |
|----------|-----------|--------------|---------------------|------------------|--------------|-------------|-------------|----------|
| An autonomous strawberry-harvesting robot Design, development, integration, and field evaluation.pdf | yu2019fruit | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| Journal of Field Robotics - 2019 - Xiong - An autonomous strawberry‐harvesting robot Design development integration and.pdf | xiong2020autonomous | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |
| Development and field evaluation of a strawberry harvesting robot with a cable-driven gripper.pdf | xiong2019development | ___ | ___ | ___ | n=___ | ___ | Page __, Table __ | ☐ |

#### 3. Motion Control Papers

| PDF File | BibTeX ID | Success Rate (%) | Processing Time (ms) | Adaptability | Dataset Size | Environment | Data Source | Verified |
|----------|-----------|------------------|---------------------|--------------|--------------|-------------|-------------|----------|
| Analysis of a motion planning problem for sweet-pepper harvesting in a dense obstacle environment.pdf | bac2016analysis | ___ | ___ | ___/100 | n=___ | ___ | Page __, Table __ | ☐ |
| Vision-based control of robotic manipulator for citrus harvesting.pdf | mehta2014vision | ___ | ___ | ___/100 | n=___ | ___ | Page __, Table __ | ☐ |
| Robotic kiwifruit harvesting using machine vision, convolutional neural networks, and robotic arms.pdf | williams2019robotic | ___ | ___ | ___/100 | n=___ | ___ | Page __, Table __ | ☐ |

## 数据验证检查清单

### 填写完成后，请检查：
- ☐ 所有数值都从PDF原文中准确提取
- ☐ 数据来源（页码、表格）已标注
- ☐ 单位统一（%用百分比，时间用毫秒）
- ☐ 环境类型准确分类
- ☐ 数据集大小包含具体数字
- ☐ 已验证数据的合理性

### 质量控制：
- ☐ 准确率不应超过100%
- ☐ 处理时间应为正数
- ☐ 数据集大小应合理（通常n>50）
- ☐ 环境分类应一致

## 数据提交格式

完成填写后，请将数据保存为：
- `REAL_PERFORMANCE_DATA_[日期].json`
- 或直接更新到现有的映射文件中

## 备注

- 如果论文中没有某项数据，请填写"N/A"
- 如果数据范围，请记录范围（如85-92%）
- 如果有多个实验结果，请记录最佳结果
- 所有数据必须可以在PDF中找到对应位置

---
*模板创建日期: 2024*
*状态: 等待手动数据填写*