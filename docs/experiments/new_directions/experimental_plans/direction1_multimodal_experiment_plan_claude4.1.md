# 研究方向1：多模态融合 - 详细实验计划与验收规范

## 一、实验总体设计

### 1.1 研究问题
- **主要问题**: WiFi CSI单模态在复杂环境下准确率不足
- **核心假设**: 多模态融合可显著提升识别准确率和鲁棒性
- **验证目标**: 证明跨模态注意力机制优于传统融合方法

### 1.2 实验时间线
```
Week 1-2: 环境搭建与数据准备
Week 3-4: 基线模型实现与验证
Week 5-6: 多模态融合模型开发
Week 7-8: 对比实验与消融研究
Week 9-10: 结果分析与论文撰写
```

## 二、详细实验计划

### 实验1：单模态基线性能评估

#### 目标
建立各模态独立性能基准

#### 实验设置
```yaml
Datasets:
  - MM-Fi: 40 subjects, 27 activities
  - SenseFi-V: 20 subjects, 10 activities
  - IMU-WiFi: 30 subjects, 12 activities

Models:
  WiFi-Only:
    - Architecture: CNN-LSTM
    - Input: CSI amplitude + phase
    - Parameters: 2.3M
  
  IMU-Only:
    - Architecture: DeepConvLSTM
    - Input: 9-axis sensor data
    - Parameters: 1.8M
  
  Vision-Only:
    - Architecture: TSN (ResNet-50)
    - Input: RGB frames
    - Parameters: 23.5M

Training:
  - Epochs: 200
  - Batch size: 32
  - Learning rate: 1e-3
  - Optimizer: Adam
  - Cross-validation: 5-fold
```

#### 评估指标
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Per-class performance
- Computational cost (FLOPs, latency)

#### 预期结果
| Model | MM-Fi | SenseFi-V | IMU-WiFi |
|-------|-------|-----------|----------|
| WiFi-Only | 83.6% | 81.2% | 85.3% |
| IMU-Only | 79.2% | 76.8% | 82.1% |
| Vision-Only | 87.4% | 89.3% | N/A |

### 实验2：多模态融合方法对比

#### 目标
验证CrossSense相对于传统融合方法的优势

#### 实验设置
```python
fusion_methods = {
    'early_fusion': {
        'strategy': 'concatenation',
        'fusion_point': 'input_level'
    },
    'late_fusion': {
        'strategy': 'average_voting',
        'fusion_point': 'decision_level'
    },
    'intermediate_fusion': {
        'strategy': 'feature_concat',
        'fusion_point': 'fc_layer'
    },
    'crosssense': {
        'strategy': 'cross_attention',
        'fusion_point': 'multi_level'
    }
}

# 对每种方法进行网格搜索
hyperparameter_grid = {
    'fusion_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
    'dropout_rate': [0.1, 0.2, 0.3],
    'attention_heads': [4, 8, 16]
}
```

#### 评估维度
1. **准确率提升**: 相对于最佳单模态
2. **计算效率**: 推理时间、内存占用
3. **融合质量**: 特征相关性分析
4. **可解释性**: 注意力权重可视化

#### 预期结果
| Method | Accuracy | Latency | Memory |
|--------|----------|---------|---------|
| Early Fusion | 90.1% | 12ms | 156MB |
| Late Fusion | 88.7% | 18ms | 248MB |
| Intermediate | 91.3% | 15ms | 187MB |
| **CrossSense** | **96.3%** | **14ms** | **124MB** |

### 实验3：鲁棒性评估（缺失模态）

#### 目标
评估模型在模态缺失情况下的性能退化

#### 实验设置
```python
missing_scenarios = [
    # 单模态缺失
    {'missing': ['wifi'], 'available': ['imu', 'vision']},
    {'missing': ['imu'], 'available': ['wifi', 'vision']},
    {'missing': ['vision'], 'available': ['wifi', 'imu']},
    
    # 双模态缺失
    {'missing': ['wifi', 'imu'], 'available': ['vision']},
    {'missing': ['wifi', 'vision'], 'available': ['imu']},
    {'missing': ['imu', 'vision'], 'available': ['wifi']},
]

# 测试不同dropout率训练的模型
dropout_rates = [0.0, 0.1, 0.2, 0.3]
```

#### 评估指标
- 准确率保持率 (相对于全模态)
- 性能退化曲线
- 模态重要性分析

#### 预期结果
| Missing Modality | w/o Dropout | w/ Dropout (0.3) |
|-----------------|-------------|------------------|
| None | 96.3% | 95.8% |
| WiFi | 45.2% | 85.7% |
| IMU | 52.3% | 91.2% |
| Vision | 48.6% | 87.3% |
| Any Two | 0% | 76.3% |

### 实验4：消融研究

#### 目标
验证各组件的贡献度

#### 实验设置
```python
ablation_configs = [
    {'name': 'Full Model', 'components': ['all']},
    {'name': 'w/o Cross-Attention', 'remove': ['cross_attn']},
    {'name': 'w/o Temporal Align', 'remove': ['temp_align']},
    {'name': 'w/o Modality Dropout', 'remove': ['mod_dropout']},
    {'name': 'w/o Auxiliary Loss', 'remove': ['aux_loss']},
    {'name': 'Fixed Fusion Weights', 'remove': ['adaptive_weights']},
]
```

#### 评估指标
- 各组件对准确率的贡献
- 训练收敛速度
- 模型复杂度变化

#### 预期结果
| Configuration | Accuracy | Δ Accuracy |
|--------------|----------|------------|
| Full Model | 96.3% | - |
| w/o Cross-Attention | 92.1% | -4.2% |
| w/o Temporal Align | 93.5% | -2.8% |
| w/o Modality Dropout | 95.0% | -1.3% |
| w/o Auxiliary Loss | 94.4% | -1.9% |
| Fixed Weights | 93.8% | -2.5% |

### 实验5：计算效率分析

#### 目标
评估模型的部署可行性

#### 测试平台
```yaml
Devices:
  Desktop:
    - GPU: NVIDIA RTX 3090
    - CPU: Intel i9-10900K
    - RAM: 32GB
  
  Laptop:
    - GPU: NVIDIA GTX 1650
    - CPU: Intel i7-9750H
    - RAM: 16GB
  
  Edge:
    - Device: NVIDIA Jetson Nano
    - CPU: Quad-core ARM A57
    - RAM: 4GB
  
  Mobile:
    - Device: Snapdragon 888
    - Framework: TensorFlow Lite
```

#### 评估指标
- 推理延迟 (ms/sample)
- 吞吐量 (samples/sec)
- 内存占用 (MB)
- 能耗 (mJ/inference)

#### 预期结果
| Platform | Latency | Throughput | Memory | Energy |
|----------|---------|------------|---------|---------|
| RTX 3090 | 8.2ms | 122 fps | 124MB | 15mJ |
| GTX 1650 | 21.5ms | 46 fps | 124MB | 28mJ |
| Jetson Nano | 47.3ms | 21 fps | 98MB | 42mJ |
| Mobile (INT8) | 68.5ms | 14 fps | 31MB | 35mJ |

### 实验6：实际场景测试

#### 目标
验证真实环境下的性能

#### 测试场景
```yaml
Environments:
  Home:
    - Rooms: Living room, Bedroom, Kitchen
    - Interference: TV, Microwave, Other WiFi
    - Subjects: 5 family members
  
  Office:
    - Area: Open space, Meeting room
    - Interference: Multiple WiFi APs
    - Subjects: 20 employees
  
  Gym:
    - Equipment: Treadmill, Weights
    - Activities: Exercise routines
    - Subjects: 15 users
```

#### 长期测试
- 连续运行7天
- 监控性能漂移
- 记录失败案例

#### 预期结果
- 家庭环境: 94.2% 准确率
- 办公环境: 91.8% 准确率
- 健身房: 89.3% 准确率
- 7天稳定性: <2% 性能下降

## 三、验收规范 (Acceptance Criteria)

### 3.1 性能指标验收

#### 必须达到 (Must Have)
- [ ] 整体准确率 ≥ 95%
- [ ] 相对最佳单模态提升 ≥ 10%
- [ ] 单模态缺失准确率 ≥ 85%
- [ ] 推理延迟 < 50ms (边缘设备)

#### 应该达到 (Should Have)
- [ ] F1-score ≥ 0.94
- [ ] 双模态缺失准确率 ≥ 75%
- [ ] 内存占用 < 128MB
- [ ] 能耗 < 50mJ/inference

#### 最好达到 (Nice to Have)
- [ ] 支持在线学习
- [ ] 自适应模态选择
- [ ] 零样本泛化能力

### 3.2 技术指标验收

#### 代码质量
- [ ] 代码覆盖率 > 80%
- [ ] 单元测试通过率 100%
- [ ] 文档完整性 100%
- [ ] PEP8/Lint合规

#### 可重现性
- [ ] 随机种子固定
- [ ] Docker容器化
- [ ] 配置文件完整
- [ ] 数据预处理脚本

#### 可扩展性
- [ ] 模块化设计
- [ ] 新模态易集成
- [ ] 超参数可配置
- [ ] 多GPU支持

### 3.3 论文发表验收

#### 实验完整性
- [ ] 至少3个公开数据集
- [ ] 至少5个基线对比
- [ ] 消融研究完整
- [ ] 统计显著性检验

#### 创新性验证
- [ ] 技术创新点 ≥ 3个
- [ ] 性能提升显著
- [ ] 理论分析充分
- [ ] 应用价值明确

## 四、风险管理

### 4.1 技术风险
| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 模态同步困难 | 中 | 高 | 时间戳校准算法 |
| 计算资源不足 | 低 | 中 | 云端GPU租用 |
| 过拟合 | 中 | 中 | 数据增强+正则化 |

### 4.2 数据风险
| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| 数据集不可用 | 低 | 高 | 多数据源备份 |
| 标注错误 | 中 | 中 | 交叉验证 |
| 隐私问题 | 低 | 高 | 数据脱敏处理 |

## 五、交付物清单

### 5.1 代码交付
- [ ] 源代码 (GitHub)
- [ ] 预训练模型
- [ ] 训练脚本
- [ ] 评估脚本
- [ ] Demo应用

### 5.2 文档交付
- [ ] 技术报告
- [ ] API文档
- [ ] 用户手册
- [ ] 部署指南

### 5.3 论文交付
- [ ] 会议论文 (8页)
- [ ] 期刊论文 (12页)
- [ ] 补充材料
- [ ] 演示视频

## 六、成功标准

### 量化标准
1. **准确率**: 96.3% (MM-Fi数据集)
2. **效率**: 21 FPS (Jetson Nano)
3. **鲁棒性**: 85%+ (单模态缺失)
4. **创新性**: 3个技术贡献

### 定性标准
1. **可解释性**: 注意力权重可视化
2. **实用性**: 真实场景可部署
3. **可扩展性**: 易于添加新模态
4. **影响力**: 顶会接收

## 七、时间节点

| 里程碑 | 截止日期 | 交付物 | 验收人 |
|--------|---------|--------|--------|
| M1: 环境搭建 | Week 2 | 开发环境 | 技术负责人 |
| M2: 基线完成 | Week 4 | 基线结果 | 项目经理 |
| M3: 模型开发 | Week 6 | 核心模型 | 首席科学家 |
| M4: 实验完成 | Week 8 | 实验报告 | 评审委员会 |
| M5: 论文提交 | Week 10 | 完整论文 | 会议PC |