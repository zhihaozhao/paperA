#!/usr/bin/env python3
"""
本地图片生成脚本 - 支持路径配置的版本
用户可以在本地轻松生成图片，无需修改代码
"""

import matplotlib
matplotlib.use('Agg')  # 使用无头模式
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# 配置区域 - 用户可以根据需要修改
# =============================================================================

# 输出路径配置
OUTPUT_DIR = "."  # 当前目录，用户可修改为自己的路径
FIGURE_PREFIX = "figure"  # 图片文件名前缀

# 图片格式配置
SAVE_FORMATS = ['pdf', 'png']  # 保存的格式
DPI = 300  # 图片分辨率
FIGSIZE_4 = (16, 12)  # Figure 4尺寸
FIGSIZE_9 = (12, 9)   # Figure 9尺寸  
FIGSIZE_10 = (16, 12) # Figure 10尺寸

# =============================================================================
# 辅助函数
# =============================================================================

def save_figure(fig, filename):
    """保存图片到指定格式和路径"""
    for fmt in SAVE_FORMATS:
        filepath = os.path.join(OUTPUT_DIR, f"{FIGURE_PREFIX}{filename}.{fmt}")
        fig.savefig(filepath, bbox_inches='tight', dpi=DPI, format=fmt)
        print(f"✅ 保存: {filepath}")

# =============================================================================
# Figure 4: Vision Model Performance Meta-Analysis
# =============================================================================

def create_figure4_vision_analysis():
    """创建Figure 4: 视觉模型性能元分析"""
    print("🎨 生成Figure 4: 视觉模型性能元分析...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGSIZE_4)
    fig.suptitle('Vision Model Performance Meta-Analysis for Fruit Harvesting (56 Studies)', fontsize=16, fontweight='bold')
    
    # (a) Algorithm Family Performance Distribution
    algorithms = ['R-CNN\nFamily', 'YOLO\nSeries', 'CNN\nClassifiers', 'Hybrid\nMethods', 'Traditional\nMethods']
    accuracy = [88.9, 90.8, 89.2, 87.1, 89.7]
    studies_count = [12, 18, 8, 6, 12]
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    bars = ax1.bar(algorithms, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Accuracy (%)')
    ax1.set_title('(a) Algorithm Family Performance Distribution')
    ax1.set_ylim(80, 95)
    ax1.grid(True, alpha=0.3)
    
    # 添加研究数量标注
    for bar, count in zip(bars, studies_count):
        height = bar.get_height()
        ax1.annotate(f'{count} studies', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # (b) Recent Model Achievements Timeline
    years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    rcnn_perf = np.array([84.8, 86.2, 87.5, 88.1, 90.7, 89.3, 89.8, 87.8, 88.5])
    yolo_perf = np.array([82.1, 84.5, 87.2, 89.1, 91.2, 88.7, 91.5, 90.3, 92.1])
    
    ax2.plot(years, rcnn_perf, 'o-', label='R-CNN Family', linewidth=2, markersize=6, color='#E74C3C')
    ax2.plot(years, yolo_perf, 's-', label='YOLO Series', linewidth=2, markersize=6, color='#3498DB')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Peak Accuracy (%)')
    ax2.set_title('(b) Recent Model Achievements Timeline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(80, 95)
    
    # (c) Real-time Processing Capability Analysis - 修正x轴范围
    accuracy_data = [90.7, 91.2, 89.8, 88.7, 92.1, 91.5, 87.8]  
    speed_data = [58, 84, 92, 95, 71, 83, 94]  
    algorithm_labels = ['Faster R-CNN', 'YOLOv4', 'YOLOv5', 'YOLO Custom', 'YOLOv8', 'YOLOv9', 'Mask R-CNN']
    
    # 高对比度颜色数组
    high_contrast_colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8800', '#AA00AA', '#00AAAA', '#AAAA00']
    scatter = ax3.scatter(speed_data, accuracy_data, s=120, c=high_contrast_colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    # 添加算法标签
    for i, label in enumerate(algorithm_labels):
        ax3.annotate(label, (speed_data[i], accuracy_data[i]), xytext=(5, 5), 
                    textcoords="offset points", fontsize=8, ha='left', va='bottom')
    
    ax3.set_xlabel('Processing Time (ms)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('(c) Real-time Processing Capability Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 修正x轴范围：数据集中在40-120ms范围
    ax3.set_ylim(70, 95)
    ax3.set_xlim(40, 120)  # 更合理的x轴范围
    
    # 添加理想区域标注
    ax3.axhline(y=90, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2, label='Target Accuracy')
    ax3.axvline(x=100, color='darkorange', linestyle='--', alpha=0.7, linewidth=2, label='Real-time Threshold')
    ax3.legend(fontsize=8)
    
    # (d) Environmental Robustness Comparison
    environments = ['Laboratory\n(Controlled)', 'Greenhouse\n(Semi-structured)', 'Orchard\n(Natural)', 'Field\n(Unstructured)']
    rcnn_env = [92.5, 89.2, 85.7, 81.3]
    yolo_env = [91.8, 90.1, 87.4, 84.2]
    
    x = np.arange(len(environments))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, rcnn_env, width, label='R-CNN Family', color='#E74C3C', alpha=0.8)
    bars2 = ax4.bar(x + width/2, yolo_env, width, label='YOLO Series', color='#3498DB', alpha=0.8)
    
    ax4.set_xlabel('Environment Type')
    ax4.set_ylabel('Average Accuracy (%)')
    ax4.set_title('(d) Environmental Robustness Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(environments)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(75, 95)
    
    plt.tight_layout()
    save_figure(fig, "4_meta_analysis")
    plt.close()
    print("✅ Figure 4 生成完成")

# =============================================================================
# Figure 9: Robot Motion Control Performance Meta-Analysis
# =============================================================================

def create_figure9_robotics_analysis():
    """创建Figure 9: 机器人运动控制性能元分析"""
    print("🎨 生成Figure 9: 机器人运动控制性能元分析...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGSIZE_9)
    fig.suptitle('Robot Motion Control Performance Meta-Analysis (60 Studies)', fontsize=14, fontweight='bold')
    
    # (a) Control System Architecture Performance
    control_systems = ['Classical\nPlanning', 'Probabilistic\nMethods', 'Optimization\nControl', 'Deep RL\nApproaches', 'Hybrid\nSystems']
    success_rates = [72.3, 81.0, 87.2, 76.1, 92.1]
    studies_count = [15, 12, 18, 10, 5]
    
    colors = ['#34495E', '#16A085', '#E67E22', '#8E44AD', '#C0392B']
    bars = ax1.bar(control_systems, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) Control System Architecture Performance')
    ax1.set_ylim(65, 95)
    ax1.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, studies_count):
        height = bar.get_height()
        ax1.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # (b) Algorithm Family Performance Comparison
    algorithms = ['RRT*', 'A3C', 'DDPG', 'PPO', 'SAC', 'Hybrid-RL']
    success_rates_alg = [82.1, 89.1, 86.9, 87.3, 89.7, 88.3]
    cycle_times = [245, 76, 178, 156, 71, 128]  # ms为单位，转换为秒显示
    
    # 创建双y轴
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar(np.arange(len(algorithms)) - 0.2, success_rates_alg, 0.4, 
                   label='Success Rate', color='#2ECC71', alpha=0.8)
    bars2 = ax2_twin.bar(np.arange(len(algorithms)) + 0.2, [ct/1000 for ct in cycle_times], 0.4,
                        label='Cycle Time', color='#E74C3C', alpha=0.8)
    
    ax2.set_xlabel('Algorithm Type')
    ax2.set_ylabel('Success Rate (%)', color='#2ECC71')
    ax2_twin.set_ylabel('Cycle Time (s)', color='#E74C3C')
    ax2.set_title('(b) Algorithm Family Achievements Comparison')
    ax2.set_xticks(np.arange(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(75, 95)
    ax2_twin.set_ylim(0.05, 0.25)
    
    # (c) Recent Robotics Model Evolution Timeline
    years_robot = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    classical_evolution = np.array([75.8, 76.2, 77.1, 78.5, 79.2, 80.1, 81.3, 82.1])
    rl_evolution = np.array([84.2, 85.9, 86.9, 87.8, 88.5, 89.1, 89.7, 90.3])
    
    ax3.plot(years_robot, classical_evolution, 'o-', label='Classical Methods', linewidth=2, markersize=6, color='#34495E')
    ax3.plot(years_robot, rl_evolution, 's-', label='RL-based Methods', linewidth=2, markersize=6, color='#8E44AD')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Peak Success Rate (%)')
    ax3.set_title('(c) Recent Robotics Model Evolution Timeline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(75, 92)
    
    # (d) Multi-Environmental Performance Analysis
    environments_robot = ['Greenhouse', 'Structured\nOrchard', 'Unstructured\nField', 'Mixed\nConditions']
    classical_perf = [85.2, 79.8, 72.4, 76.1]
    rl_perf = [91.3, 87.6, 84.2, 86.8]
    hybrid_perf = [93.1, 90.4, 88.7, 91.2]
    
    x_env = np.arange(len(environments_robot))
    width_env = 0.25
    
    bars1_env = ax4.bar(x_env - width_env, classical_perf, width_env, label='Classical', color='#34495E', alpha=0.8)
    bars2_env = ax4.bar(x_env, rl_perf, width_env, label='Deep RL', color='#8E44AD', alpha=0.8)
    bars3_env = ax4.bar(x_env + width_env, hybrid_perf, width_env, label='Hybrid', color='#C0392B', alpha=0.8)
    
    ax4.set_xlabel('Environment Type')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('(d) Multi-Environmental Performance Analysis')
    ax4.set_xticks(x_env)
    ax4.set_xticklabels(environments_robot)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(70, 95)
    
    plt.tight_layout()
    save_figure(fig, "9_motion_planning")
    plt.close()
    print("✅ Figure 9 生成完成")

# =============================================================================
# Figure 10: Critical Analysis and Future Trends
# =============================================================================

def create_figure10_critical_analysis():
    """创建Figure 10: 批判性分析和未来趋势"""
    print("🎨 生成Figure 10: 批判性分析和未来趋势...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGSIZE_10)
    fig.suptitle('Critical Analysis and Future Trends in Autonomous Fruit Harvesting Research', fontsize=16, fontweight='bold')
    
    # (a) 当前研究的根本性问题分析
    fundamental_problems = ['Lab-Field\nGap', 'Cost-Benefit\nMismatch', 'Limited\nGeneralization', 'Environmental\nSensitivity', 'Energy\nInefficiency', 'Maintenance\nComplexity']
    problem_severity = [8.5, 9.2, 8.8, 7.9, 8.3, 8.7]  # 问题严重程度 (1-10)
    research_attention = [4.2, 3.8, 5.1, 6.2, 3.9, 4.5]  # 研究关注度 (1-10)
    
    # 创建双轴图显示问题严重程度与研究关注度的不匹配
    x_pos = np.arange(len(fundamental_problems))
    bars1 = ax1.bar(x_pos - 0.2, problem_severity, 0.4, label='Problem Severity', color='#E74C3C', alpha=0.8)
    bars2 = ax1.bar(x_pos + 0.2, research_attention, 0.4, label='Research Attention', color='#3498DB', alpha=0.8)
    
    # 标注差距
    for i, (sev, att) in enumerate(zip(problem_severity, research_attention)):
        gap = sev - att
        ax1.annotate(f'Gap: {gap:.1f}', xy=(i, max(sev, att) + 0.2), ha='center', fontsize=8, 
                    color='red' if gap > 3 else 'orange', fontweight='bold')
    
    ax1.set_xlabel('Fundamental Problems')
    ax1.set_ylabel('Score (1-10)')
    ax1.set_title('(a) Research-Reality Mismatch Analysis')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(fundamental_problems, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)
    
    # (b) 技术发展瓶颈识别矩阵
    bottlenecks = ['Perception\nAccuracy', 'Real-time\nProcessing', 'Mechanical\nReliability', 'Cost\nControl', 'Energy\nEfficiency', 'Multi-crop\nAdaptability']
    technical_difficulty = [8.9, 9.1, 8.6, 9.3, 8.4, 9.5]  # 技术难度
    commercial_urgency = [9.2, 8.8, 9.4, 9.8, 7.6, 8.9]   # 商业紧迫性
    current_progress = [6.8, 5.9, 7.2, 4.3, 5.5, 3.8]     # 当前进展
    
    # 创建气泡图，气泡大小表示技术难度
    colors_bubble = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    scatter = ax2.scatter(commercial_urgency, current_progress, 
                        s=[d*30 for d in technical_difficulty], 
                        c=colors_bubble, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for i, bottleneck in enumerate(bottlenecks):
        ax2.annotate(bottleneck, (commercial_urgency[i], current_progress[i]), 
                    xytext=(5, 5), textcoords="offset points", fontsize=9, ha='left')
    
    # 添加危险区域标识
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Critical Threshold')
    ax2.fill_between([7, 10], [0, 0], [5, 5], alpha=0.2, color='red', label='High Risk Zone')
    
    ax2.set_xlabel('Commercial Urgency (1-10)')
    ax2.set_ylabel('Current Progress Level (1-10)')
    ax2.set_title('(b) Technical Bottleneck Matrix')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(7, 10)
    ax2.set_ylim(3, 8)
    
    # (c) 未解决关键挑战的时间演进
    years = np.array([2015, 2017, 2019, 2021, 2023, 2024])
    
    # 各项挑战的严重程度随时间变化（显示问题持续性）
    occlusion_challenge = np.array([8.5, 8.3, 8.0, 7.8, 7.6, 7.4])  # 遮挡问题略有改善
    cost_challenge = np.array([9.0, 9.1, 9.3, 9.4, 9.3, 9.2])       # 成本问题持续严重
    generalization_challenge = np.array([8.8, 8.9, 8.7, 8.5, 8.4, 8.3])  # 泛化性问题缓慢改善
    deployment_challenge = np.array([9.5, 9.4, 9.2, 8.9, 8.6, 8.4])      # 部署问题有所改善
    
    ax3.plot(years, occlusion_challenge, 'o-', linewidth=3, markersize=8, 
            color='#E74C3C', label='Occlusion Handling', alpha=0.8)
    ax3.plot(years, cost_challenge, 's-', linewidth=3, markersize=8, 
            color='#8E44AD', label='Cost-Effectiveness', alpha=0.8)
    ax3.plot(years, generalization_challenge, '^-', linewidth=3, markersize=8, 
            color='#F39C12', label='Cross-crop Generalization', alpha=0.8)
    ax3.plot(years, deployment_challenge, 'd-', linewidth=3, markersize=8, 
            color='#2ECC71', label='Field Deployment', alpha=0.8)
    
    # 添加危机水平线
    ax3.axhline(y=8.5, color='red', linestyle=':', alpha=0.7, label='Crisis Level')
    
    # 标注关键时间点
    ax3.annotate('COVID-19 Impact', xy=(2021, 9.4), xytext=(2019, 9.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=10, ha='center')
    ax3.annotate('AI Boom', xy=(2023, 7.6), xytext=(2022, 6.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2), fontsize=10, ha='center')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Challenge Severity (1-10)')
    ax3.set_title('(c) Persistent Challenges Evolution (2015-2024)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(6.5, 10)
    
    # (d) 批判性趋势分析 - 研究热点vs实际需求错位
    research_topics = ['Deep Learning\nArchitectures', 'Novel Sensors', 'Advanced\nAlgorithms', 'Cost\nOptimization', 'Field\nValidation', 'Commercial\nViability']
    research_publications = [95, 78, 87, 23, 31, 18]  # 相对发表数量
    industry_demand = [65, 45, 55, 92, 88, 95]        # 行业需求程度
    
    # 创建对比图显示研究热点与需求的错位
    x_pos = np.arange(len(research_topics))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, research_publications, width, 
                    label='Research Publications (%)', color='#3498DB', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, industry_demand, width, 
                    label='Industry Demand (%)', color='#E67E22', alpha=0.8)
    
    # 标注错位严重的领域
    for i, (pub, dem) in enumerate(zip(research_publications, industry_demand)):
        mismatch = abs(pub - dem)
        if mismatch > 30:  # 错位严重的标红
            ax4.annotate(f'Mismatch!\n±{mismatch}', xy=(i, max(pub, dem) + 5), 
                        ha='center', fontsize=8, color='red', fontweight='bold')
    
    ax4.set_xlabel('Research Areas')
    ax4.set_ylabel('Relative Intensity (%)')
    ax4.set_title('(d) Research-Industry Priority Misalignment')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(research_topics, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 110)
    
    # 添加警告线
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Balance Line')
    
    plt.tight_layout()
    save_figure(fig, "10_technology_roadmap")
    plt.close()
    print("✅ Figure 10 生成完成")

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 生成所有图片"""
    print("=" * 60)
    print("🚀 开始生成论文图片 (本地版本)")
    print("=" * 60)
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📊 输出格式: {SAVE_FORMATS}")
    print(f"📏 分辨率: {DPI} DPI")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 生成所有图片
        create_figure4_vision_analysis()
        print()
        create_figure9_robotics_analysis()
        print()
        create_figure10_critical_analysis()
        print()
        
        print("=" * 60)
        print("🎉 所有图片生成完成！")
        print("=" * 60)
        print("生成的文件:")
        for fmt in SAVE_FORMATS:
            print(f"  - {FIGURE_PREFIX}4_meta_analysis.{fmt}")
            print(f"  - {FIGURE_PREFIX}9_motion_planning.{fmt}")
            print(f"  - {FIGURE_PREFIX}10_technology_roadmap.{fmt}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        print("请检查matplotlib和numpy是否已安装")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())