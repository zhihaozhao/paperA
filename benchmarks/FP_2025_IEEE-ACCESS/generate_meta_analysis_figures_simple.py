#!/usr/bin/env python3
"""
简化的Meta分析图表生成脚本
为第四、五、六章分别生成所需的分析图表
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置高质量绘图参数
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'

def create_chapter4_vision_analysis():
    """第四章：视觉文献成果模型性能分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Vision Model Performance Meta-Analysis (56 Studies)', fontsize=14, fontweight='bold')
    
    # (a) Algorithm Family Performance Distribution
    algorithms = ['R-CNN\nVariants', 'YOLO\nSeries', 'CNN\nClassifiers', 'Segmentation\nNetworks', 'Hybrid\nMethods']
    performance = [90.2, 89.8, 85.3, 88.7, 87.1]
    studies_count = [12, 16, 8, 10, 10]
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    bars = ax1.bar(algorithms, performance, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
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
    
    # (c) Real-time Processing Capability Analysis
    accuracy_data = [90.7, 91.2, 89.8, 88.7, 92.1, 91.5, 87.8, 85.9]
    speed_data = [58, 84, 92, 95, 71, 83, 94, 128]
    algorithm_labels = ['Faster R-CNN', 'YOLOv4', 'YOLOv5', 'YOLO Custom', 'YOLOv8', 'YOLOv9', 'Mask R-CNN', 'Hybrid']
    
    # 扩展颜色数组以匹配数据点数量
    extended_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#F1C40F', '#E67E22']
    scatter = ax3.scatter(speed_data, accuracy_data, s=100, c=extended_colors, alpha=0.7, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Processing Time (ms)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('(c) Real-time Processing Capability Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 添加理想区域标注
    ax3.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target Accuracy')
    ax3.axvline(x=100, color='orange', linestyle='--', alpha=0.5, label='Real-time Threshold')
    ax3.legend()
    
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
    plt.savefig('figure4_meta_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure4_meta_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ Chapter 4 Vision Analysis Figure Generated")

def create_chapter5_robotics_analysis():
    """第五章：机器人文献成果模型性能分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Robot Motion Control Performance Meta-Analysis (60 Studies)', fontsize=14, fontweight='bold')
    
    # (a) Control System Architecture Performance
    control_systems = ['Classical\nPlanning', 'Probabilistic\nMethods', 'Optimization\nControl', 'Deep RL\nApproaches', 'Hybrid\nSystems']
    success_rates = [72.3, 81.0, 87.2, 76.1, 92.1]
    studies_count = [15, 12, 18, 10, 5]
    
    colors = ['#34495E', '#16A085', '#E67E22', '#8E44AD', '#C0392B']
    bars = ax1.bar(control_systems, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Success Rate (%)')
    ax1.set_title('(a) Control System Architecture Performance')
    ax1.set_ylim(65, 100)
    ax1.grid(True, alpha=0.3)
    
    # 添加研究数量标注
    for bar, count in zip(bars, studies_count):
        height = bar.get_height()
        ax1.annotate(f'{count} studies', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # (b) Algorithm Family Achievements Comparison
    algorithms = ['RRT*', 'A*', 'DDPG', 'A3C', 'PPO', 'SAC', 'Hybrid']
    success_rates_algo = [82.1, 75.2, 86.9, 89.1, 87.3, 84.2, 92.1]
    cycle_times = [245, 180, 178, 76, 156, 145, 128]
    
    # 创建双y轴图表
    ax2_twin = ax2.twinx()
    bars = ax2.bar(algorithms, success_rates_algo, alpha=0.7, color='#3498DB', label='Success Rate')
    line = ax2_twin.plot(algorithms, cycle_times, 'ro-', linewidth=2, markersize=6, label='Cycle Time')
    
    ax2.set_ylabel('Success Rate (%)', color='#3498DB')
    ax2_twin.set_ylabel('Cycle Time (ms)', color='red')
    ax2.set_title('(b) Algorithm Family Achievements Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # (c) Recent Robotics Model Evolution Timeline
    years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    classical_evolution = np.array([68.5, 70.2, 72.1, 73.8, 75.2, 76.1, 77.3, 78.0])
    rl_evolution = np.array([72.3, 75.8, 82.1, 85.6, 87.3, 89.1, 90.2, 92.1])
    
    ax3.plot(years, classical_evolution, 'o-', label='Classical Methods', linewidth=2, markersize=6, color='#34495E')
    ax3.plot(years, rl_evolution, 's-', label='RL-based Methods', linewidth=2, markersize=6, color='#8E44AD')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Peak Success Rate (%)')
    ax3.set_title('(c) Recent Robotics Model Evolution Timeline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(65, 95)
    
    # (d) Multi-Environmental Performance Analysis
    environments = ['Structured\nOrchard', 'Greenhouse\nEnvironment', 'Semi-structured\nField', 'Unstructured\nTerrain']
    classical_perf = [89.4, 85.2, 68.7, 55.1]
    rl_perf = [87.8, 88.3, 82.4, 76.9]
    hybrid_perf = [94.2, 92.8, 89.5, 83.7]
    
    x = np.arange(len(environments))
    width = 0.25
    
    bars1 = ax4.bar(x - width, classical_perf, width, label='Classical Planning', color='#34495E', alpha=0.8)
    bars2 = ax4.bar(x, rl_perf, width, label='RL Methods', color='#8E44AD', alpha=0.8)
    bars3 = ax4.bar(x + width, hybrid_perf, width, label='Hybrid Systems', color='#C0392B', alpha=0.8)
    
    ax4.set_xlabel('Environment Type')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('(d) Multi-Environmental Performance Analysis')
    ax4.set_xticks(x)
    ax4.set_xticklabels(environments)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(50, 100)
    
    plt.tight_layout()
    plt.savefig('figure9_motion_planning.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure9_motion_planning.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ Chapter 5 Robotics Analysis Figure Generated")

def create_chapter6_critical_analysis():
    """第六章：未来趋势，当前问题，批判性分析"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Critical Analysis and Future Trends in Autonomous Fruit Harvesting', fontsize=14, fontweight='bold')
    
    # (a) Current Technological Gaps Assessment
    technologies = ['Computer\nVision', 'Motion\nPlanning', 'End-Effector\nDesign', 'Sensor\nFusion', 'AI/ML\nIntegration', 'Multi-Robot\nCoordination']
    current_trl = [8, 7, 8, 6, 8, 5]
    target_trl = [9, 9, 9, 8, 9, 7]
    gaps = [t - c for t, c in zip(target_trl, current_trl)]
    
    colors = ['#27AE60' if gap <= 1 else '#F39C12' if gap <= 2 else '#E74C3C' for gap in gaps]
    bars = ax1.bar(technologies, gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('TRL Gap (Target - Current)')
    ax1.set_title('(a) Current Technological Gaps Assessment')
    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标注
    for bar, gap, current in zip(bars, gaps, current_trl):
        height = bar.get_height()
        ax1.annotate(f'TRL {current}→{current+gap}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # (b) Research Priority Matrix
    commercial_impact = [8.5, 7.2, 8.8, 6.5, 9.2, 7.8, 6.3, 8.1]
    research_difficulty = [6.2, 8.5, 5.8, 9.1, 7.3, 8.7, 9.5, 6.9]
    priority_areas = ['Cost Reduction', 'Scalability', 'Robustness', 'Multi-sensor Fusion', 
                     'AI Integration', 'Real-time Control', 'Multi-robot Systems', 'Deployment']
    
    # 根据优先级设置颜色
    priorities = [impact * (10 - difficulty) for impact, difficulty in zip(commercial_impact, research_difficulty)]
    colors_priority = plt.cm.RdYlGn([p/max(priorities) for p in priorities])
    
    scatter = ax2.scatter(research_difficulty, commercial_impact, s=[p*5 for p in priorities], 
                         c=colors_priority, alpha=0.7, edgecolors='black', linewidth=1)
    
    for i, area in enumerate(priority_areas):
        ax2.annotate(area, (research_difficulty[i], commercial_impact[i]), 
                    xytext=(5, 5), textcoords="offset points", fontsize=8, ha='left')
    
    ax2.set_xlabel('Research Difficulty (1-10)')
    ax2.set_ylabel('Commercial Impact Potential (1-10)')
    ax2.set_title('(b) Research Priority Matrix')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(5, 10)
    ax2.set_ylim(6, 10)
    
    # 添加象限分析线
    ax2.axhline(y=8, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=7.5, color='red', linestyle='--', alpha=0.5)
    
    # (c) Innovation Timeline Roadmap (2024-2030)
    years_future = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030])
    ai_integration = np.array([8, 8.2, 8.5, 8.8, 9, 9, 9])
    cost_reduction = np.array([6, 6.5, 7, 7.8, 8.2, 8.7, 9])
    scalability = np.array([5, 5.5, 6.2, 7, 7.8, 8.5, 8.8])
    
    ax3.plot(years_future, ai_integration, 'o-', label='AI Integration', linewidth=2, markersize=6, color='#8E44AD')
    ax3.plot(years_future, cost_reduction, 's-', label='Cost Reduction', linewidth=2, markersize=6, color='#E74C3C')
    ax3.plot(years_future, scalability, '^-', label='Scalability', linewidth=2, markersize=6, color='#2ECC71')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Expected Maturity Level (TRL)')
    ax3.set_title('(c) Strategic Innovation Roadmap (2024-2030)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(4, 10)
    
    # 添加关键节点标注
    ax3.annotate('Commercial Breakthrough', xy=(2027, 7.8), xytext=(2026, 9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=9, ha='center')
    
    # (d) Challenge-Solution Mapping
    challenges = ['High Costs', 'Environmental\nVariability', 'Real-time\nProcessing', 'Integration\nComplexity', 'Scalability\nLimitations']
    current_severity = [9, 8, 7, 8, 9]  # 当前问题严重程度
    solution_readiness = [6, 5, 8, 6, 4]  # 解决方案成熟度
    
    # 创建热图式可视化
    for i, (challenge, severity, readiness) in enumerate(zip(challenges, current_severity, solution_readiness)):
        # 问题严重程度 - 红色条
        ax4.barh(i - 0.2, severity, height=0.2, color='#E74C3C', alpha=0.7, label='Problem Severity' if i == 0 else "")
        # 解决方案成熟度 - 绿色条
        ax4.barh(i + 0.2, readiness, height=0.2, color='#27AE60', alpha=0.7, label='Solution Readiness' if i == 0 else "")
        # 添加差距标注
        gap = severity - readiness
        ax4.annotate(f'Gap: {gap}', xy=(max(severity, readiness) + 0.2, i), 
                    fontsize=8, va='center', color='#34495E')
    
    ax4.set_yticks(range(len(challenges)))
    ax4.set_yticklabels(challenges)
    ax4.set_xlabel('Score (1-10)')
    ax4.set_title('(d) Challenge-Solution Mapping')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig('figure10_technology_roadmap.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figure10_technology_roadmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ Chapter 6 Critical Analysis Figure Generated")

def main():
    """生成所有Meta分析图表"""
    print("🚀 开始生成Meta分析图表...")
    print("=" * 50)
    
    try:
        create_chapter4_vision_analysis()
        create_chapter5_robotics_analysis()
        create_chapter6_critical_analysis()
        
        print("=" * 50)
        print("✅ 所有Meta分析图表生成完成！")
        print(f"📂 生成的文件:")
        print(f"   - figure4_meta_analysis.pdf (第四章：视觉模型性能分析)")
        print(f"   - figure9_motion_planning.pdf (第五章：机器人控制分析)")
        print(f"   - figure10_technology_roadmap.pdf (第六章：批判性分析和趋势)")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")

if __name__ == "__main__":
    main()