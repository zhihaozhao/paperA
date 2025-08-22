#!/usr/bin/env python3
"""
Motion Planning Figure Generation for IEEE Access Paper Section V
Creates publication-quality figures for motion control analysis in fruit-picking robots

Author: Research Team  
Date: August 2025
Purpose: Generate high-impact motion planning visualizations for journal submission
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set publication-ready style for IEEE journals
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15
})

def create_motion_planning_data():
    """Create comprehensive motion planning performance data from literature review"""
    
    # Data extracted from Table in Section V
    data = {
        'Study': ['Silwal et al.', 'Arad et al.', 'Xiong et al.', 'Williams et al.',
                 'Xiong et al.', 'Lehnert et al.', 'Ling et al.', 'Lin et al.',
                 'Sepulveda et al.', 'Bac et al.', 'Mehta et al.', 'Williams et al.',
                 'Kang et al.', 'Vougioukas', 'Verbiest et al.', 'Zhang et al.'],
        'Year': [2017, 2020, 2020, 2019, 2019, 2017, 2019, 2021, 2020, 2016, 2016, 2020, 2020, 2019, 2022, 2023],
        'Fruit': ['Apple', 'Sweet Pepper', 'Strawberry', 'Kiwifruit', 'Strawberry', 
                 'Sweet Pepper', 'Tomato', 'Guava', 'Aubergine', 'Sweet Pepper',
                 'Citrus', 'Kiwifruit', 'Apple', 'General', 'Pepper', 'Apple'],
        'Algorithm': ['7-DOF + Path Planning', 'Vision-Navigation', 'Dual-arm + Obstacle Sep.',
                     'Multi-arm Coordination', 'Adaptive Path Correction', '7-DOF Planning',
                     'Dual-arm + Binocular', 'Recurrent DDPG', 'Dual-arm + SVM',
                     'Bi-RRT', 'Visual Servo', 'Vision-guided Path', 'PointNet + Grasping',
                     'Multi-robot Coordination', 'RL-based Collision-free', 'Deep RL'],
        'Success_Rate': [84, 39.5, 75, 70, 75, 58, 87.5, 90.9, 91.67, 63, 85, 51, 85, 70, 92, 88],
        'Cycle_Time': [7.6, 24.0, 6.1, 8.0, 7.5, 12.0, 8.0, 0.029, 26.0, 15.0, 10.0, 5.5, 6.5, 10.0, 0.05, 5.0],
        'Environment': ['Commercial Orchard', 'Greenhouse', 'Polytunnel', 'Orchard', 
                       'Field', 'Protected Crop', 'Dense Vegetation', 'Unstructured Orchard',
                       'Lab', 'Dense Obstacles', 'Simulation', 'Orchard', 'Field',
                       'Orchard', 'Lab/Field', 'Simulation'],
        'Algorithm_Type': ['Classical', 'Hybrid', 'Classical', 'Classical', 'Adaptive',
                          'Classical', 'Vision-based', 'RL', 'ML-based', 'Classical',
                          'Control-based', 'Vision-based', 'DL-based', 'Multi-robot',
                          'RL', 'Deep RL']
    }
    
    return pd.DataFrame(data)

def create_motion_planning_analysis_figure():
    """Create comprehensive 4-panel motion planning analysis figure"""
    
    print("Creating Motion Planning Analysis Figure...")
    
    # Create data
    df = create_motion_planning_data()
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Motion Planning Analysis for Autonomous Fruit Harvesting\n(16 Studies, 2015-2024)', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Panel A: System Architecture Overview
    ax1 = axes[0, 0]
    
    # Create system architecture diagram
    components = ['Perception\nSystem', 'Motion\nPlanner', 'Control\nSystem', 'End\nEffector']
    y_positions = [0.8, 0.6, 0.4, 0.2]
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
    
    for i, (comp, y_pos, color) in enumerate(zip(components, y_positions, colors)):
        # Create component boxes
        box = FancyBboxPatch((0.1, y_pos-0.05), 0.8, 0.1, 
                           boxstyle="round,pad=0.02", 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax1.add_patch(box)
        ax1.text(0.5, y_pos, comp, ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Add arrows between components
        if i < len(components) - 1:
            ax1.arrow(0.5, y_pos-0.05, 0, -0.05, head_width=0.03, head_length=0.02, 
                     fc='black', ec='black')
    
    # Add feedback loop
    ax1.arrow(0.95, 0.2, 0, 0.55, head_width=0.03, head_length=0.02, 
             fc='red', ec='red', linestyle='--', alpha=0.7)
    ax1.text(0.97, 0.5, 'Feedback', rotation=90, ha='center', va='center', 
             color='red', fontweight='bold')
    
    ax1.set_xlim(0, 1.2)
    ax1.set_ylim(0, 1)
    ax1.set_title('(A) System Architecture Integration', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Algorithm Performance Trade-offs
    ax2 = axes[0, 1]
    
    # Success rate vs cycle time scatter
    algorithms = df['Algorithm_Type'].unique()
    algo_colors = {'Classical': '#E74C3C', 'Hybrid': '#F39C12', 'RL': '#27AE60', 
                  'Vision-based': '#3498DB', 'DL-based': '#9B59B6', 'ML-based': '#1ABC9C',
                  'Control-based': '#34495E', 'Adaptive': '#E67E22', 'Multi-robot': '#8E44AD',
                  'Deep RL': '#2ECC71'}
    
    for algo_type in algorithms:
        algo_data = df[df['Algorithm_Type'] == algo_type]
        color = algo_colors.get(algo_type, '#95A5A6')
        
        ax2.scatter(algo_data['Cycle_Time'], algo_data['Success_Rate'], 
                   c=color, label=algo_type, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Cycle Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')  
    ax2.set_title('(B) Algorithm Performance Trade-offs', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 27)
    ax2.set_ylim(20, 95)
    
    # Panel C: Algorithm Evolution Timeline
    ax3 = axes[1, 0]
    
    # Group by algorithm type and year
    timeline_data = df.groupby(['Year', 'Algorithm_Type']).size().unstack(fill_value=0)
    
    # Create stacked bar chart
    bottom = np.zeros(len(timeline_data.index))
    colors_list = [algo_colors.get(col, '#95A5A6') for col in timeline_data.columns]
    
    bars = []
    for i, (col, color) in enumerate(zip(timeline_data.columns, colors_list)):
        bar = ax3.bar(timeline_data.index, timeline_data[col], bottom=bottom, 
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars.append(bar)
        bottom += timeline_data[col]
    
    ax3.set_xlabel('Publication Year', fontweight='bold')
    ax3.set_ylabel('Number of Studies', fontweight='bold')
    ax3.set_title('(C) Algorithm Adoption Timeline', fontweight='bold')
    ax3.legend(timeline_data.columns, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Environmental Performance Analysis
    ax4 = axes[1, 1]
    
    # Environment vs success rate analysis
    env_performance = df.groupby('Environment')['Success_Rate'].agg(['mean', 'std', 'count'])
    env_performance = env_performance.sort_values('mean', ascending=False)
    
    # Create bar plot with error bars
    bars = ax4.bar(range(len(env_performance)), env_performance['mean'], 
                  yerr=env_performance['std'], capsize=5, 
                  color=['#27AE60' if x > 75 else '#F39C12' if x > 50 else '#E74C3C' 
                        for x in env_performance['mean']], 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, env_performance['count'])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax4.set_xlabel('Environment Type', fontweight='bold')
    ax4.set_ylabel('Average Success Rate (%)', fontweight='bold')
    ax4.set_title('(D) Environmental Performance Comparison', fontweight='bold')
    ax4.set_xticks(range(len(env_performance)))
    ax4.set_xticklabels(env_performance.index, rotation=45, ha='right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Adjust layout to prevent overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    
    # Save figure
    plt.savefig('fig_motion_planning_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_motion_planning_analysis.pdf', dpi=300, bbox_inches='tight')
    print("✅ Motion Planning Analysis Figure saved: fig_motion_planning_analysis.png/.pdf")
    
    return fig

def create_enhanced_motion_control_table():
    """Create enhanced motion control performance table for better readability"""
    
    df = create_motion_planning_data()
    
    # Create summary statistics table
    summary_data = []
    
    # Group by algorithm type
    for algo_type in df['Algorithm_Type'].unique():
        algo_subset = df[df['Algorithm_Type'] == algo_type]
        
        summary_data.append({
            'Algorithm_Type': algo_type,
            'Studies': len(algo_subset),
            'Avg_Success_Rate': f"{algo_subset['Success_Rate'].mean():.1f}%",
            'Success_Range': f"{algo_subset['Success_Rate'].min():.0f}-{algo_subset['Success_Rate'].max():.0f}%",
            'Avg_Cycle_Time': f"{algo_subset['Cycle_Time'].mean():.1f}s",
            'Time_Range': f"{algo_subset['Cycle_Time'].min():.1f}-{algo_subset['Cycle_Time'].max():.1f}s",
            'Primary_Fruits': ', '.join(algo_subset['Fruit'].unique()[:3])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by average success rate
    summary_df = summary_df.sort_values('Avg_Success_Rate', ascending=False)
    
    # Create LaTeX table
    latex_table = """
\\begin{table*}[htbp]
\\centering
\\small
\\caption{Enhanced Motion Control Algorithm Performance Summary (2015-2024)}
\\label{tab:motion_control_summary}
\\begin{tabular}{p{2.5cm}p{1cm}p{2cm}p{2cm}p{2cm}p{2cm}p{3.5cm}}
\\toprule
\\textbf{Algorithm Type} & \\textbf{Studies} & \\textbf{Avg Success} & \\textbf{Success Range} & \\textbf{Avg Cycle Time} & \\textbf{Time Range} & \\textbf{Primary Applications} \\\\
\\midrule
"""
    
    for _, row in summary_df.iterrows():
        latex_table += f"{row['Algorithm_Type']} & {row['Studies']} & {row['Avg_Success_Rate']} & {row['Success_Range']} & {row['Avg_Cycle_Time']} & {row['Time_Range']} & {row['Primary_Fruits']} \\\\\n"
        latex_table += "\\midrule\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Save enhanced table
    with open('enhanced_motion_control_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✅ Enhanced Motion Control Table saved: enhanced_motion_control_table.tex")
    return summary_df

def create_motion_planning_dashboard():
    """Create comprehensive motion planning dashboard figure"""
    
    print("Creating Motion Planning Dashboard...")
    
    df = create_motion_planning_data()
    
    # Create figure with complex layout
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4, 
                         top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    # Panel 1: Algorithm Performance Matrix (large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Create performance heatmap
    perf_matrix = df.pivot_table(values='Success_Rate', 
                                index='Algorithm_Type', 
                                columns='Environment', 
                                aggfunc='mean').fillna(0)
    
    im = ax1.imshow(perf_matrix.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add text annotations
    for i in range(len(perf_matrix.index)):
        for j in range(len(perf_matrix.columns)):
            if perf_matrix.iloc[i, j] > 0:
                text = ax1.text(j, i, f'{perf_matrix.iloc[i, j]:.0f}%',
                               ha='center', va='center', fontweight='bold',
                               color='white' if perf_matrix.iloc[i, j] < 50 else 'black')
    
    ax1.set_xticks(range(len(perf_matrix.columns)))
    ax1.set_yticks(range(len(perf_matrix.index)))
    ax1.set_xticklabels(perf_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(perf_matrix.index, fontsize=10)
    ax1.set_title('Algorithm Performance by Environment', fontweight='bold', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Success Rate (%)', fontweight='bold')
    
    # Panel 2: Evolution Timeline
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Timeline plot
    for algo_type in df['Algorithm_Type'].unique():
        algo_data = df[df['Algorithm_Type'] == algo_type]
        if len(algo_data) > 1:
            ax2.plot(algo_data['Year'], algo_data['Success_Rate'], 'o-', 
                    label=algo_type, linewidth=2, markersize=6, alpha=0.8)
    
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('Performance Evolution Timeline', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Cycle Time Analysis
    ax3 = fig.add_subplot(gs[1, 2:])
    
    # Box plot of cycle times by algorithm type
    cycle_data = []
    cycle_labels = []
    
    for algo_type in df['Algorithm_Type'].unique():
        algo_subset = df[df['Algorithm_Type'] == algo_type]
        if len(algo_subset) > 0:
            # Filter out extremely small values (likely in different units)
            cycle_times = algo_subset['Cycle_Time']
            cycle_times = cycle_times[cycle_times > 1]  # Focus on realistic times
            if len(cycle_times) > 0:
                cycle_data.append(cycle_times)
                cycle_labels.append(f'{algo_type}\n(n={len(cycle_times)})')
    
    bp = ax3.boxplot(cycle_data, labels=cycle_labels, patch_artist=True)
    
    # Color boxes
    colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB', '#9B59B6']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Cycle Time (seconds)', fontweight='bold')
    ax3.set_title('Cycle Time Distribution by Algorithm', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Key Performance Indicators
    ax4 = fig.add_subplot(gs[2, :])
    
    # Create KPI summary
    kpis = {
        'Best Success Rate': f"{df['Success_Rate'].max():.1f}% (Verbiest et al., RL-based)",
        'Fastest Cycle': f"{df[df['Cycle_Time'] > 1]['Cycle_Time'].min():.1f}s (Zhang et al., Deep RL)",
        'Most Robust': f"Multi-environment validation (Vougioukas, Multi-robot)",
        'Commercial Ready': f"84% success in orchards (Silwal et al., 7-DOF)",
        'Future Trend': f"RL approaches: 90.9% success, 29ms planning (Lin et al.)"
    }
    
    # Display KPIs as text boxes
    y_pos = 0.7
    for i, (kpi, value) in enumerate(kpis.items()):
        color = ['#27AE60', '#3498DB', '#F39C12', '#E74C3C', '#9B59B6'][i]
        
        # Create colored box
        box = FancyBboxPatch((0.02 + i*0.19, y_pos-0.1), 0.18, 0.2, 
                           boxstyle="round,pad=0.02", 
                           facecolor=color, alpha=0.3, edgecolor=color)
        ax4.add_patch(box)
        
        # Add KPI text
        ax4.text(0.11 + i*0.19, y_pos+0.05, kpi, ha='center', va='center', 
                fontweight='bold', fontsize=11)
        ax4.text(0.11 + i*0.19, y_pos-0.05, value, ha='center', va='center', 
                fontsize=10, wrap=True)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Key Performance Indicators & Research Highlights', fontweight='bold', fontsize=14)
    ax4.axis('off')
    
    # Save figure
    plt.savefig('fig_motion_planning_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_motion_planning_analysis.pdf', dpi=300, bbox_inches='tight')
    print("✅ Motion Planning Dashboard saved: fig_motion_planning_analysis.png/.pdf")
    
    return fig

if __name__ == "__main__":
    print("🤖 Generating Motion Planning Analysis Figures for Section V...")
    print("📊 Creating IEEE Access Quality Visualizations...")
    
    try:
        # Generate main motion planning figure
        fig1 = create_motion_planning_analysis_figure()
        
        # Generate enhanced dashboard
        fig2 = create_motion_planning_dashboard()
        
        # Generate enhanced table
        summary_df = create_enhanced_motion_control_table()
        
        print("\n🎉 ALL SECTION V FIGURES AND TABLES GENERATED!")
        print("\n📁 Files Created:")
        print("  • fig_motion_planning_analysis.png/pdf - Main 4-panel analysis")
        print("  • enhanced_motion_control_table.tex - Improved summary table")
        print("\n📋 Integration Instructions:")
        print("  1. Copy figures to IEEE Access directory")
        print("  2. Uncomment \\includegraphics in Section V")
        print("  3. Replace existing table with enhanced version")
        print("  4. Compile LaTeX document")
        
        # Display plots
        plt.show()
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        print("💡 Note: Run this script in environment with pandas, matplotlib, seaborn installed")