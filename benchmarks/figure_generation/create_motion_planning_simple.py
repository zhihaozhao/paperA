#!/usr/bin/env python3
"""
Simplified Motion Planning Figure for Section V (No external dependencies)
Generates motion planning analysis using only matplotlib and numpy
"""

import matplotlib.pyplot as plt
import numpy as np

# IEEE publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_simple_motion_planning_figure():
    """Create simplified 4-panel motion planning figure without external dependencies"""
    
    print("Creating Simplified Motion Planning Figure...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motion Planning Analysis for Autonomous Fruit Harvesting', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Panel A: Algorithm Performance Comparison
    ax1 = axes[0, 0]
    
    algorithms = ['A*', 'RRT', 'Bi-RRT', 'DDPG', 'Deep RL', 'Multi-robot']
    success_rates = [75, 70, 63, 90.9, 88, 70]
    colors = ['#E74C3C', '#F39C12', '#FF6B6B', '#27AE60', '#2ECC71', '#3498DB']
    
    bars = ax1.bar(algorithms, success_rates, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('(A) Algorithm Performance Comparison', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Panel B: Cycle Time vs Success Rate
    ax2 = axes[0, 1]
    
    cycle_times = [7.6, 24.0, 6.1, 8.0, 7.5, 0.029, 8.0, 26.0, 15.0, 5.5]
    success_rates_scatter = [84, 39.5, 75, 70, 75, 90.9, 87.5, 91.67, 63, 51]
    
    scatter = ax2.scatter(cycle_times, success_rates_scatter, 
                         c=range(len(cycle_times)), cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    
    # Add trend line
    if len(cycle_times) > 3:
        z = np.polyfit(cycle_times, success_rates_scatter, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(cycle_times), max(cycle_times), 100)
        ax2.plot(x_trend, p(x_trend), '--', color='red', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Cycle Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('(B) Speed vs Performance Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Evolution Timeline
    ax3 = axes[1, 0]
    
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    classical_count = [1, 2, 2, 1, 4, 3, 1, 1, 1, 0]
    rl_count = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    vision_count = [0, 1, 1, 1, 2, 2, 0, 0, 0, 0]
    
    ax3.plot(years, classical_count, 'o-', label='Classical Methods', linewidth=2, color='#E74C3C')
    ax3.plot(years, rl_count, 's-', label='RL-based Methods', linewidth=2, color='#27AE60')
    ax3.plot(years, vision_count, '^-', label='Vision-based Methods', linewidth=2, color='#3498DB')
    
    ax3.set_xlabel('Publication Year', fontweight='bold')
    ax3.set_ylabel('Number of Studies', fontweight='bold')
    ax3.set_title('(C) Algorithm Adoption Timeline', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Environment Performance Analysis
    ax4 = axes[1, 1]
    
    environments = ['Lab', 'Greenhouse', 'Field', 'Orchard']
    env_performance = [85, 70, 65, 75]
    env_counts = [3, 4, 5, 4]
    
    bars = ax4.bar(environments, env_performance, 
                  color=['#27AE60', '#F39C12', '#E74C3C', '#3498DB'], alpha=0.8)
    
    # Add count labels
    for bar, count in zip(bars, env_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Average Success Rate (%)', fontweight='bold')
    ax4.set_title('(D) Performance by Environment', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.subplots_adjust(hspace=0.35, wspace=0.4)
    
    return fig

def create_simple_enhanced_table():
    """Create enhanced motion control table in LaTeX format"""
    
    print("Creating Enhanced Motion Control Table...")
    
    # Enhanced table with better readability
    latex_table = """
\\begin{table*}[htbp]
\\centering
\\small
\\caption{Enhanced Motion Control Algorithm Performance Analysis for Fruit Harvesting (2015-2024)}
\\label{tab:motion_control_enhanced}
\\begin{tabular}{p{2.5cm}p{1.2cm}p{1.8cm}p{2.2cm}p{2.5cm}p{2cm}p{3cm}}
\\toprule
\\textbf{Algorithm Family} & \\textbf{Studies} & \\textbf{Success Rate} & \\textbf{Cycle Time} & \\textbf{Key Advantages} & \\textbf{Limitations} & \\textbf{Best Applications} \\\\
\\midrule
\\textbf{Deep RL} & 3 & 90.4\\% $\\pm$ 2.1 & 5.2s $\\pm$ 2.8 & Real-time adaptation, high accuracy & Training complexity & Unstructured orchards \\\\
\\midrule
\\textbf{Vision-based} & 4 & 73.1\\% $\\pm$ 15.2 & 7.8s $\\pm$ 1.2 & Robust perception integration & Light sensitivity & Greenhouse environments \\\\
\\midrule
\\textbf{Classical} & 6 & 70.8\\% $\\pm$ 9.4 & 9.7s $\\pm$ 3.2 & Reliable, well-tested & Limited adaptability & Structured orchards \\\\
\\midrule
\\textbf{Multi-robot} & 2 & 70.0\\% $\\pm$ 0 & 10.0s $\\pm$ 0 & Scalable operations & Coordination complexity & Large-scale harvesting \\\\
\\midrule
\\textbf{Hybrid/Adaptive} & 1 & 75.0\\% & 7.5s & Balanced performance & Implementation complexity & Mixed environments \\\\
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    
    # Save table
    with open('enhanced_motion_control_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✅ Enhanced table saved: enhanced_motion_control_table.tex")
    return latex_table

if __name__ == "__main__":
    print("🤖 Generating Simplified Motion Planning Figures for Section V...")
    
    try:
        # Create figure
        fig = create_simple_motion_planning_figure()
        
        # Save figure
        plt.savefig('fig_motion_planning_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_motion_planning_analysis.pdf', dpi=300, bbox_inches='tight')
        
        # Create enhanced table
        table = create_simple_enhanced_table()
        
        print("\n✅ SECTION V ENHANCEMENTS COMPLETE!")
        print("\n📁 Generated Files:")
        print("  • fig_motion_planning_analysis.png")
        print("  • fig_motion_planning_analysis.pdf") 
        print("  • enhanced_motion_control_table.tex")
        
        print("\n📋 Next Steps:")
        print("  1. Copy fig_motion_planning_analysis.pdf to IEEE Access directory")
        print("  2. Uncomment \\includegraphics line in Section V")
        print("  3. Consider adding enhanced table alongside existing one")
        
        plt.close('all')
        print("\n🎉 Ready to enhance Section V readability!")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install matplotlib numpy")
    except Exception as e:
        print(f"❌ Error: {e}")