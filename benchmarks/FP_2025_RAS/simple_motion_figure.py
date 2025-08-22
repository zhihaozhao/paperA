#!/usr/bin/env python3
"""
Simple Motion Planning Figure Generator - No External Dependencies
Creates motion planning analysis figure for Section V using hardcoded data
"""

try:
    import matplotlib.pyplot as plt
    import numpy as np
    print("✅ Successfully imported matplotlib and numpy")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install with: pip install matplotlib numpy")
    exit(1)

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

def create_motion_planning_figure():
    """Create 4-panel motion planning analysis figure"""
    
    print("🤖 Creating Motion Planning Analysis Figure...")
    
    # Create figure with good spacing
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Motion Planning Analysis for Autonomous Fruit Harvesting\n(16 Studies, 2015-2024)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Panel A: Algorithm Performance Comparison
    ax1 = axes[0, 0]
    
    algorithms = ['Classical\n(A*, RRT)', 'Vision-based\n(Servo)', 'Deep RL\n(DDPG)', 'Multi-robot\n(Coord)', 'Hybrid\n(Adaptive)']
    success_rates = [70.8, 73.1, 90.4, 70.0, 75.0]
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']
    
    bars = ax1.bar(range(len(algorithms)), success_rates, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, fontsize=10)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('(A) Algorithm Family Performance', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Cycle Time vs Success Rate Trade-off
    ax2 = axes[0, 1]
    
    # Real data from literature
    studies = ['Silwal', 'Arad', 'Xiong', 'Williams', 'Lehnert', 'Ling', 'Lin', 'Verbiest', 'Zhang']
    cycle_times = [7.6, 24.0, 6.1, 8.0, 12.0, 8.0, 5.0, 5.0, 5.0]
    success_rates_scatter = [84, 39.5, 75, 70, 58, 87.5, 90.9, 92, 88]
    
    # Color by algorithm type
    algorithm_types = ['Classical', 'Hybrid', 'Classical', 'Classical', 'Classical', 
                      'Vision', 'RL', 'RL', 'Deep RL']
    type_colors = {'Classical': '#E74C3C', 'Hybrid': '#F39C12', 'Vision': '#3498DB', 
                   'RL': '#27AE60', 'Deep RL': '#2ECC71'}
    point_colors = [type_colors[algo] for algo in algorithm_types]
    
    scatter = ax2.scatter(cycle_times, success_rates_scatter, 
                         c=point_colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add study labels
    for i, study in enumerate(studies):
        ax2.annotate(study, (cycle_times[i], success_rates_scatter[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)
    
    # Add performance zones
    ax2.axhspan(80, 100, alpha=0.2, color='green', label='High Performance Zone')
    ax2.axhspan(60, 80, alpha=0.2, color='yellow', label='Moderate Performance')
    ax2.axhspan(0, 60, alpha=0.2, color='red', label='Low Performance')
    
    ax2.set_xlabel('Cycle Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('(B) Performance vs Speed Trade-offs', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 26)
    ax2.set_ylim(30, 95)
    
    # Panel C: Algorithm Evolution Timeline  
    ax3 = axes[1, 0]
    
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
    classical_studies = [0, 1, 2, 1, 3, 3, 0, 1, 1, 0]
    rl_studies = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    vision_studies = [0, 0, 1, 0, 2, 2, 0, 0, 0, 0]
    multi_robot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    
    # Stacked area plot
    ax3.fill_between(years, 0, classical_studies, color='#E74C3C', alpha=0.7, label='Classical Methods')
    ax3.fill_between(years, classical_studies, np.array(classical_studies) + np.array(rl_studies), 
                    color='#27AE60', alpha=0.7, label='RL-based Methods')
    ax3.fill_between(years, np.array(classical_studies) + np.array(rl_studies),
                    np.array(classical_studies) + np.array(rl_studies) + np.array(vision_studies),
                    color='#3498DB', alpha=0.7, label='Vision-based')
    
    ax3.set_xlabel('Publication Year', fontweight='bold')
    ax3.set_ylabel('Number of Studies', fontweight='bold')
    ax3.set_title('(C) Algorithm Evolution Timeline', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2014, 2025)
    
    # Panel D: Environment Performance Analysis
    ax4 = axes[1, 1]
    
    environments = ['Lab/\nSimulation', 'Greenhouse/\nProtected', 'Field/\nOrchard', 'Commercial/\nReal-world']
    env_performance = [85, 75, 65, 70]
    env_counts = [4, 5, 4, 3]
    
    bars = ax4.bar(range(len(environments)), env_performance, 
                  color=['#9B59B6', '#3498DB', '#F39C12', '#27AE60'], alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    # Add count labels on bars
    for i, (bar, count, perf) in enumerate(zip(bars, env_counts, env_performance)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{perf:.0f}%', ha='center', va='center', fontweight='bold', 
                color='white', fontsize=12)
    
    ax4.set_xticks(range(len(environments)))
    ax4.set_xticklabels(environments, fontsize=11)
    ax4.set_ylabel('Average Success Rate (%)', fontweight='bold')
    ax4.set_title('(D) Performance by Environment', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # Improve overall layout spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    plt.subplots_adjust(hspace=0.4, wspace=0.35)
    
    return fig

def main():
    """Generate motion planning figure"""
    
    try:
        print("🚀 Starting Motion Planning Figure Generation...")
        
        # Create figure
        fig = create_motion_planning_figure()
        
        # Save in multiple formats
        plt.savefig('fig_motion_planning_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.savefig('fig_motion_planning_analysis.pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        print("\n✅ SUCCESS! Motion Planning Figure Generated:")
        print("  📁 fig_motion_planning_analysis.png")
        print("  📁 fig_motion_planning_analysis.pdf")
        
        print("\n📋 Integration Steps:")
        print("  1. Copy fig_motion_planning_analysis.pdf to IEEE Access directory")
        print("  2. Uncomment \\includegraphics line in Section V")
        print("  3. Compile LaTeX document")
        
        # Close to free memory
        plt.close('all')
        
        print("\n🎉 Ready to enhance Section V readability!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🤖 Motion planning figure generation completed successfully!")
    else:
        print("\n💡 Please check dependencies: pip install matplotlib numpy")