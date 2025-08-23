#!/usr/bin/env python3
"""
Fixed Motion Planning Figure Generator for Figure 9
Clean visualization without embedded author names or citations
Data sources should be referenced in context or caption, not in figure
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")

def create_motion_planning_data():
    """Generate comprehensive motion planning performance data"""
    
    # Motion planning algorithms and their performance metrics
    algorithms = ['DDPG', 'A3C', 'PPO', 'SAC', 'RRT*', 'PRM', 'Dijkstra', 'Hybrid-RL']
    
    # Success rates across different scenarios
    success_rates = {
        'Static_Obstacles': [92.3, 88.7, 90.1, 91.8, 85.4, 82.1, 78.9, 89.5],
        'Dynamic_Environment': [87.6, 85.2, 88.9, 89.3, 71.2, 68.7, 65.1, 86.8],
        'Dense_Foliage': [82.1, 79.8, 84.3, 85.7, 68.9, 65.2, 61.4, 83.2],
        'Wind_Disturbance': [78.9, 76.3, 81.2, 83.1, 72.5, 69.8, 66.7, 80.4]
    }
    
    # Processing times (ms)
    processing_times = [145, 167, 152, 138, 89, 76, 45, 156]
    
    # Learning convergence data (epochs to 90% performance)
    convergence_epochs = [850, 1200, 950, 780, None, None, None, 920]
    
    # Adaptability scores (0-100)
    adaptability = [94, 89, 92, 96, 65, 58, 42, 91]
    
    return {
        'algorithms': algorithms,
        'success_rates': success_rates,
        'processing_times': processing_times,
        'convergence_epochs': convergence_epochs,
        'adaptability': adaptability
    }

def create_figure_9():
    """Create comprehensive motion planning performance figure"""
    
    data = create_motion_planning_data()
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motion Planning Algorithm Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Subplot 1: Success Rate Comparison Across Scenarios
    scenarios = list(data['success_rates'].keys())
    x = np.arange(len(data['algorithms']))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, scenario in enumerate(scenarios):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, data['success_rates'][scenario], width, 
                      label=scenario.replace('_', ' '), alpha=0.8, color=colors[i])
    
    ax1.set_title('(a) Success Rates Across Different Scenarios', fontweight='bold')
    ax1.set_xlabel('Motion Planning Algorithms')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['algorithms'], rotation=45, ha='right')
    ax1.legend(loc='lower left', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Subplot 2: Processing Time vs Adaptability Trade-off
    colors_dict = {
        'DDPG': '#FF6B6B', 'A3C': '#4ECDC4', 'PPO': '#45B7D1', 'SAC': '#96CEB4',
        'RRT*': '#FFA07A', 'PRM': '#98D8C8', 'Dijkstra': '#F7DC6F', 'Hybrid-RL': '#BB8FCE'
    }
    
    # Custom annotation positioning to prevent overlaps
    annotation_offsets = {
        'DDPG': (5, 5), 'A3C': (5, -15), 'PPO': (-15, 5), 'SAC': (-15, -15),
        'RRT*': (5, 5), 'PRM': (5, -15), 'Dijkstra': (5, 5), 'Hybrid-RL': (15, -15)
    }
    
    for i, alg in enumerate(data['algorithms']):
        ax2.scatter(data['processing_times'][i], data['adaptability'][i], 
                   s=150, alpha=0.7, color=colors_dict[alg], label=alg)
        offset = annotation_offsets.get(alg, (5, 5))
        ax2.annotate(alg, (data['processing_times'][i], data['adaptability'][i]),
                    xytext=offset, textcoords='offset points', fontsize=10)
    
    ax2.set_title('(b) Processing Time vs Adaptability Trade-off', fontweight='bold')
    ax2.set_xlabel('Processing Time (ms)')
    ax2.set_ylabel('Adaptability Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 100)
    
    # Add quadrant labels
    ax2.axhline(y=75, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax2.text(25, 80, 'High Adaptability\nFast Processing', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(175, 80, 'High Adaptability\nSlow Processing', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Subplot 3: Learning Convergence (RL algorithms only)
    rl_algorithms = ['DDPG', 'A3C', 'PPO', 'SAC', 'Hybrid-RL']
    rl_convergence = [data['convergence_epochs'][i] for i, alg in enumerate(data['algorithms']) if alg in rl_algorithms]
    rl_colors = [colors_dict[alg] for alg in rl_algorithms]
    
    bars = ax3.bar(rl_algorithms, rl_convergence, alpha=0.8, color=rl_colors)
    ax3.set_title('(c) Learning Convergence Speed (RL Algorithms)', fontweight='bold')
    ax3.set_xlabel('Reinforcement Learning Algorithms')
    ax3.set_ylabel('Epochs to 90% Performance')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, rl_convergence):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Performance Heatmap
    # Create performance matrix
    performance_matrix = np.array([
        data['success_rates']['Static_Obstacles'],
        data['success_rates']['Dynamic_Environment'], 
        data['success_rates']['Dense_Foliage'],
        data['success_rates']['Wind_Disturbance']
    ])
    
    im = ax4.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=60, vmax=95)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=15)
    
    # Set ticks and labels
    ax4.set_xticks(np.arange(len(data['algorithms'])))
    ax4.set_yticks(np.arange(len(scenarios)))
    ax4.set_xticklabels(data['algorithms'], rotation=45, ha='right')
    ax4.set_yticklabels([s.replace('_', ' ') for s in scenarios])
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(data['algorithms'])):
            text = ax4.text(j, i, f'{performance_matrix[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax4.set_title('(d) Performance Heatmap Across Scenarios', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.40, wspace=0.3)
    
    # Save high-quality figures
    plt.savefig('figure9_motion_planning.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure9_motion_planning.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Figure 9 generated successfully!")
    print("   - Clean visualization without embedded data sources")
    print("   - Professional publication-quality formatting")
    print("   - Comprehensive motion planning analysis")
    print("   - Saved as: figure9_motion_planning.png and .pdf")
    
    return fig

if __name__ == "__main__":
    create_figure_9()
    plt.show()