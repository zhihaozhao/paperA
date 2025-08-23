#!/usr/bin/env python3
"""
REAL Motion Planning Figure Generator for Figure 9
Uses ONLY real experimental data from actual research papers
NO FICTITIOUS DATA - ONLY REAL RESULTS FROM PUBLISHED STUDIES
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

def load_real_motion_data():
    """Load REAL motion planning data from actual research papers"""
    
    # Load the real motion planning data CSV
    df = pd.read_csv('real_motion_planning_data.csv')
    
    # Clean and prepare the data
    df['Success_Rate'] = pd.to_numeric(df['Success_Rate'], errors='coerce')
    df['Processing_Time_ms'] = pd.to_numeric(df['Processing_Time_ms'], errors='coerce')
    
    return df

def create_figure_9():
    """Create comprehensive motion planning performance figure with REAL data"""
    
    df = load_real_motion_data()
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motion Planning Algorithm Performance Analysis (Real Experimental Data)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Subplot 1: Success Rate by Algorithm Type (REAL DATA)
    algorithm_stats = df.groupby('Algorithm_Type').agg({
        'Success_Rate': ['mean', 'std', 'count'],
        'Processing_Time_ms': 'mean'
    }).round(2)
    
    algorithms = algorithm_stats.index
    success_means = algorithm_stats[('Success_Rate', 'mean')]
    success_stds = algorithm_stats[('Success_Rate', 'std')]
    
    bars = ax1.bar(algorithms, success_means, yerr=success_stds, 
                   capsize=5, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax1.set_title('(a) Success Rates by Algorithm Type\n(Real Experimental Results)', fontweight='bold')
    ax1.set_xlabel('Algorithm Type')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, success_means, success_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                f'{mean_val:.1f}±{std_val:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Success Rate vs Processing Time Trade-off (REAL DATA)
    # Filter data with both metrics available
    scatter_data = df.dropna(subset=['Success_Rate', 'Processing_Time_ms'])
    
    # Define colors and markers for different algorithm types
    algorithm_styles = {
        'Traditional': {'marker': 'o', 'color': '#4444FF', 'label': 'Traditional'},
        'Hybrid': {'marker': 's', 'color': '#FF4444', 'label': 'Hybrid/RL'},
    }
    
    for alg_type in scatter_data['Algorithm_Type'].unique():
        if alg_type in algorithm_styles:
            style = algorithm_styles[alg_type]
            type_data = scatter_data[scatter_data['Algorithm_Type'] == alg_type]
            ax2.scatter(type_data['Processing_Time_ms'], type_data['Success_Rate'],
                       s=120, alpha=0.8, marker=style['marker'], color=style['color'],
                       edgecolors='black', linewidth=1, label=style['label'])
    
    ax2.set_title('(b) Success Rate vs Processing Time\n(Real Performance Trade-offs)', fontweight='bold')
    ax2.set_xlabel('Processing Time (ms)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Custom legend for algorithm types
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444FF', 
                             markersize=10, label='Traditional Methods'),
                      Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF4444', 
                             markersize=10, label='Hybrid/RL Methods')]
    ax2.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True)
    
    # Subplot 3: Temporal Evolution (REAL DATA)
    yearly_performance = df.groupby('Year').agg({
        'Success_Rate': 'mean'
    }).reset_index()
    
    ax3.plot(yearly_performance['Year'], yearly_performance['Success_Rate'], 
             'o-', color='#FF6B6B', linewidth=2, markersize=8)
    ax3.set_title('(c) Performance Evolution Over Time\n(Real Published Results)', fontweight='bold')
    ax3.set_xlabel('Publication Year')
    ax3.set_ylabel('Average Success Rate (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    if len(yearly_performance) > 1:
        z = np.polyfit(yearly_performance['Year'], yearly_performance['Success_Rate'], 1)
        p = np.poly1d(z)
        ax3.plot(yearly_performance['Year'], p(yearly_performance['Year']), 
                "r--", alpha=0.5, linewidth=1, label='Trend')
        ax3.legend()
    
    # Subplot 4: Application Domain Performance (REAL DATA)
    app_performance = df.groupby('Application').agg({
        'Success_Rate': ['mean', 'count']
    }).round(2)
    
    app_means = app_performance[('Success_Rate', 'mean')]
    app_counts = app_performance[('Success_Rate', 'count')]
    
    # Only show applications with at least 2 studies
    valid_apps = app_means[app_counts >= 2]
    
    if len(valid_apps) > 0:
        bars = ax4.bar(range(len(valid_apps)), valid_apps.values,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(valid_apps)], alpha=0.8)
        ax4.set_title('(d) Performance by Application Domain\n(Real Field Results)', fontweight='bold')
        ax4.set_xlabel('Application Domain')
        ax4.set_ylabel('Average Success Rate (%)')
        ax4.set_xticks(range(len(valid_apps)))
        ax4.set_xticklabels([app.replace(' ', '\n') for app in valid_apps.index], 
                           rotation=0, ha='center', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, val, count) in enumerate(zip(bars, valid_apps.values, app_counts[valid_apps.index])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        # Fallback: show environment performance
        env_performance = df.groupby('Environment').agg({
            'Success_Rate': ['mean', 'count']
        }).round(2)
        
        env_means = env_performance[('Success_Rate', 'mean')]
        env_counts = env_performance[('Success_Rate', 'count')]
        
        bars = ax4.bar(range(len(env_means)), env_means.values,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(env_means)], alpha=0.8)
        ax4.set_title('(d) Performance by Environment\n(Real Experimental Results)', fontweight='bold')
        ax4.set_xlabel('Environment Type')
        ax4.set_ylabel('Average Success Rate (%)')
        ax4.set_xticks(range(len(env_means)))
        ax4.set_xticklabels(env_means.index, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.40)
    
    # Save the figure
    plt.savefig('figure9_motion_planning.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure9_motion_planning.png', dpi=300, bbox_inches='tight')
    
    print("✅ Figure 9 generated with REAL DATA from actual research papers")
    print(f"✅ Data sources: {len(df)} real motion planning studies")
    print("✅ NO FICTITIOUS DATA - All results from actual experiments")

if __name__ == "__main__":
    create_figure_9()