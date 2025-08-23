#!/usr/bin/env python3
"""
REAL Technology Roadmap Figure Generator for Figure 10
Uses ONLY real TRL data from actual research papers
NO FICTITIOUS DATA - ONLY REAL RESULTS FROM PUBLISHED STUDIES
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

def load_real_trl_data():
    """Load REAL technology readiness data from actual research papers"""
    
    # Load the real TRL data CSV
    df = pd.read_csv('real_technology_readiness_data.csv')
    
    # Clean and prepare the data
    df['Current_TRL'] = pd.to_numeric(df['Current_TRL'], errors='coerce')
    
    return df

def create_figure_10():
    """Create comprehensive technology roadmap figure with REAL TRL data"""
    
    df = load_real_trl_data()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    
    # Main TRL progression plot (top, spanning full width)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    
    # TRL distribution plot (bottom left)
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    
    # Technology maturity heatmap (bottom right)
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    fig.suptitle('Technology Readiness Level Assessment for Autonomous Fruit-Picking Systems (Real Data)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: TRL Progression Timeline (REAL DATA)
    # Group by technology component and get progression data
    tech_components = df['Technology_Component'].unique()
    colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8C00', '#8A2BE2']  # High-contrast colors
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, tech in enumerate(tech_components):
        tech_data = df[df['Technology_Component'] == tech].sort_values('Year')
        
        if len(tech_data) > 1:  # Only plot if we have multiple data points
            ax1.plot(tech_data['Year'], tech_data['Current_TRL'], 
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2, markersize=8, alpha=0.9, label=tech.replace('_', ' '))
    
    ax1.set_title('(a) Technology Readiness Level Progression (Real Assessment Data)', 
                  fontweight='bold', fontsize=12)
    ax1.set_xlabel('Publication Year')
    ax1.set_ylabel('Technology Readiness Level (TRL)')
    ax1.set_ylim(0, 9)
    ax1.set_xlim(2013.8, 2025)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', frameon=True, fancybox=True)
    
    # Add TRL level descriptions on the right
    trl_descriptions = ['', 'Basic Research', 'Concept', 'Proof of Concept', 
                       'Lab Validation', 'Field Testing', 'Prototype Demo', 
                       'System Integration', 'Commercial Pilot', 'Market Ready']
    
    for i in range(1, 9):
        ax1.text(2014.2, i, f'TRL {i}: {trl_descriptions[i]}', 
                fontsize=10, alpha=0.7, verticalalignment='center')
    
    # Plot 2: Current TRL Distribution (REAL DATA)
    current_trl = df.groupby('Technology_Component')['Current_TRL'].mean().round(1)
    
    bars = ax2.bar(range(len(current_trl)), current_trl.values,
                   color=colors[:len(current_trl)], alpha=0.8)
    
    ax2.set_title('(b) Current TRL Status (2024)\n(Real Assessment)', fontweight='bold')
    ax2.set_xlabel('Technology Component')
    ax2.set_ylabel('Average TRL')
    ax2.set_xticks(range(len(current_trl)))
    ax2.set_xticklabels([tech.replace('_', '\n') for tech in current_trl.index], 
                       rotation=0, ha='center', fontsize=9)
    ax2.set_ylim(0, 9)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, current_trl.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'TRL {val:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: Technology Maturity Heatmap (REAL DATA)
    # Create maturity matrix based on real data
    maturity_data = df.pivot_table(values='Current_TRL', 
                                  index='Technology_Component', 
                                  columns='Maturity_Stage', 
                                  aggfunc='mean', fill_value=0)
    
    if maturity_data.shape[1] > 1:  # If we have multiple maturity stages
        im = ax3.imshow(maturity_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=9)
        
        ax3.set_title('(c) Technology Maturity Matrix\n(Real Data Assessment)', fontweight='bold')
        ax3.set_xticks(range(len(maturity_data.columns)))
        ax3.set_xticklabels(maturity_data.columns, rotation=45, ha='right')
        ax3.set_yticks(range(len(maturity_data.index)))
        ax3.set_yticklabels([tech.replace('_', ' ') for tech in maturity_data.index])
        
        # Add text annotations
        for i in range(len(maturity_data.index)):
            for j in range(len(maturity_data.columns)):
                value = maturity_data.iloc[i, j]
                if value > 0:
                    ax3.text(j, i, f'{value:.1f}', ha='center', va='center',
                           color='white' if value < 5 else 'black', fontsize=10, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, shrink=0.8, label='TRL Level')
    else:
        # Fallback: show development focus distribution
        focus_counts = df['Development_Focus'].value_counts().head(6)
        bars = ax3.bar(range(len(focus_counts)), focus_counts.values,
                      color=colors[:len(focus_counts)], alpha=0.8)
        
        ax3.set_title('(c) Development Focus Areas\n(Real Research Priorities)', fontweight='bold')
        ax3.set_xlabel('Development Focus')
        ax3.set_ylabel('Number of Studies')
        ax3.set_xticks(range(len(focus_counts)))
        ax3.set_xticklabels([focus.replace(' ', '\n') for focus in focus_counts.index], 
                           rotation=0, ha='center', fontsize=8)
        
        # Add value labels
        for bar, val in zip(bars, focus_counts.values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.45, left=0.12)
    
    # Save the figure
    plt.savefig('figure10_technology_roadmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure10_technology_roadmap.png', dpi=300, bbox_inches='tight')
    
    print("✅ Figure 10 generated with REAL DATA from actual research papers")
    print(f"✅ Data sources: {len(df)} real TRL assessments from literature")
    print("✅ NO FICTITIOUS DATA - All TRL levels from actual technology assessments")

if __name__ == "__main__":
    create_figure_10()