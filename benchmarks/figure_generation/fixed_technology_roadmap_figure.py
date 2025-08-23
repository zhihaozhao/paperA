#!/usr/bin/env python3
"""
Fixed Technology Roadmap Figure Generator for Figure 10
Clean TRL visualization without embedded citations
TRL definitions should be explained in context, not embedded in figure
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.dates import YearLocator, DateFormatter
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")

def create_trl_data():
    """Create technology readiness level progression data for fruit-picking systems"""
    
    # Technology components and their TRL progression over time
    technologies = {
        'Computer Vision': {
            'years': [2015, 2017, 2019, 2021, 2023, 2024],
            'trl_levels': [3, 4, 5, 6, 7, 8],
            'milestones': ['Basic algorithms', 'Lab validation', 'Field testing', 'Prototype demo', 'System integration', 'Commercial pilot']
        },
        'Motion Planning': {
            'years': [2015, 2017, 2019, 2021, 2023, 2024],
            'trl_levels': [2, 3, 4, 5, 6, 7],
            'milestones': ['Concept formulation', 'Proof of concept', 'Lab validation', 'Field testing', 'Prototype demo', 'System integration']
        },
        'End-Effector Design': {
            'years': [2015, 2017, 2019, 2021, 2023, 2024],
            'trl_levels': [4, 5, 6, 7, 8, 8],
            'milestones': ['Lab validation', 'Field testing', 'Prototype demo', 'System integration', 'Commercial pilot', 'Market ready']
        },
        'Sensor Fusion': {
            'years': [2015, 2017, 2019, 2021, 2023, 2024],
            'trl_levels': [2, 3, 3, 4, 5, 6],
            'milestones': ['Concept formulation', 'Proof of concept', 'Extended validation', 'Lab validation', 'Field testing', 'Prototype demo']
        },
        'AI/ML Integration': {
            'years': [2015, 2017, 2019, 2021, 2023, 2024],
            'trl_levels': [1, 3, 4, 6, 7, 8],
            'milestones': ['Basic research', 'Proof of concept', 'Lab validation', 'Prototype demo', 'System integration', 'Commercial pilot']
        }
    }
    
    return technologies

def create_figure_10():
    """Create comprehensive technology roadmap figure with TRL progression"""
    
    data = create_trl_data()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    
    # Main TRL progression plot (top, spanning full width)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    
    # TRL distribution plot (bottom left)
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    
    # Technology maturity heatmap (bottom right)
    ax3 = plt.subplot2grid((3, 2), (2, 1))
    
    fig.suptitle('Technology Readiness Level Roadmap for Autonomous Fruit-Picking Systems', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: TRL Progression Timeline
    colors = ['#FF0000', '#0000FF', '#00AA00', '#FF8C00', '#8A2BE2']  # Red, Blue, Green, Orange, Purple
    line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for better distinction
    markers = ['o', 's', '^', 'D', 'v']  # Different markers for each line
    
    for i, (tech, tech_data) in enumerate(data.items()):
        ax1.plot(tech_data['years'], tech_data['trl_levels'], 
                marker=markers[i], linewidth=3, markersize=8, linestyle=line_styles[i],
                label=tech, color=colors[i], alpha=0.9)
        
        # Add milestone annotations at key points (only for select technologies to reduce clutter)
        if tech in ['Computer Vision', 'Motion Planning', 'AI/ML Integration']:
            for j, (year, trl, milestone) in enumerate(zip(tech_data['years'][::3], 
                                                          tech_data['trl_levels'][::3], 
                                                          tech_data['milestones'][::3])):
                if j < 2:  # Only first 2 annotations per technology
                    ax1.annotate(milestone, 
                               xy=(year, trl), 
                               xytext=(10, 15), 
                               textcoords='offset points',
                               fontsize=9, alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=colors[i], alpha=0.3))
    
    # Add TRL level descriptions as horizontal lines
    trl_descriptions = {
        1: 'Basic Research',
        2: 'Concept Formulation', 
        3: 'Proof of Concept',
        4: 'Lab Validation',
        5: 'Field Testing',
        6: 'Prototype Demo',
        7: 'System Integration',
        8: 'Commercial Pilot',
        9: 'Market Ready'
    }
    
    for trl_level, description in trl_descriptions.items():
        ax1.axhline(y=trl_level, color='gray', linestyle='--', alpha=0.3)
        ax1.text(2014.2, trl_level, f'TRL {trl_level}', 
                rotation=0, va='center', ha='right', fontsize=10, fontweight='bold')
    
    ax1.set_title('(a) Technology Readiness Level Progression (2015-2024)', fontweight='bold', pad=20)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Technology Readiness Level (TRL)')
    ax1.set_xlim(2013.8, 2024.5)
    ax1.set_ylim(0.5, 9.5)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Current TRL Distribution (2024)
    current_trls = [data[tech]['trl_levels'][-1] for tech in data.keys()]
    tech_names = list(data.keys())
    
    bars = ax2.barh(tech_names, current_trls, color=colors, alpha=0.8)
    ax2.set_title('(b) Current TRL Status (2024)', fontweight='bold')
    ax2.set_xlabel('Technology Readiness Level')
    ax2.set_xlim(0, 9)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, trl in zip(bars, current_trls):
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'TRL {trl}', ha='left', va='center', fontweight='bold')
    
    # Plot 3: Technology Maturity Heatmap
    # Create maturity matrix (years vs technologies)
    years = list(range(2015, 2025))
    tech_matrix = np.zeros((len(data), len(years)))
    
    for i, (tech, tech_data) in enumerate(data.items()):
        for year, trl in zip(tech_data['years'], tech_data['trl_levels']):
            year_idx = year - 2015
            tech_matrix[i, year_idx] = trl
    
    # Fill in missing years with interpolation
    for i in range(len(data)):
        non_zero_indices = np.nonzero(tech_matrix[i, :])[0]
        if len(non_zero_indices) > 1:
            tech_matrix[i, :] = np.interp(range(len(years)), 
                                        non_zero_indices, 
                                        tech_matrix[i, non_zero_indices])
    
    im = ax3.imshow(tech_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('TRL Level', rotation=270, labelpad=15)
    
    # Set ticks and labels
    ax3.set_xticks(range(len(years)))
    ax3.set_yticks(range(len(data)))
    ax3.set_xticklabels(years, rotation=45)
    ax3.set_yticklabels(tech_names)
    
    # Add text annotations
    for i in range(len(data)):
        for j in range(len(years)):
            if tech_matrix[i, j] > 0:
                            text = ax3.text(j, i, f'{int(tech_matrix[i, j])}',
                           ha="center", va="center", color="black", 
                           fontweight='bold', fontsize=10)
    
    ax3.set_title('(c) Technology Maturity Evolution', fontweight='bold')
    ax3.set_xlabel('Year')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.85, hspace=0.45, left=0.12)
    
    # Save high-quality figures
    plt.savefig('figure10_technology_roadmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('figure10_technology_roadmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Figure 10 generated successfully!")
    print("   - Clear TRL progression visualization")
    print("   - No embedded citations or author names")
    print("   - Professional technology roadmap format")
    print("   - TRL definitions should be explained in paper context")
    print("   - Saved as: figure10_technology_roadmap.png and .pdf")
    
    return fig

if __name__ == "__main__":
    create_figure_10()
    plt.show()