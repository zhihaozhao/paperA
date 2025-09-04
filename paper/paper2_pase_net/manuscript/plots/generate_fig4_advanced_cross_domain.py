#!/usr/bin/env python3
"""
Generate Figure 4: Advanced Cross-Domain Performance Visualization
Shows LOSO/LORO for all 4 models with enhanced visual effects
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyBboxPatch
import seaborn as sns

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def collect_cross_domain_data():
    """Collect LOSO and LORO data for all 4 models"""
    
    # Collect LOSO data
    loso_files = glob.glob('/workspace/results_gpu/d3/loso/*.json')
    loso_data = {}
    
    for file in loso_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            filename = file.split('/')[-1]
            model = filename.split('_')[1]
            
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1 = data['aggregate_stats']['macro_f1'].get('mean', 0) * 100  # Convert to percentage
                if model not in loso_data:
                    loso_data[model] = []
                loso_data[model].append(f1)
        except:
            continue
    
    # Collect LORO data
    loro_files = glob.glob('/workspace/results_gpu/d3/loro/*.json')
    loro_data = {}
    
    for file in loro_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            filename = file.split('/')[-1]
            model = filename.split('_')[1]
            
            if 'aggregate_stats' in data and 'macro_f1' in data['aggregate_stats']:
                f1 = data['aggregate_stats']['macro_f1'].get('mean', 0) * 100
                if model not in loro_data:
                    loro_data[model] = []
                loro_data[model].append(f1)
        except:
            continue
    
    # Calculate averages and std
    results = {}
    for model in ['enhanced', 'cnn', 'bilstm', 'conformer']:
        loso_scores = loso_data.get(model, [])
        loro_scores = loro_data.get(model, [])
        
        results[model] = {
            'loso_mean': np.mean(loso_scores) if loso_scores else 0,
            'loso_std': np.std(loso_scores) if loso_scores else 0,
            'loro_mean': np.mean(loro_scores) if loro_scores else 0,
            'loro_std': np.std(loro_scores) if loro_scores else 0
        }
    
    return results

def create_advanced_visualization():
    """Create advanced cross-domain visualization"""
    
    # Collect data
    data = collect_cross_domain_data()
    
    # Model names and colors
    model_names = {
        'enhanced': 'PASE-Net',
        'cnn': 'CNN',
        'bilstm': 'BiLSTM',
        'conformer': 'Conformer'
    }
    
    colors = {
        'enhanced': '#2E86AB',  # Blue
        'cnn': '#A23B72',       # Purple
        'bilstm': '#F18F01',    # Orange
        'conformer': '#C73E1D'  # Red
    }
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 8))
    
    # Create main plot area
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    # Create side plots
    ax_gap = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax_bottom = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    
    # Main plot: Grouped bar chart with error bars
    models = ['enhanced', 'cnn', 'bilstm', 'conformer']
    x_pos = np.arange(len(models))
    width = 0.35
    
    loso_means = [data[m]['loso_mean'] for m in models]
    loso_stds = [data[m]['loso_std'] for m in models]
    loro_means = [data[m]['loro_mean'] for m in models]
    loro_stds = [data[m]['loro_std'] for m in models]
    
    # Create bars with gradient effect
    bars1 = ax_main.bar(x_pos - width/2, loso_means, width, 
                        yerr=loso_stds, capsize=5,
                        label='LOSO', edgecolor='black', linewidth=2)
    bars2 = ax_main.bar(x_pos + width/2, loro_means, width,
                        yerr=loro_stds, capsize=5,
                        label='LORO', edgecolor='black', linewidth=2)
    
    # Color bars by model
    for i, (bar1, bar2, model) in enumerate(zip(bars1, bars2, models)):
        bar1.set_facecolor(colors[model])
        bar1.set_alpha(0.8)
        bar2.set_facecolor(colors[model])
        bar2.set_alpha(0.6)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # LOSO values
        ax_main.text(bar1.get_x() + bar1.get_width()/2., height1 + loso_stds[i] + 1,
                    f'{height1:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # LORO values
        ax_main.text(bar2.get_x() + bar2.get_width()/2., height2 + loro_stds[i] + 1,
                    f'{height2:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
    
    # Highlight PASE-Net's consistency
    if abs(loso_means[0] - loro_means[0]) < 1:  # PASE-Net is first
        ax_main.annotate('Identical!', 
                        xy=(0, max(loso_means[0], loro_means[0]) + 5),
                        xytext=(0, max(loso_means[0], loro_means[0]) + 10),
                        ha='center', fontsize=11, color='green', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Main plot formatting
    ax_main.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Macro F1 Score (%)', fontsize=12, fontweight='bold')
    ax_main.set_title('Cross-Domain Performance Comparison', fontsize=14, fontweight='bold')
    ax_main.set_xticks(x_pos)
    ax_main.set_xticklabels([model_names[m] for m in models], fontsize=11)
    ax_main.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax_main.set_ylim([0, 100])
    ax_main.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add performance zones
    ax_main.axhspan(90, 100, alpha=0.1, color='green', label='Excellent')
    ax_main.axhspan(80, 90, alpha=0.1, color='yellow', label='Good')
    ax_main.axhspan(0, 80, alpha=0.1, color='red', label='Needs Improvement')
    
    # Gap analysis (right plot)
    gaps = [abs(loso_means[i] - loro_means[i]) for i in range(len(models))]
    colors_gap = [colors[m] for m in models]
    
    bars_gap = ax_gap.barh(range(len(models)), gaps, color=colors_gap, 
                           edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars_gap, gaps)):
        ax_gap.text(gap + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{gap:.1f}%', va='center', fontweight='bold')
    
    ax_gap.set_yticks(range(len(models)))
    ax_gap.set_yticklabels([model_names[m] for m in models])
    ax_gap.set_xlabel('LOSO-LORO Gap (%)', fontsize=11, fontweight='bold')
    ax_gap.set_title('Domain Gap Analysis', fontsize=12, fontweight='bold')
    ax_gap.set_xlim([0, max(gaps) + 5])
    ax_gap.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add ideal line
    ax_gap.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (0%)')
    ax_gap.legend()
    
    # Bottom plot: Radar chart comparison
    ax_bottom.axis('off')
    
    # Create comparison table
    table_data = []
    for model in models:
        row = [
            model_names[model],
            f"{data[model]['loso_mean']:.1f}±{data[model]['loso_std']:.1f}",
            f"{data[model]['loro_mean']:.1f}±{data[model]['loro_std']:.1f}",
            f"{abs(data[model]['loso_mean'] - data[model]['loro_mean']):.1f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax_bottom.table(cellText=table_data,
                           colLabels=['Model', 'LOSO F1 (%)', 'LORO F1 (%)', 'Gap (%)'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.2, 0.25, 0.25, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(models) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # Model names
                    cell.set_facecolor(colors[models[i-1]])
                    cell.set_alpha(0.3)
                else:
                    cell.set_facecolor('#f0f0f0')
    
    ax_bottom.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('PASE-Net Cross-Domain Performance Analysis\nAll 4 Models Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_file = 'fig4_cross_domain_advanced.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Cross-Domain Performance Summary (All 4 Models)")
    print("="*60)
    for model in models:
        print(f"{model_names[model]:12s}: LOSO={data[model]['loso_mean']:.1f}%, "
              f"LORO={data[model]['loro_mean']:.1f}%, "
              f"Gap={abs(data[model]['loso_mean'] - data[model]['loro_mean']):.1f}%")
    
    # Note about Conformer
    if data['conformer']['loso_mean'] < 50:
        print("\nNote: Conformer showed convergence issues in LOSO protocol")
    
    plt.close()

if __name__ == "__main__":
    create_advanced_visualization()