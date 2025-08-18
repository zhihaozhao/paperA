#!/usr/bin/env python3
"""
Method 2: Python Matplotlib Professional Plotting
需要安装: pip install matplotlib pandas numpy seaborn
适用于Python环境的高质量IEEE期刊图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# IEEE IoTJ publication style setup
def setup_ieee_style():
    """Setup IEEE IoTJ compliant plotting style."""
    plt.style.use('default')  # Start with clean style
    
    # IEEE IoTJ parameters
    plt.rcParams.update({
        'figure.figsize': (6.73, 3.94),  # 17.1cm x 10cm in inches
        'figure.dpi': 300,
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'text.usetex': False  # For compatibility
    })

def create_figure3_cross_domain():
    """Generate Figure 3: D3 Cross-Domain Performance."""
    setup_ieee_style()
    
    # Load data
    with open('figure3_d3_cross_domain_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Process data
    models = ['enhanced', 'cnn', 'bilstm', 'conformer_lite']
    model_labels = ['Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite']
    
    loso_means = []
    loso_stds = []
    loro_means = []
    loro_stds = []
    
    for model in models:
        loso_row = next(row for row in data if row['protocol'] == 'LOSO' and row['model'] == model)
        loro_row = next(row for row in data if row['protocol'] == 'LORO' and row['model'] == model)
        
        loso_means.append(float(loso_row['mean_f1']))
        loso_stds.append(float(loso_row['std_f1']))
        loro_means.append(float(loro_row['mean_f1']))
        loro_stds.append(float(loro_row['std_f1']))
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(6.73, 3.94))  # IEEE IoTJ double column
    
    x = np.arange(len(models))
    width = 0.35
    
    # IEEE IoTJ color scheme
    loso_color = '#2E86AB'  # Blue
    loro_color = '#E84855'  # Red-orange
    
    # Create bars
    bars1 = ax.bar(x - width/2, loso_means, width, yerr=loso_stds,
                   label='LOSO', color=loso_color, alpha=0.8, 
                   capsize=3, error_kw={'linewidth': 0.5})
    bars2 = ax.bar(x + width/2, loro_means, width, yerr=loro_stds,
                   label='LORO', color=loro_color, alpha=0.8,
                   capsize=3, error_kw={'linewidth': 0.5})
    
    # Highlight Enhanced model (first bars)
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(1.5)
    bars2[0].set_edgecolor('black') 
    bars2[0].set_linewidth(1.5)
    
    # Formatting
    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Cross-Domain Generalization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + loso_stds[i] + 0.01,
                f'{loso_means[i]:.3f}±{loso_stds[i]:.3f}',
                ha='center', va='bottom', fontsize=7)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + loro_stds[i] + 0.01,
                f'{loro_means[i]:.3f}±{loro_stds[i]:.3f}',
                ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('figure3_d3_cross_domain_matplotlib.pdf', format='pdf', dpi=300)
    plt.close()
    
    return fig

def create_figure4_label_efficiency():
    """Generate Figure 4: D4 Label Efficiency Curve."""
    setup_ieee_style()
    plt.rcParams['figure.figsize'] = (6.73, 4.72)  # 17.1cm x 12cm
    
    # Load data
    with open('figure4_d4_label_efficiency_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Process data
    x_data = [float(row['label_ratio_percent']) for row in data]
    y_data = [float(row['mean_f1']) for row in data]
    errors = [float(row['std_f1']) for row in data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6.73, 4.72))
    
    # Efficient range background
    ax.axvspan(0, 20, alpha=0.2, color='lightgreen', label='Efficient Range (≤20%)')
    
    # Error ribbon
    ax.fill_between(x_data, np.array(y_data) - np.array(errors), 
                    np.array(y_data) + np.array(errors),
                    alpha=0.3, color='#2E86AB')
    
    # Main efficiency curve
    line = ax.plot(x_data, y_data, 'o-', color='#2E86AB', linewidth=2.5, 
                   markersize=8, markerfacecolor='white', 
                   markeredgecolor='#2E86AB', markeredgewidth=2,
                   label='Enhanced Fine-tune')
    
    # Target lines
    ax.axhline(y=0.80, color='#FF6B6B', linestyle='--', linewidth=1.5, 
               label='Target: 80% F1')
    ax.axhline(y=0.90, color='orange', linestyle=':', linewidth=1, 
               label='Ideal: 90% F1')
    
    # Key achievement annotation
    ax.annotate('Key Achievement:\n82.1% F1 @ 20% Labels',
                xy=(20, 0.821), xytext=(35, 0.87),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2),
                fontsize=10, ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFACD', 
                         edgecolor='#FF6B6B', linewidth=1))
    
    # Formatting
    ax.set_xlabel('Label Ratio (%)')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Sim2Real Label Efficiency Breakthrough')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    
    # Add data point labels
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        ax.text(x, y + errors[i] + 0.03, f'{y:.3f}',
                ha='center', va='bottom', fontsize=8, color='#2E86AB')
    
    plt.tight_layout()
    plt.savefig('figure4_d4_label_efficiency_matplotlib.pdf', format='pdf', dpi=300)
    plt.close()
    
    return fig

def main():
    """Main function to generate both figures."""
    print("📊 Python Matplotlib Method (Publication Quality)")
    print("=" * 60)
    
    try:
        # Try to import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("✅ Matplotlib available - generating publication figures...")
        
        # Generate figures
        fig3 = create_figure3_cross_domain()
        fig4 = create_figure4_label_efficiency()
        
        print("✅ Figure 3 saved: figure3_d3_cross_domain_matplotlib.pdf")
        print("✅ Figure 4 saved: figure4_d4_label_efficiency_matplotlib.pdf")
        
        print("\n💡 Matplotlib Method Advantages:")
        print("• IEEE IoTJ compliant output (300 DPI PDF)")
        print("• Professional scientific plotting")
        print("• Precise control over all elements")
        print("• Easy iteration and customization")
        
    except ImportError:
        print("⚠️  Matplotlib not available in current environment")
        print("💡 This script can be run in environments with matplotlib installed")
        print("   Installation: pip install matplotlib pandas numpy")
        print("\n📋 Script Features:")
        print("• IEEE IoTJ compliance (Times font, 300 DPI)")
        print("• Professional error bars and annotations") 
        print("• Color-blind friendly palette")
        print("• Publication-ready PDF export")

if __name__ == "__main__":
    main()