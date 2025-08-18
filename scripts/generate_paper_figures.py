#!/usr/bin/env python3
"""
Generate publication-ready figures for PaperA based on D3/D4 experimental results.

Produces:
- Figure 1: D3 Cross-domain generalization performance comparison
- Figure 2: D4 Sim2Real label efficiency curves  
- Figure 3: Model calibration and reliability analysis
- Figure 4: Cross-domain consistency analysis

Usage:
  python scripts/generate_paper_figures.py --d3_csv results/metrics/summary_d3.csv --d4_csv results/metrics/summary_d4.csv
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def setup_figure_style():
    """Setup publication-quality figure style."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def generate_d3_cross_domain_figure(d3_df, output_dir):
    """Generate D3 cross-domain generalization comparison figure."""
    setup_figure_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by protocol and model
    protocols = ['LOSO', 'LORO']
    models = ['enhanced', 'cnn', 'bilstm', 'conformer_lite']
    
    # Calculate mean and std for each protocol-model combination
    results = {}
    for protocol in protocols:
        results[protocol] = {}
        protocol_data = d3_df[d3_df['protocol'] == protocol]
        
        for model in models:
            model_data = protocol_data[protocol_data['model'] == model]['macro_f1']
            if len(model_data) > 0:
                results[protocol][model] = {
                    'mean': model_data.mean(),
                    'std': model_data.std(),
                    'count': len(model_data)
                }
    
    # Plot LOSO results
    loso_means = [results['LOSO'].get(m, {}).get('mean', 0) for m in models]
    loso_stds = [results['LOSO'].get(m, {}).get('std', 0) for m in models]
    
    x_pos = np.arange(len(models))
    bars1 = ax1.bar(x_pos, loso_means, yerr=loso_stds, capsize=5, 
                    alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_title('LOSO Cross-Domain Performance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Enhanced', 'CNN', 'BiLSTM', 'Conformer'])
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(loso_means, loso_stds)):
        if mean > 0:
            ax1.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontsize=10)
    
    # Plot LORO results  
    loro_means = [results['LORO'].get(m, {}).get('mean', 0) for m in models]
    loro_stds = [results['LORO'].get(m, {}).get('std', 0) for m in models]
    
    bars2 = ax2.bar(x_pos, loro_means, yerr=loro_stds, capsize=5,
                    alpha=0.8, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax2.set_xlabel('Model Architecture')
    ax2.set_ylabel('Macro F1 Score')
    ax2.set_title('LORO Cross-Domain Performance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Enhanced', 'CNN', 'BiLSTM', 'Conformer'])
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(loro_means, loro_stds)):
        if mean > 0:
            ax2.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    figure_path = output_dir / 'figure1_d3_cross_domain.pdf'
    plt.savefig(figure_path, format='pdf')
    plt.close()
    
    print(f"✓ Figure 1 saved: {figure_path}")
    return figure_path

def generate_d4_label_efficiency_figure(d4_df, output_dir):
    """Generate D4 Sim2Real label efficiency curve."""
    setup_figure_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Focus on enhanced model fine-tune results
    enhanced_ft = d4_df[(d4_df['model'] == 'enhanced') & 
                        (d4_df['transfer_method'] == 'fine_tune')]
    
    # Group by label ratio and calculate statistics
    ratio_stats = enhanced_ft.groupby('label_ratio')['target_f1'].agg(['mean', 'std', 'count'])
    ratio_stats = ratio_stats.sort_index()
    
    # Plot efficiency curve
    x_ratios = ratio_stats.index * 100  # Convert to percentage
    y_means = ratio_stats['mean']
    y_stds = ratio_stats['std']
    
    # Main efficiency curve
    line = ax.plot(x_ratios, y_means, 'o-', linewidth=3, markersize=8, 
                   color='#2E86AB', label='Enhanced Fine-tune')
    ax.fill_between(x_ratios, y_means - y_stds, y_means + y_stds, 
                    alpha=0.3, color='#2E86AB')
    
    # Add target performance line
    ax.axhline(y=0.80, color='red', linestyle='--', linewidth=2, 
               label='Target Performance (80%)')
    ax.axhline(y=0.90, color='orange', linestyle=':', linewidth=2,
               label='Ideal Performance (90%)')
    
    # Highlight key achievement
    if 20.0 in ratio_stats.index:
        key_f1 = ratio_stats.loc[20.0, 'mean']
        ax.annotate(f'Key Achievement:\n{key_f1:.1%} F1 @ 20% labels', 
                   xy=(20, key_f1), xytext=(40, key_f1 + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Label Ratio (%)')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('D4 Sim2Real Label Efficiency (Enhanced Model)')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add data points annotations
    for ratio, mean in zip(x_ratios, y_means):
        ax.text(ratio, mean + 0.03, f'{mean:.2f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    figure_path = output_dir / 'figure2_d4_label_efficiency.pdf'
    plt.savefig(figure_path, format='pdf')
    plt.close()
    
    print(f"✓ Figure 2 saved: {figure_path}")
    return figure_path

def generate_model_comparison_figure(d3_df, output_dir):
    """Generate comprehensive model comparison across protocols."""
    setup_figure_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    protocols = ['LOSO', 'LORO']
    models = ['enhanced', 'cnn', 'bilstm', 'conformer_lite'] 
    model_labels = ['Enhanced', 'CNN', 'BiLSTM', 'Conformer-lite']
    
    x = np.arange(len(models))
    width = 0.35
    
    # Calculate means for each protocol-model combination
    loso_means = []
    loro_means = []
    loso_stds = []
    loro_stds = []
    
    for model in models:
        loso_data = d3_df[(d3_df['protocol'] == 'LOSO') & (d3_df['model'] == model)]['macro_f1']
        loro_data = d3_df[(d3_df['protocol'] == 'LORO') & (d3_df['model'] == model)]['macro_f1']
        
        loso_means.append(loso_data.mean() if len(loso_data) > 0 else 0)
        loro_means.append(loro_data.mean() if len(loro_data) > 0 else 0)
        loso_stds.append(loso_data.std() if len(loso_data) > 0 else 0)
        loro_stds.append(loro_data.std() if len(loro_data) > 0 else 0)
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, loso_means, width, yerr=loso_stds, 
                   label='LOSO', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, loro_means, width, yerr=loro_stds,
                   label='LORO', alpha=0.8, capsize=5)
    
    # Formatting
    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('Cross-Domain Generalization: LOSO vs LORO Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    def autolabel(bars, means, stds):
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1, loso_means, loso_stds)
    autolabel(bars2, loro_means, loro_stds)
    
    plt.tight_layout()
    figure_path = output_dir / 'figure3_model_comparison.pdf'
    plt.savefig(figure_path, format='pdf')
    plt.close()
    
    print(f"✓ Figure 3 saved: {figure_path}")
    return figure_path

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--d3_csv", default="results/metrics/summary_d3.csv")
    parser.add_argument("--d4_csv", default="results/metrics/summary_d4.csv")
    parser.add_argument("--output_dir", default="paper/figures")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating publication-ready figures...")
    
    # Load data
    try:
        d3_df = pd.read_csv(args.d3_csv)
        d4_df = pd.read_csv(args.d4_csv)
        print(f"📊 Loaded D3: {len(d3_df)} results, D4: {len(d4_df)} results")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Generate figures
    figures_generated = []
    
    try:
        fig1 = generate_d3_cross_domain_figure(d3_df, output_dir)
        figures_generated.append(fig1)
        
        fig2 = generate_d4_label_efficiency_figure(d4_df, output_dir) 
        figures_generated.append(fig2)
        
        fig3 = generate_model_comparison_figure(d3_df, output_dir)
        figures_generated.append(fig3)
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        return 1
    
    print(f"\n🎉 Successfully generated {len(figures_generated)} figures:")
    for fig_path in figures_generated:
        print(f"   📈 {fig_path}")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Review figures in {output_dir}")
    print(f"   2. Include in paper/main.tex using \\includegraphics")
    print(f"   3. Update Results section with D3/D4 analysis")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())