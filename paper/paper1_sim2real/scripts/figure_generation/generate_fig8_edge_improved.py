#!/usr/bin/env python3
"""
Improved Figure 8: Comprehensive Edge Deployment Analysis with Complete Legends
Enhanced version with better organization, external legends, and subfigure labels
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Enhanced publication style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.titleweight': 'bold',
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5
})

def get_comprehensive_edge_data():
    """Get comprehensive experimental data for all models with complete metrics"""
    
    # Complete experimental data from Xavier AGX 32G
    edge_data = {
        'models': {
            'Enhanced': {
                'total_params': 640713,
                'model_size_mb': 2.44,
                'cpu_latency_ms': 338.91,
                'gpu_latency_ms': 5.29,
                'throughput_sps': 607.1,
                'speedup_factor': 64.1,
                'memory_efficiency': 0.95,
                'accuracy_f1': 83.0,
                'batch_scaling': {
                    'batch_1': {'throughput': 189.2, 'latency': 5.29},
                    'batch_4': {'throughput': 501.0, 'latency': 2.0},
                    'batch_8': {'throughput': 607.1, 'latency': 1.65}
                }
            },
            'CNN': {
                'total_params': 644216,
                'model_size_mb': 2.46,
                'cpu_latency_ms': 7.13,
                'gpu_latency_ms': 0.90,
                'throughput_sps': 7076.2,
                'speedup_factor': 7.9,
                'memory_efficiency': 0.92,
                'accuracy_f1': 84.2,
                'batch_scaling': {
                    'batch_1': {'throughput': 1113.0, 'latency': 0.90},
                    'batch_4': {'throughput': 4717.0, 'latency': 0.21},
                    'batch_8': {'throughput': 7076.2, 'latency': 0.14}
                }
            },
            'BiLSTM': {
                'total_params': 583688,
                'model_size_mb': 2.23,
                'cpu_latency_ms': 75.46,
                'gpu_latency_ms': 8.97,
                'throughput_sps': 851.3,
                'speedup_factor': 8.4,
                'memory_efficiency': 0.89,
                'accuracy_f1': 80.3,
                'batch_scaling': {
                    'batch_1': {'throughput': 112.0, 'latency': 8.97},
                    'batch_4': {'throughput': 424.0, 'latency': 2.36},
                    'batch_8': {'throughput': 851.3, 'latency': 1.17}
                }
            }
        },
        'deployment_scenarios': {
            'Smart Home Hub': {'min_throughput': 50, 'max_latency': 10, 'max_memory': 5},
            'Industrial Monitor': {'min_throughput': 100, 'max_latency': 5, 'max_memory': 8},
            'IoT Gateway': {'min_throughput': 200, 'max_latency': 15, 'max_memory': 4},
            'Edge Server': {'min_throughput': 500, 'max_latency': 20, 'max_memory': 16}
        }
    }
    
    return edge_data

def create_improved_edge_deployment_figure():
    """Create comprehensive edge deployment figure with complete legends"""
    
    data = get_comprehensive_edge_data()
    
    # Create figure with optimized layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35, 
                  height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Define consistent color scheme and markers
    model_colors = {
        'Enhanced': '#E31A1C',    # Red - our proposed model
        'CNN': '#1F78B4',        # Blue - baseline
        'BiLSTM': '#33A02C'      # Green - baseline
    }
    
    model_markers = {
        'Enhanced': 'o',  # Circle
        'CNN': 's',       # Square  
        'BiLSTM': '^'     # Triangle
    }
    
    models = ['Enhanced', 'CNN', 'BiLSTM']
    
    # (a) CPU vs GPU Latency Comparison with Error Bars
    ax1 = fig.add_subplot(gs[0, 0])
    
    cpu_times = [data['models'][m]['cpu_latency_ms'] for m in models]
    gpu_times = [data['models'][m]['gpu_latency_ms'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU Inference', 
                   alpha=0.7, color=[model_colors[m] for m in models],
                   edgecolor='black', linewidth=1.5, hatch='///')
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU Inference',
                   alpha=0.9, color=[model_colors[m] for m in models],
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.1f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    # Add real-time threshold line
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label='Real-time Threshold (10ms)')
    
    ax1.set_xlabel('Model Architecture', fontweight='bold')
    ax1.set_ylabel('Inference Latency (ms)', fontweight='bold')
    ax1.set_title('(a) CPU vs GPU Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # (b) Batch Throughput Scaling Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    
    batch_sizes = [1, 4, 8]
    for model in models:
        throughputs = [data['models'][model]['batch_scaling'][f'batch_{b}']['throughput'] 
                      for b in batch_sizes]
        ax2.plot(batch_sizes, throughputs, 
                marker=model_markers[model], color=model_colors[model],
                label=f'{model} Model', linewidth=3, markersize=10,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor=model_colors[model])
    
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax2.set_title('(b) GPU Throughput Scalability', fontweight='bold')
    ax2.set_xticks(batch_sizes)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # (c) Memory Efficiency vs Model Size
    ax3 = fig.add_subplot(gs[0, 2])
    
    params = [data['models'][m]['total_params']/1000 for m in models]  # In K
    memory = [data['models'][m]['model_size_mb'] for m in models]
    accuracy = [data['models'][m]['accuracy_f1'] for m in models]
    
    # Bubble plot: size represents accuracy
    bubble_sizes = [(acc/100) * 500 for acc in accuracy]  # Scale bubble size
    
    for i, model in enumerate(models):
        ax3.scatter(params[i], memory[i], s=bubble_sizes[i], 
                   c=model_colors[model], alpha=0.7, 
                   edgecolors='black', linewidth=2,
                   marker=model_markers[model], label=f'{model} (F1: {accuracy[i]:.1f}%)')
    
    ax3.set_xlabel('Model Parameters (K)', fontweight='bold')
    ax3.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax3.set_title('(c) Memory Efficiency Analysis', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # (d) Performance-Efficiency Trade-off Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Normalize metrics for radar-like comparison
    metrics = ['Accuracy', 'Speed', 'Memory', 'Throughput']
    
    # Get normalized values (higher is better)
    normalized_data = {}
    max_acc = max(data['models'][m]['accuracy_f1'] for m in models)
    min_latency = min(data['models'][m]['gpu_latency_ms'] for m in models)
    min_memory = min(data['models'][m]['model_size_mb'] for m in models)
    max_throughput = max(data['models'][m]['throughput_sps'] for m in models)
    
    for model in models:
        normalized_data[model] = [
            data['models'][model]['accuracy_f1'] / max_acc,  # Accuracy (higher better)
            min_latency / data['models'][model]['gpu_latency_ms'],  # Speed (lower latency better)
            min_memory / data['models'][model]['model_size_mb'],  # Memory (lower better)
            data['models'][model]['throughput_sps'] / max_throughput  # Throughput (higher better)
        ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        ax4.bar(x + i*width, normalized_data[model], width, 
               label=model, color=model_colors[model], alpha=0.8,
               edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Performance Metrics', fontweight='bold')
    ax4.set_ylabel('Normalized Score', fontweight='bold')
    ax4.set_title('(d) Multi-Metric Performance Comparison', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metrics)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # (e) Deployment Scenario Feasibility Matrix
    ax5 = fig.add_subplot(gs[1, 1:])  # Span two columns
    
    scenarios = list(data['deployment_scenarios'].keys())
    feasibility_matrix = []
    
    for model in models:
        model_data = data['models'][model]
        row = []
        for scenario_name in scenarios:
            scenario = data['deployment_scenarios'][scenario_name]
            
            # Check feasibility criteria
            throughput_ok = model_data['throughput_sps'] >= scenario['min_throughput']
            latency_ok = model_data['gpu_latency_ms'] <= scenario['max_latency']
            memory_ok = model_data['model_size_mb'] <= scenario['max_memory']
            
            # Overall feasibility score
            score = sum([throughput_ok, latency_ok, memory_ok]) / 3
            row.append(score)
        
        feasibility_matrix.append(row)
    
    feasibility_matrix = np.array(feasibility_matrix)
    
    # Create heatmap
    im = ax5.imshow(feasibility_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(scenarios)):
            score = feasibility_matrix[i, j]
            if score >= 0.8:
                text = 'Excellent'
                color = 'black'
            elif score >= 0.6:
                text = 'Good'
                color = 'black'
            elif score >= 0.4:
                text = 'Fair'
                color = 'white'
            else:
                text = 'Poor'
                color = 'white'
            
            ax5.text(j, i, f'{text}\n({score:.1f})', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=9)
    
    ax5.set_xticks(range(len(scenarios)))
    ax5.set_xticklabels(scenarios, rotation=45, ha='right')
    ax5.set_yticks(range(len(models)))
    ax5.set_yticklabels([f'{m} Model' for m in models])
    ax5.set_title('(e) IoT Deployment Scenario Compatibility Matrix', 
                  fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.6, aspect=20)
    cbar.set_label('Feasibility Score', rotation=270, labelpad=20, fontweight='bold')
    cbar.set_ticks([0, 0.33, 0.67, 1.0])
    cbar.set_ticklabels(['Poor', 'Fair', 'Good', 'Excellent'])
    
    # (f) Summary Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create summary data table
    summary_data = []
    headers = ['Model', 'Parameters\n(K)', 'Memory\n(MB)', 'GPU Latency\n(ms)', 
              'Throughput\n(sps)', 'Speedup\nFactor', 'Accuracy\n(% F1)', 'Best Use Case']
    
    use_cases = {
        'Enhanced': 'Balanced Performance',
        'CNN': 'High Throughput',
        'BiLSTM': 'Memory Efficient'
    }
    
    for model in models:
        model_data = data['models'][model]
        row = [
            model,
            f"{model_data['total_params']/1000:.0f}",
            f"{model_data['model_size_mb']:.2f}",
            f"{model_data['gpu_latency_ms']:.2f}",
            f"{model_data['throughput_sps']:.0f}",
            f"{model_data['speedup_factor']:.1f}x",
            f"{model_data['accuracy_f1']:.1f}%",
            use_cases[model]
        ]
        summary_data.append(row)
    
    # Create table
    table = ax6.table(cellText=summary_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Header styling
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.15)
    
    # Model-specific row coloring
    for i in range(1, len(models) + 1):
        model_name = models[i-1]
        row_color = model_colors[model_name] + '20'  # Add transparency
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(row_color)
            if j == 0:  # Model name column
                table[(i, j)].set_text_props(weight='bold')
    
    ax6.set_title('(f) Comprehensive Edge Deployment Performance Summary', 
                  fontweight='bold', fontsize=13, pad=20)
    
    # Overall figure title
    fig.suptitle('Comprehensive Edge Deployment Analysis on Xavier AGX 32G Platform\n'
                'Complete Performance Characterization for IoT WiFi CSI HAR Systems', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add technical notes
    notes_text = (
        "Technical Notes: (1) Real-time threshold: <10ms latency, (2) Xavier AGX 32G: ARM Cortex-A78AE + NVIDIA Volta GPU\n"
        "(3) Throughput measured at batch size 8, (4) Memory includes model weights only, (5) Speedup: CPU/GPU ratio"
    )
    
    fig.text(0.02, 0.01, notes_text, fontsize=9, style='italic',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_path = Path('paper/paper1_sim2real/plots/fig8_edge_deployment_analysis.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SUCCESS] Improved Figure 8 saved to {output_path}")
    
    return fig

def generate_edge_performance_latex_table():
    """Generate LaTeX table for edge performance comparison"""
    
    data = get_comprehensive_edge_data()
    
    latex_table = """
\\begin{table*}[t]
\\centering
\\caption{Comprehensive Edge Deployment Performance Analysis on Xavier AGX 32G Platform}
\\label{tab:edge_deployment_analysis}
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
\\textbf{Model} & \\textbf{Parameters} & \\textbf{Memory} & \\textbf{GPU Latency} & \\textbf{Throughput} & \\textbf{Speedup} & \\textbf{Accuracy} \\\\
 & \\textbf{(K)} & \\textbf{(MB)} & \\textbf{(ms)} & \\textbf{(sps)} & \\textbf{Factor} & \\textbf{(\\% F1)} \\\\
\\midrule"""
    
    for model in ['Enhanced', 'CNN', 'BiLSTM']:
        model_data = data['models'][model]
        latex_table += f"""
{model} & {model_data['total_params']/1000:.0f} & {model_data['model_size_mb']:.2f} & {model_data['gpu_latency_ms']:.2f} & {model_data['throughput_sps']:.0f} & {model_data['speedup_factor']:.1f}× & {model_data['accuracy_f1']:.1f} \\\\"""
    
    latex_table += """
\\bottomrule
\\end{tabular}
\\end{table*}"""
    
    print("\n" + "="*80)
    print("LATEX TABLE FOR EDGE DEPLOYMENT ANALYSIS:")
    print("="*80)
    print(latex_table)
    
    return latex_table

if __name__ == "__main__":
    print("Generating Improved Figure 8: Comprehensive Edge Deployment Analysis...")
    fig = create_improved_edge_deployment_figure()
    generate_edge_performance_latex_table()
    print("\n[SUCCESS] Complete edge deployment analysis figure generated!")
    plt.show()