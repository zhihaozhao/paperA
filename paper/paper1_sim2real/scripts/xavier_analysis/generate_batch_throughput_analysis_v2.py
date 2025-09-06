#!/usr/bin/env python3
"""
Generate Optimized Figure: Batch Throughput Analysis for Edge Deployment
Creates: plots/batch_throughput_analysis.pdf
Version: v2 - Improved text positioning and title formatting
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set publication style with improved spacing
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlepad': 15,  # Increase title padding
    'figure.subplot.top': 0.92,  # Increase top margin
    'figure.subplot.bottom': 0.15,  # Increase bottom margin
    'figure.subplot.left': 0.1,   # Increase left margin
    'figure.subplot.right': 0.95,  # Adjust right margin
})

def load_xavier_data():
    """Load Xavier experimental results from results_gpu directory"""
    # Paths relative to root directory
    base_path = Path("../../../../results_gpu/D1")
    cpu_path = base_path / "xavier_d1_cpu_20250905_170332.json"
    gpu_path = base_path / "xavier_d1_gpu_20250905_171132.json"
    
    with open(cpu_path, 'r') as f:
        cpu_data = json.load(f)
    
    with open(gpu_path, 'r') as f:
        gpu_data = json.load(f)
    
    return cpu_data, gpu_data

def create_optimized_batch_throughput_figure():
    """
    Create optimized batch throughput analysis figure with improved text positioning
    """
    
    cpu_data, gpu_data = load_xavier_data()
    
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    models = ['enhanced', 'cnn', 'bilstm']
    model_names = ['PASE-Net', 'CNN', 'BiLSTM']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    batch_sizes = [1, 4, 8]
    
    # (a) Per-sample latency scaling - Improved layout
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        gpu_model = gpu_data['models'][model]
        per_sample_times = []
        for batch in batch_sizes:
            batch_key = f"batch_{batch}"
            per_sample_times.append(gpu_model['batch_results'][batch_key]['avg_per_sample_time_ms'])
        
        ax1.plot(batch_sizes, per_sample_times, 'o-', color=color, label=name, 
                linewidth=2.5, markersize=7)
    
    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Per-Sample Latency (ms)', fontweight='bold')
    ax1.set_title('Batch Processing Efficiency', fontweight='bold', pad=12)
    ax1.set_xticks(batch_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax1.set_yscale('log')
    
    # Real-time threshold with better positioning
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    ax1.text(2.5, 12, 'Real-time\\nthreshold', fontsize=7, color='red', 
            ha='center', va='bottom', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # (b) Absolute throughput comparison - Improved bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.25
    x = np.arange(len(batch_sizes))
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        gpu_model = gpu_data['models'][model]
        throughputs = []
        for batch in batch_sizes:
            batch_key = f"batch_{batch}"
            throughputs.append(gpu_model['batch_results'][batch_key]['throughput_samples_per_sec'])
        
        bars = ax2.bar(x + i*width, throughputs, width, label=name, color=color, alpha=0.8)
        
        # Add value labels on top of bars (only for batch=8 to avoid clutter)
        if True:  # Show all values but smaller font
            for j, (bar, val) in enumerate(zip(bars, throughputs)):
                if val > 100:  # Only show for high values
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.05,
                            f'{int(val)}', ha='center', va='bottom', fontsize=6, rotation=0)
    
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Throughput (samples/sec)', fontweight='bold')
    ax2.set_title('Throughput Scalability', fontweight='bold', pad=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # (c) Power-performance analysis - Improved scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    
    cpu_power = 10  # Watts
    gpu_power = 25  # Watts
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        cpu_time = cpu_data['models'][model]['avg_inference_time_ms']
        gpu_time = gpu_data['models'][model]['batch_results']['batch_1']['avg_per_sample_time_ms']
        
        # Plot CPU point
        cpu_point = ax3.scatter(cpu_power, cpu_time, c=color, s=120, alpha=0.8, 
                               marker='s', edgecolor='black', linewidth=1)
        
        # Plot GPU point
        gpu_point = ax3.scatter(gpu_power, gpu_time, c=color, s=120, alpha=0.8, 
                               marker='o', edgecolor='black', linewidth=1)
        
        # Connection line
        ax3.plot([cpu_power, gpu_power], [cpu_time, gpu_time], 
                color=color, alpha=0.6, linestyle='--', linewidth=1.5)
        
        # Model labels with better positioning
        if i == 0:  # PASE-Net
            ax3.annotate(name, xy=(cpu_power, cpu_time), 
                        xytext=(-25, 15), textcoords='offset points',
                        fontsize=7, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        elif i == 1:  # CNN
            ax3.annotate(name, xy=(gpu_power, gpu_time), 
                        xytext=(10, -15), textcoords='offset points',
                        fontsize=7, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        else:  # BiLSTM
            ax3.annotate(name, xy=(gpu_power, gpu_time), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=7, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Legend with custom markers
    cpu_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=8, label='CPU Mode', markeredgecolor='black')
    gpu_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=8, label='GPU Mode', markeredgecolor='black')
    ax3.legend(handles=[cpu_marker, gpu_marker], loc='upper right', fontsize=7, framealpha=0.9)
    
    ax3.set_xlabel('Power Consumption (W)', fontweight='bold')
    ax3.set_ylabel('Inference Time (ms)', fontweight='bold')
    ax3.set_title('Power vs Performance', fontweight='bold', pad=12)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Real-time threshold with better positioning
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    ax3.text(22, 8, 'Real-time', fontsize=7, color='red', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Adjust layout to prevent overlaps
    plt.tight_layout(pad=2.0)
    
    # Save figure with high quality
    output_path = "batch_throughput_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Optimized figure saved to: {output_path}")
    
    return fig

def main():
    """Generate optimized batch throughput analysis figure"""
    print("Generating optimized batch throughput analysis figure v2...")
    print("Loading Xavier AGX 32G experimental data from results_gpu/D1/")
    
    try:
        fig = create_optimized_batch_throughput_figure()
        print("SUCCESS: Optimized batch_throughput_analysis.pdf generated!")
        print("Improvements:")
        print("  - Better text positioning and spacing")
        print("  - Improved title formatting")
        print("  - Enhanced legend placement")
        print("  - Cleaner annotation layout")
    except Exception as e:
        print(f"ERROR: Error generating figure: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())