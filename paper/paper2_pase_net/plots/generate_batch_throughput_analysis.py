#!/usr/bin/env python3
"""
Generate Figure: Batch Throughput Analysis for Edge Deployment Section
Creates: plots/batch_throughput_analysis.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm'
})

def load_xavier_data():
    """Load Xavier experimental results from results_gpu directory"""
    # Paths relative to paper/paper2_pase_net/manuscript/plots
    base_path = Path("../../../../results_gpu/D1")
    cpu_path = base_path / "xavier_d1_cpu_20250905_170332.json"
    gpu_path = base_path / "xavier_d1_gpu_20250905_171132.json"
    
    with open(cpu_path, 'r') as f:
        cpu_data = json.load(f)
    
    with open(gpu_path, 'r') as f:
        gpu_data = json.load(f)
    
    return cpu_data, gpu_data

def create_batch_throughput_figure():
    """
    Create comprehensive batch throughput analysis figure
    Figure shows: (a) Per-sample latency scaling, (b) Throughput comparison, (c) Power-performance trade-off
    """
    
    cpu_data, gpu_data = load_xavier_data()
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    models = ['enhanced', 'cnn', 'bilstm']
    model_names = ['PASE-Net', 'CNN', 'BiLSTM']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    # Extract data
    batch_sizes = [1, 4, 8]
    
    # (a) Per-sample latency scaling
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        gpu_model = gpu_data['models'][model]
        per_sample_times = []
        for batch in batch_sizes:
            batch_key = f"batch_{batch}"
            per_sample_times.append(gpu_model['batch_results'][batch_key]['avg_per_sample_time_ms'])
        
        ax1.plot(batch_sizes, per_sample_times, 'o-', color=color, label=name, 
                linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Per-Sample Inference Time (ms)')
    ax1.set_title('(a) Per-sample inference time scaling across batch sizes')
    ax1.set_xticks(batch_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Add real-time threshold line
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(6, 12, 'Real-time threshold (10ms)', fontsize=8, color='red')
    
    # (b) Absolute throughput comparison
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.25
    x = np.arange(len(batch_sizes))
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        gpu_model = gpu_data['models'][model]
        throughputs = []
        for batch in batch_sizes:
            batch_key = f"batch_{batch}"
            throughputs.append(gpu_model['batch_results'][batch_key]['throughput_samples_per_sec'])
        
        ax2.bar(x + i*width, throughputs, width, label=name, color=color, alpha=0.8)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('(b) Absolute throughput comparison')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # (c) Power-performance trade-off analysis
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Power consumption estimates (based on Xavier AGX 32G documentation)
    cpu_power = 10  # Watts
    gpu_power = 25  # Watts
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        cpu_time = cpu_data['models'][model]['avg_inference_time_ms']
        gpu_time = gpu_data['models'][model]['batch_results']['batch_1']['avg_per_sample_time_ms']
        
        # Plot CPU point
        ax3.scatter(cpu_power, cpu_time, c=color, s=150, alpha=0.7, 
                   marker='s', edgecolor='black', linewidth=1)
        
        # Plot GPU point
        ax3.scatter(gpu_power, gpu_time, c=color, s=150, alpha=0.7, 
                   marker='o', edgecolor='black', linewidth=1)
        
        # Draw connection line showing improvement
        ax3.plot([cpu_power, gpu_power], [cpu_time, gpu_time], 
                color=color, alpha=0.5, linestyle='--', linewidth=2)
        
        # Add model label
        if i == 0:  # Only label for first model to avoid clutter
            ax3.text(cpu_power-1, cpu_time, f'{name}', ha='right', va='center', 
                    fontsize=8, color=color, fontweight='bold')
    
    # Add legend for markers
    cpu_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=10, label='CPU Mode', markeredgecolor='black')
    gpu_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markersize=10, label='GPU Mode', markeredgecolor='black')
    ax3.legend(handles=[cpu_marker, gpu_marker], loc='upper right')
    
    ax3.set_xlabel('Power Consumption (Watts)')
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('(c) Power-performance trade-off analysis')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add real-time threshold
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.text(20, 12, 'Real-time threshold', fontsize=8, color='red')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "batch_throughput_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    return fig

def main():
    """Generate batch throughput analysis figure"""
    print("Generating batch throughput analysis figure...")
    print("Loading Xavier AGX 32G experimental data from results_gpu/D1/")
    
    try:
        fig = create_batch_throughput_figure()
        print("SUCCESS: batch_throughput_analysis.pdf generated successfully!")
        print("Figure shows:")
        print("  (a) Per-sample inference time scaling across batch sizes")
        print("  (b) Absolute throughput comparison revealing CNN's superior scaling")  
        print("  (c) Power-performance trade-off analysis between CPU and GPU modes")
    except Exception as e:
        print(f"ERROR: Error generating figure: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())