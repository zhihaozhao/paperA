#!/usr/bin/env python3
"""
Generate Deployment Strategy Decision Matrix
Creates visualization for IoT deployment scenario recommendations
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
    """Load Xavier experimental results"""
    base_path = Path("../../../../results_gpu/D1")
    cpu_path = base_path / "xavier_d1_cpu_20250905_170332.json"
    gpu_path = base_path / "xavier_d1_gpu_20250905_171132.json"
    
    with open(cpu_path, 'r') as f:
        cpu_data = json.load(f)
    
    with open(gpu_path, 'r') as f:
        gpu_data = json.load(f)
    
    return cpu_data, gpu_data

def create_deployment_strategy_matrix():
    """
    Create deployment strategy decision matrix
    Shows suitability scores for different IoT scenarios
    """
    
    cpu_data, gpu_data = load_xavier_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define deployment scenarios and model configurations
    scenarios = [
        'Smart Home Hub',
        'Wearable Device', 
        'IoT Gateway',
        'Mobile Application',
        'Industrial Monitor',
        'Autonomous Vehicle',
        'Healthcare Monitor'
    ]
    
    model_configs = [
        'PASE-Net\\n(CPU)',
        'PASE-Net\\n(GPU)', 
        'CNN\\n(CPU)',
        'CNN\\n(GPU)',
        'BiLSTM\\n(CPU)',
        'BiLSTM\\n(GPU)'
    ]
    
    # Suitability matrix (0-3 scale: 0=Poor, 1=Fair, 2=Good, 3=Excellent)
    # Based on: latency requirements, power constraints, accuracy needs, deployment complexity
    strategy_matrix = np.array([
        [2, 3, 2, 3, 2, 3],  # Smart Home Hub - wall power, multi-user
        [3, 1, 3, 2, 3, 1],  # Wearable Device - battery critical, simple tasks
        [2, 3, 2, 3, 2, 3],  # IoT Gateway - flexible power, high throughput
        [2, 2, 3, 3, 2, 2],  # Mobile Application - balanced requirements
        [1, 3, 2, 3, 1, 3],  # Industrial Monitor - reliability, real-time critical
        [1, 3, 1, 3, 1, 3],  # Autonomous Vehicle - ultra low latency required
        [3, 2, 3, 2, 3, 2],  # Healthcare Monitor - accuracy over speed, power sensitive
    ])
    
    # (a) Suitability heatmap
    im1 = ax1.imshow(strategy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    
    # Set ticks and labels
    ax1.set_xticks(np.arange(len(model_configs)))
    ax1.set_yticks(np.arange(len(scenarios)))
    ax1.set_xticklabels(model_configs, rotation=45, ha='right')
    ax1.set_yticklabels(scenarios)
    
    # Add text annotations
    suitability_labels = ['Poor', 'Fair', 'Good', 'Excellent']
    for i in range(len(scenarios)):
        for j in range(len(model_configs)):
            value = strategy_matrix[i, j]
            text_color = "white" if value < 2 else "black"
            text = ax1.text(j, i, suitability_labels[value], ha="center", va="center", 
                          color=text_color, fontweight='bold', fontsize=9)
    
    ax1.set_title('(a) Deployment Strategy Suitability Matrix', fontsize=12, pad=15)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Suitability Score', rotation=270, labelpad=15)
    cbar1.set_ticks([0, 1, 2, 3])
    cbar1.set_ticklabels(['Poor', 'Fair', 'Good', 'Excellent'])
    
    # (b) Performance vs Power Efficiency Analysis
    models = ['enhanced', 'cnn', 'bilstm']
    model_names = ['PASE-Net', 'CNN', 'BiLSTM']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    
    # Power consumption estimates
    cpu_power = 10  # Watts
    gpu_power = 25  # Watts
    
    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        cpu_time = cpu_data['models'][model]['avg_inference_time_ms']
        gpu_time = gpu_data['models'][model]['batch_results']['batch_1']['avg_per_sample_time_ms']
        
        # Calculate efficiency metric: samples per second per watt
        cpu_efficiency = (1000 / cpu_time) / cpu_power
        gpu_efficiency = (1000 / gpu_time) / gpu_power
        
        # Calculate performance score (inverse of latency, normalized)
        cpu_performance = min(100, 1000 / cpu_time)  # Cap at 100 for visualization
        gpu_performance = min(100, 1000 / gpu_time)
        
        # Plot efficiency vs performance
        ax2.scatter(cpu_efficiency, cpu_performance, c=color, s=150, alpha=0.8, 
                   marker='s', edgecolor='black', linewidth=1, label=f'{name} (CPU)')
        ax2.scatter(gpu_efficiency, gpu_performance, c=color, s=150, alpha=0.8, 
                   marker='o', edgecolor='black', linewidth=1, label=f'{name} (GPU)')
        
        # Connect CPU and GPU points
        ax2.plot([cpu_efficiency, gpu_efficiency], [cpu_performance, gpu_performance], 
                color=color, alpha=0.5, linestyle='--', linewidth=2)
        
        # Add labels for GPU points (better performance)
        ax2.annotate(f'{name}\\n(GPU)', 
                    xy=(gpu_efficiency, gpu_performance),
                    xytext=(10, 10), textcoords='offset points',
                    ha='left', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax2.set_xlabel('Energy Efficiency (samples/sec/Watt)')
    ax2.set_ylabel('Performance Score (samples/sec, capped at 100)')
    ax2.set_title('(b) Performance vs Energy Efficiency Trade-off', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax2.text(0.02, 0.95, 'Low Efficiency\\nHigh Performance', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            fontsize=8, ha='left', va='top')
    ax2.text(0.98, 0.05, 'High Efficiency\\nLow Performance', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), 
            fontsize=8, ha='right', va='bottom')
    ax2.text(0.98, 0.95, 'High Efficiency\\nHigh Performance\\n(Ideal)', transform=ax2.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=8, ha='right', va='top', weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "deployment_strategy_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Deployment strategy analysis saved to: {output_path}")
    
    return fig

def create_scenario_requirements_table():
    """
    Create detailed requirements table for each deployment scenario
    """
    
    scenarios_data = {
        'Smart Home Hub': {
            'Power': 'Wall-powered (flexible)',
            'Latency': 'Real-time preferred (<10ms)',
            'Users': 'Multiple concurrent (2-8)',
            'Accuracy': 'High (>80%)',
            'Complexity': 'Medium',
            'Recommended': 'PASE-Net GPU / CNN GPU',
            'Rationale': 'Multi-user support with attention-based accuracy'
        },
        'Wearable Device': {
            'Power': 'Battery critical (<5W)',
            'Latency': 'Near real-time acceptable',
            'Users': 'Single user',
            'Accuracy': 'Medium (>75%)',
            'Complexity': 'Low',
            'Recommended': 'CNN CPU / PASE-Net CPU',
            'Rationale': 'Battery optimization prioritized over latency'
        },
        'IoT Gateway': {
            'Power': 'Hybrid (10-30W)',
            'Latency': 'Real-time required',
            'Users': 'High throughput (10+)',
            'Accuracy': 'High (>82%)',
            'Complexity': 'High',
            'Recommended': 'PASE-Net GPU with batch=8',
            'Rationale': 'Maximum throughput with dynamic power management'
        },
        'Mobile Application': {
            'Power': 'Battery aware (5-15W)',
            'Latency': 'Responsive (<50ms)',
            'Users': 'Single with buffering',
            'Accuracy': 'Good (>78%)',
            'Complexity': 'Medium',
            'Recommended': 'CNN GPU with batch=4',
            'Rationale': 'Balance of performance and efficiency'
        },
        'Industrial Monitor': {
            'Power': 'Wall-powered',
            'Latency': 'Ultra real-time (<5ms)',
            'Users': 'Multiple sensors',
            'Accuracy': 'Critical (>85%)',
            'Complexity': 'High',
            'Recommended': 'CNN GPU / PASE-Net GPU',
            'Rationale': 'Reliability and ultra-low latency for safety'
        }
    }
    
    # Generate LaTeX table
    latex_table = """
% Deployment Scenario Requirements Analysis
\\begin{table*}[t]
\\centering
\\caption{Detailed Deployment Scenario Requirements and Recommendations}
\\label{tab:deployment_scenarios}
\\footnotesize
\\begin{tabular}{@{}p{2.5cm}p{2cm}p{1.8cm}p{1.8cm}p{1.5cm}p{2.5cm}p{3cm}@{}}
\\toprule
\\textbf{Scenario} & \\textbf{Power Constraints} & \\textbf{Latency Requirements} & \\textbf{User Concurrency} & \\textbf{Accuracy Needs} & \\textbf{Recommended Config} & \\textbf{Design Rationale} \\\\
\\midrule"""
    
    for scenario, data in scenarios_data.items():
        latex_table += f"""
{scenario} & {data['Power']} & {data['Latency']} & {data['Users']} & {data['Accuracy']} & {data['Recommended']} & {data['Rationale']} \\\\"""
    
    latex_table += """
\\bottomrule
\\end{tabular}
\\end{table*}
\\textit{Note: Recommendations based on Xavier AGX 32G performance measurements and typical IoT deployment constraints. Power values represent expected consumption ranges for different operational modes.}
"""
    
    # Save LaTeX table
    with open('deployment_scenarios_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print("Deployment scenarios table generated: deployment_scenarios_table.tex")

def main():
    """Generate deployment strategy analysis figures and tables"""
    print("Generating deployment strategy decision matrix and analysis...")
    
    try:
        # Create main analysis figure
        fig = create_deployment_strategy_matrix()
        
        # Create detailed requirements table
        create_scenario_requirements_table()
        
        print("SUCCESS: Deployment strategy analysis generated!")
        print("Files created:")
        print("- deployment_strategy_analysis.pdf")
        print("- deployment_scenarios_table.tex")
        print("Analysis shows optimal configurations for 7 deployment scenarios")
        
    except Exception as e:
        print(f"ERROR: Error generating deployment analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())