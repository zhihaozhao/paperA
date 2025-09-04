#!/usr/bin/env python3
"""
Quick Experiment Runner for NVIDIA AGX Xavier
Run this directly on Xavier to get missing experimental data
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

def check_xavier_status():
    """Check Xavier hardware status"""
    print("="*60)
    print("XAVIER STATUS CHECK")
    print("="*60)
    
    commands = [
        ("GPU Info", "nvidia-smi"),
        ("Jetson Stats", "sudo jtop --json"),
        ("Memory", "free -h"),
        ("CPU Info", "lscpu | grep 'Model name\|CPU(s)'"),
        ("Disk Space", "df -h /"),
        ("Temperature", "cat /sys/devices/virtual/thermal/thermal_zone*/temp")
    ]
    
    for name, cmd in commands:
        print(f"\n{name}:")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(result.stdout[:500])  # Limit output
        except:
            print(f"  Could not run: {cmd}")

def install_dependencies():
    """Install required Python packages"""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    packages = [
        "torch",  # PyTorch for Jetson
        "torchvision",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pandas"
    ]
    
    print("Required packages:", ", ".join(packages))
    print("\nTo install PyTorch on Jetson:")
    print("wget https://nvidia.box.com/shared/static/pytorch_jp46.whl")
    print("pip3 install pytorch_jp46.whl")

def run_all_experiments():
    """Run all supplementary experiments"""
    
    experiments = [
        {
            'name': 'Model Efficiency',
            'script': 'measure_efficiency.py',
            'output': 'xavier_efficiency_results.json'
        },
        {
            'name': 'Calibration Test',
            'script': 'test_calibration.py',
            'output': 'xavier_calibration_results.json'
        },
        {
            'name': 'Cross-Domain Quick Test',
            'script': 'quick_cross_domain.py',
            'output': 'xavier_cross_domain_results.json'
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\nRunning: {exp['name']}...")
        try:
            # Run experiment
            subprocess.run([sys.executable, exp['script']], check=True)
            
            # Load results
            if Path(exp['output']).exists():
                with open(exp['output'], 'r') as f:
                    results[exp['name']] = json.load(f)
                print(f"  ✓ Completed: {exp['name']}")
        except Exception as e:
            print(f"  ✗ Failed: {exp['name']} - {e}")
            results[exp['name']] = {'error': str(e)}
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"xavier_all_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Results saved to: {output_file}")
    
    # Generate LaTeX summary
    if 'Model Efficiency' in results and not 'error' in results['Model Efficiency']:
        print("\n" + "="*60)
        print("LATEX TABLE FOR PAPER")
        print("="*60)
        generate_latex_table(results['Model Efficiency'])

def generate_latex_table(efficiency_results):
    """Generate LaTeX table from results"""
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Model Efficiency Measured on NVIDIA AGX Xavier}")
    print("\\label{tab:xavier_efficiency}")
    print("\\small")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Latency (ms)} & \\textbf{Memory (MB)} & \\textbf{Edge Ready} \\\\")
    print("\\midrule")
    
    for model, metrics in efficiency_results.items():
        edge = "\\checkmark" if metrics.get('edge_ready', False) else "$\\times$"
        print(f"{model} & {metrics.get('parameters_M', '-')} & "
              f"{metrics.get('inference_ms', '-')}$\\pm${metrics.get('inference_std', '-')} & "
              f"{metrics.get('memory_mb', '-')} & {edge} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\textit{Note: Measured on NVIDIA AGX Xavier (32GB) in MAXN mode.}")
    print("\\end{table}")

if __name__ == "__main__":
    print("NVIDIA AGX XAVIER EXPERIMENT RUNNER")
    print("="*60)
    
    # Check system
    check_xavier_status()
    
    # Run experiments
    run_all_experiments()
    
    print("\nDone! Please send the results file back for paper update.")
