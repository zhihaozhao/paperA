
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_fall_type_performance():
    """Load real fall type detection performance"""
    loso_dir = Path("/workspace/results_gpu/d3/loso")
    
    fall_types = ['epileptic_fall', 'elderly_fall', 'fall_cantgetup']
    models = ['enhanced', 'cnn', 'bilstm', 'conformer']
    
    results = {}
    for model in models:
        model_results = []
        for f in loso_dir.glob(f"loso_{model}_*.json"):
            with open(f) as file:
                data = json.load(file)
                if 'aggregate_stats' in data:
                    scores = []
                    for fall_type in fall_types:
                        key = f'{fall_type}_f1'
                        if key in data['aggregate_stats']:
                            scores.append(data['aggregate_stats'][key]['mean'])
                    if scores:
                        model_results.append(scores)
        
        if model_results:
            results[model] = np.mean(model_results, axis=0) * 100
    
    return results, fall_types

# Create figure
results, fall_types = load_fall_type_performance()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(fall_types))
width = 0.2
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, (model, color) in enumerate(zip(['enhanced', 'cnn', 'bilstm'], colors[:3])):
    if model in results:
        offset = (i - 1) * width
        ax.bar(x + offset, results[model], width, label=model.upper(), color=color)

ax.set_xlabel('Fall Type')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Fall Type Detection Performance (Real Data)')
ax.set_xticks(x)
ax.set_xticklabels(['Epileptic Fall', 'Elderly Fall', 'Fall (Can\'t Get Up)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, model in enumerate(['enhanced', 'cnn', 'bilstm']):
    if model in results:
        offset = (i - 1) * width
        for j, val in enumerate(results[model]):
            ax.text(j + offset, val + 1, f'{val:.1f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('fig6_fall_types_REAL.pdf')
print("Fall type analysis completed with real data!")
