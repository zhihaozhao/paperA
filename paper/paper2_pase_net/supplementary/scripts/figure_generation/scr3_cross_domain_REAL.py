
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load REAL LOSO/LORO results
loso_dir = Path("/workspace/results_gpu/d3/loso")
loro_dir = Path("/workspace/results_gpu/d3/loro")

results = {}
for protocol, dir_path in [('LOSO', loso_dir), ('LORO', loro_dir)]:
    for f in dir_path.glob(f"{protocol.lower()}_*.json"):
        if 'summary' not in str(f):
            with open(f) as file:
                data = json.load(file)
                model = f.stem.split('_')[1]
                if model not in results:
                    results[model] = {}
                if 'aggregate_stats' in data:
                    f1 = data['aggregate_stats']['macro_f1']['mean']
                    results[model][protocol] = f1 * 100

# Create figure with REAL data
models = ['enhanced', 'cnn', 'bilstm']
loso_scores = [results[m].get('LOSO', 0) for m in models]
loro_scores = [results[m].get('LORO', 0) for m in models]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, loso_scores, width, label='LOSO (Real)', color='#2E86AB')
bars2 = ax.bar(x + width/2, loro_scores, width, label='LORO (Real)', color='#A23B72')

ax.set_xlabel('Model')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Cross-Domain Performance on Real WiFi CSI Data')
ax.set_xticks(x)
ax.set_xticklabels(['PASE-Net', 'CNN', 'BiLSTM'])
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('fig3_cross_domain_REAL.pdf', dpi=300, bbox_inches='tight')
print(f"PASE-Net LOSO: {results['enhanced']['LOSO']:.1f}%")
print(f"PASE-Net LORO: {results['enhanced']['LORO']:.1f}%")
