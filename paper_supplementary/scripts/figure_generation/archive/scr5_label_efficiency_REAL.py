
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_label_efficiency():
    """Load real label efficiency results"""
    d4_dir = Path("/workspace/results_gpu/d4/sim2real")
    
    results = {}
    for f in d4_dir.glob("enhanced_*.json"):
        with open(f) as file:
            data = json.load(file)
            if 'label_ratio' in data:
                ratio = data['label_ratio'] * 100
                results[ratio] = {
                    'zero_shot': data.get('zero_shot_metrics', {}).get('macro_f1', 0) * 100,
                    'fine_tuned': data.get('target_metrics', {}).get('macro_f1', 0) * 100
                }
    
    return results

# Create figure
results = load_label_efficiency()
ratios = sorted(results.keys())
zero_shot = [results[r]['zero_shot'] for r in ratios]
fine_tuned = [results[r]['fine_tuned'] for r in ratios]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(ratios))
width = 0.35

ax.bar(x - width/2, zero_shot, width, label='Zero-Shot', color='#E63946')
ax.bar(x + width/2, fine_tuned, width, label='Fine-Tuned', color='#06FFA5')

ax.set_xlabel('Label Ratio (%)')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Label Efficiency: Sim2Real Transfer (Real Data)')
ax.set_xticks(x)
ax.set_xticklabels([f'{r:.0f}%' for r in ratios])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig5_label_efficiency_REAL.pdf')
print(f"Label ratios tested: {ratios}")
