
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load REAL calibration results
d6_dir = Path("/workspace/results_gpu/d6")

calibration_data = {}
for f in d6_dir.glob("*.json"):
    with open(f) as file:
        data = json.load(file)
        model = f.stem.split('_')[1]
        if 'metrics' in data:
            calibration_data[model] = {
                'ece_raw': data['metrics'].get('ece_raw', 0),
                'ece_cal': data['metrics'].get('ece_cal', 0),
                'temperature': data['metrics'].get('temperature', 1.0),
                'brier': data['metrics'].get('brier', 0)
            }

# Create calibration comparison figure
models = list(calibration_data.keys())
ece_raw = [calibration_data[m]['ece_raw'] for m in models]
ece_cal = [calibration_data[m]['ece_cal'] for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ECE comparison
x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, ece_raw, width, label='Before Calibration', color='#E63946')
ax1.bar(x + width/2, ece_cal, width, label='After Calibration', color='#06FFA5')
ax1.set_xlabel('Model')
ax1.set_ylabel('Expected Calibration Error (ECE)')
ax1.set_title('Calibration Performance (Real Data)')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Temperature scaling values
temps = [calibration_data[m]['temperature'] for m in models]
ax2.bar(models, temps, color='#457B9D')
ax2.set_xlabel('Model')
ax2.set_ylabel('Temperature')
ax2.set_title('Learned Temperature Parameters')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_calibration_REAL.pdf', dpi=300, bbox_inches='tight')

for model in models:
    print(f"{model}: ECE {calibration_data[model]['ece_raw']:.3f} → {calibration_data[model]['ece_cal']:.3f}")
