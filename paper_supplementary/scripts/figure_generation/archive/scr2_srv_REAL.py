
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_real_srv_data():
    """Load real SRV experimental results"""
    d2_dir = Path("/workspace/results_gpu/d2")
    
    srv_results = defaultdict(lambda: defaultdict(list))
    
    for f in d2_dir.glob("*.json"):
        with open(f) as file:
            data = json.load(file)
            if 'args' in data and 'metrics' in data:
                model = data['args'].get('model', 'unknown')
                noise = data['args'].get('label_noise_prob', 0)
                f1 = data['metrics'].get('macro_f1', 0)
                srv_results[model][noise].append(f1)
    
    # Average results
    performance_matrix = []
    models = ['cnn', 'bilstm', 'conformer_lite', 'enhanced']
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    for model in models:
        row = []
        for noise in noise_levels:
            scores = srv_results[model].get(noise, [0])
            row.append(np.mean(scores))
        performance_matrix.append(row)
    
    return np.array(performance_matrix), models, noise_levels

# Create figure with REAL data
matrix, models, noises = load_real_srv_data()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.7, vmax=1.0)
ax.set_xlabel('Label Noise Level')
ax.set_ylabel('Model')
ax.set_title('SRV Robustness (Real Data)')
plt.colorbar(im, ax=ax)
plt.savefig('fig2c_srv_REAL.pdf')
