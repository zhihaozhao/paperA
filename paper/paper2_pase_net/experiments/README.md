# NVIDIA AGX Xavier Quick Experiments

## 🚀 Quick Start

1. **Copy to Xavier:**
```bash
scp -r experiments/ xavier@<xavier-ip>:~/
```

2. **On Xavier, run:**
```bash
cd experiments/
python3 run_xavier_experiments.py
```

3. **Get results back:**
```bash
scp xavier@<xavier-ip>:~/experiments/xavier_all_results_*.json ./
```

## 📊 What These Experiments Provide

1. **Model Efficiency** (Fixes Table 1)
   - Actual parameter counts
   - Real inference time on Xavier
   - Memory usage
   - Edge deployment feasibility

2. **Calibration Metrics** (Validates ECE claims)
   - Raw ECE
   - Calibrated ECE  
   - Temperature scaling effectiveness
   - Actual improvement percentage

3. **Quick Cross-Domain Test**
   - Verify LOSO/LORO performance
   - Small-scale validation

## ⚡ Expected Runtime

- Setup: ~5 minutes
- Experiments: ~10-15 minutes
- Total: ~20 minutes

## 📝 Results Format

Results will be in JSON format, ready to update the paper:
```json
{
  "Model Efficiency": {
    "PASE-Net": {
      "parameters_M": 0.53,
      "inference_ms": 8.2,
      "memory_mb": 24,
      "edge_ready": true
    }
  }
}
```

## ✅ This Solves

- ❌ Hardcoded parameters → ✅ Real measurements
- ❌ No inference time → ✅ Xavier benchmarks
- ❌ Missing calibration → ✅ Actual ECE values
