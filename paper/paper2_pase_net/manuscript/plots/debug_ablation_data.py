#!/usr/bin/env python3
"""
Debug script to check ablation data collection
"""

import json
import pathlib
import statistics as st

ROOT = pathlib.Path("/workspace")
D2 = ROOT / "results_gpu" / "d2"

def parse_val(s: str) -> float:
    return float(s.replace("p", "."))

def collect_debug_data():
    """Debug data collection"""
    
    print("="*60)
    print("DEBUG: ABLATION DATA COLLECTION")
    print("="*60)
    
    # Check if directory exists
    if not D2.exists():
        print(f"ERROR: Directory does not exist: {D2}")
        return
    
    print(f"Data directory: {D2}")
    
    # List all relevant files
    pattern = "paperA_*_hard_s*_cla*_env*_lab*.json"
    files = list(D2.glob(pattern))
    print(f"\nTotal files found: {len(files)}")
    
    if not files:
        print("ERROR: No files found!")
        return
    
    # Sample first few files
    print("\nSample files:")
    for f in files[:5]:
        print(f"  {f.name}")
    
    # Collect data for each model
    models = ["enhanced", "cnn", "bilstm"]
    
    for model in models:
        print(f"\n{'-'*40}")
        print(f"Model: {model}")
        print(f"{'-'*40}")
        
        model_files = [f for f in files if f"paperA_{model}_" in f.name]
        print(f"Files for {model}: {len(model_files)}")
        
        if not model_files:
            print(f"  WARNING: No files for {model}")
            continue
        
        # Parse and collect data
        data_by_condition = {}
        
        for f in model_files:
            name = f.stem
            parts = name.split("_")
            
            # Parse parameters
            cla_val = None
            env_val = None
            lab_val = None
            
            for part in parts:
                if part.startswith("cla"):
                    cla_val = parse_val(part[3:])
                elif part.startswith("env"):
                    env_val = parse_val(part[3:])
                elif part.startswith("lab"):
                    lab_val = parse_val(part[3:])
            
            # Read JSON
            try:
                with open(f) as fp:
                    result = json.load(fp)
                
                # Try to find F1 score
                f1 = None
                if "metrics" in result and "macro_f1" in result["metrics"]:
                    f1 = result["metrics"]["macro_f1"]
                elif "macro_f1" in result:
                    f1 = result["macro_f1"]
                elif "f1" in result:
                    f1 = result["f1"]
                
                if f1 is not None and cla_val is not None and env_val is not None:
                    key = (cla_val, env_val, lab_val)
                    if key not in data_by_condition:
                        data_by_condition[key] = []
                    data_by_condition[key].append(f1)
                    
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")
        
        # Show summary
        if data_by_condition:
            print(f"\nConditions found: {len(data_by_condition)}")
            
            # Group by label noise = 0.05
            target_noise = 0.05
            filtered_data = {(c, e): vals for (c, e, l), vals in data_by_condition.items() 
                           if abs(l - target_noise) < 0.01}
            
            print(f"Conditions with label_noise={target_noise}: {len(filtered_data)}")
            
            if filtered_data:
                print("\nPerformance by condition:")
                for (cla, env), vals in sorted(filtered_data.items()):
                    mean_f1 = st.mean(vals) if vals else 0
                    print(f"  Class={cla:.1f}, Env={env:.1f}: "
                          f"{len(vals)} points, mean={mean_f1:.3f}, "
                          f"values={[f'{v:.3f}' for v in vals[:3]]}")
                
                # Check for empty data issue
                all_vals = [v for vals in filtered_data.values() for v in vals]
                if all_vals:
                    print(f"\nOverall: mean={st.mean(all_vals):.3f}, "
                          f"min={min(all_vals):.3f}, max={max(all_vals):.3f}")
                else:
                    print("\nWARNING: No valid F1 scores found!")
        else:
            print(f"  WARNING: No data collected for {model}")

if __name__ == "__main__":
    collect_debug_data()