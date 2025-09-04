#!/usr/bin/env python3
"""
Master script to extract all experimental data for paper figures and tables.
This ensures complete reproducibility from raw experimental results to paper figures.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import individual extraction modules
from extract_srv_data import extract_srv_performance
from extract_cross_domain_data import extract_cross_domain_performance
from extract_calibration_data import extract_calibration_metrics
from extract_label_efficiency_data import extract_label_efficiency
from extract_fall_detection_data import extract_fall_detection_performance
from extract_table1_data import extract_table1_data

def ensure_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "extracted_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

def main():
    """Run all data extraction pipelines"""
    
    print("="*80)
    print("MASTER DATA EXTRACTION PIPELINE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*80)
    
    # Ensure output directory exists
    output_dir = ensure_directories()
    
    # Track extraction status
    extraction_status = {}
    
    # 1. Extract SRV data for Figure 2(c)
    print("\n[1/6] Extracting SRV Performance Data (Figure 2c)...")
    try:
        srv_data = extract_srv_performance()
        output_file = output_dir / "srv_performance.json"
        with open(output_file, 'w') as f:
            json.dump(srv_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['srv'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['srv'] = f'FAILED: {e}'
    
    # 2. Extract Cross-Domain data for Figure 3
    print("\n[2/6] Extracting Cross-Domain Performance (Figure 3)...")
    try:
        cross_domain_data = extract_cross_domain_performance()
        output_file = output_dir / "cross_domain_performance.json"
        with open(output_file, 'w') as f:
            json.dump(cross_domain_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['cross_domain'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['cross_domain'] = f'FAILED: {e}'
    
    # 3. Extract Calibration data for Figure 4
    print("\n[3/6] Extracting Calibration Metrics (Figure 4)...")
    try:
        calibration_data = extract_calibration_metrics()
        output_file = output_dir / "calibration_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['calibration'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['calibration'] = f'FAILED: {e}'
    
    # 4. Extract Label Efficiency data for Figure 5
    print("\n[4/6] Extracting Label Efficiency Data (Figure 5)...")
    try:
        label_data = extract_label_efficiency()
        output_file = output_dir / "label_efficiency.json"
        with open(output_file, 'w') as f:
            json.dump(label_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['label_efficiency'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['label_efficiency'] = f'FAILED: {e}'
    
    # 5. Extract Fall Detection data for Figure 6
    print("\n[5/6] Extracting Fall Detection Performance (Figure 6)...")
    try:
        fall_data = extract_fall_detection_performance()
        output_file = output_dir / "fall_detection_performance.json"
        with open(output_file, 'w') as f:
            json.dump(fall_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['fall_detection'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['fall_detection'] = f'FAILED: {e}'
    
    # 6. Extract Table 1 data
    print("\n[6/6] Extracting Table 1 Data...")
    try:
        table1_data = extract_table1_data()
        output_file = output_dir / "table1_data.json"
        with open(output_file, 'w') as f:
            json.dump(table1_data, f, indent=2)
        print(f"✅ Saved to {output_file}")
        extraction_status['table1'] = 'SUCCESS'
    except Exception as e:
        print(f"❌ Failed: {e}")
        extraction_status['table1'] = f'FAILED: {e}'
    
    # Save extraction status
    status_file = output_dir / "extraction_status.json"
    with open(status_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'status': extraction_status
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    success_count = sum(1 for v in extraction_status.values() if v == 'SUCCESS')
    total_count = len(extraction_status)
    
    print(f"Successfully extracted: {success_count}/{total_count}")
    for name, status in extraction_status.items():
        symbol = "✅" if status == 'SUCCESS' else "❌"
        print(f"  {symbol} {name}: {status}")
    
    print(f"\nAll extracted data saved to: {output_dir}")
    print(f"Status log saved to: {status_file}")
    
    if success_count == total_count:
        print("\n🎉 All data extraction completed successfully!")
        return 0
    else:
        print("\n⚠️ Some extractions failed. Please check the logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())