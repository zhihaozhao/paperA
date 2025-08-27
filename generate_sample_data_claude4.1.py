"""
Generate sample data for testing experiments
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
import os

def generate_ntu_fi_sample(data_dir, num_classes=6, samples_per_class=10):
    """Generate sample NTU-Fi format data"""
    print(f"Generating NTU-Fi sample data in {data_dir}")
    
    for mode in ['train_amp', 'test_amp']:
        mode_dir = Path(data_dir) / mode
        
        for class_id in range(num_classes):
            class_dir = mode_dir / f'class_{class_id}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for sample_id in range(samples_per_class):
                # Generate random CSI data
                # Shape: [342, 2000] (3 antennas * 114 subcarriers, 2000 time steps)
                csi_data = np.random.randn(342, 2000) * 10 + 42.3199
                
                # Save as .mat file
                mat_data = {'CSIamp': csi_data}
                sio.savemat(class_dir / f'sample_{sample_id}.mat', mat_data)
    
    print(f"  Created {num_classes} classes with {samples_per_class} samples each")

def generate_ut_har_sample(data_dir):
    """Generate sample UT-HAR format data"""
    print(f"Generating UT-HAR sample data in {data_dir}")
    
    for mode in ['train', 'val', 'test']:
        # Generate random data
        num_samples = 100 if mode == 'train' else 50
        data = np.random.randn(num_samples, 250, 90)
        labels = np.random.randint(0, 7, num_samples)
        
        # Save as .csv (actually numpy arrays)
        data_path = Path(data_dir) / 'data' / f'{mode}_data.csv'
        label_path = Path(data_dir) / 'label' / f'{mode}_label.csv'
        
        data_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'wb') as f:
            np.save(f, data)
        with open(label_path, 'wb') as f:
            np.save(f, labels)
    
    print(f"  Created train/val/test splits")

def generate_widar_sample(data_dir, num_classes=6, samples_per_class=10):
    """Generate sample Widar format data"""
    print(f"Generating Widar sample data in {data_dir}")
    
    for mode in ['train', 'test']:
        mode_dir = Path(data_dir) / mode
        
        for class_id in range(num_classes):
            class_dir = mode_dir / f'gesture_{class_id}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for sample_id in range(samples_per_class):
                # Generate random BVP data
                # Shape: [22*400] = 8800 values
                data = np.random.randn(8800) * 0.0119 + 0.0025
                
                # Save as CSV
                csv_path = class_dir / f'sample_{sample_id}.csv'
                np.savetxt(csv_path, data.reshape(-1, 1), delimiter=',')
    
    print(f"  Created {num_classes} classes with {samples_per_class} samples each")

if __name__ == "__main__":
    print("Generating sample data for testing...")
    print("=" * 50)
    
    # Generate samples for each dataset
    generate_ntu_fi_sample('Data/NTU-Fi_HAR')
    generate_ut_har_sample('Data/UT_HAR')
    generate_widar_sample('Data/Widardata')
    
    print("\nSample data generation complete!")
    print("You can now run experiments with this test data.")
