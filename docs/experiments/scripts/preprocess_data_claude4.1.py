#!/usr/bin/env python3
"""
CSI Data Preprocessing Pipeline
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm

class CSIPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_raw_csi(self, file_path):
        """Load raw CSI data from various formats"""
        if file_path.suffix == '.mat':
            import scipy.io
            data = scipy.io.loadmat(file_path)
            csi = data['csi_data']
        elif file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                csi = f['csi'][:]
        elif file_path.suffix == '.npy':
            csi = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        return csi
    
    def phase_sanitization(self, csi_complex):
        """Remove random phase offset"""
        phase = np.angle(csi_complex)
        # Linear fitting across subcarriers
        subcarriers = np.arange(phase.shape[1])
        for t in range(phase.shape[0]):
            for ant in range(phase.shape[2]):
                # Fit linear phase
                coef = np.polyfit(subcarriers, phase[t, :, ant], 1)
                phase_fit = np.polyval(coef, subcarriers)
                phase[t, :, ant] -= phase_fit
        
        # Reconstruct complex CSI
        amplitude = np.abs(csi_complex)
        csi_sanitized = amplitude * np.exp(1j * phase)
        return csi_sanitized
    
    def amplitude_normalization(self, csi_amplitude):
        """Normalize amplitude per antenna"""
        normalized = np.zeros_like(csi_amplitude)
        for ant in range(csi_amplitude.shape[2]):
            ant_data = csi_amplitude[:, :, ant]
            # Min-max normalization
            min_val = ant_data.min()
            max_val = ant_data.max()
            if max_val > min_val:
                normalized[:, :, ant] = (ant_data - min_val) / (max_val - min_val)
            else:
                normalized[:, :, ant] = ant_data
        return normalized
    
    def temporal_filtering(self, csi_data, fs=100):
        """Apply bandpass filter for human motion"""
        # Design Butterworth filter
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 20 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter per subcarrier-antenna pair
        filtered = np.zeros_like(csi_data)
        for sub in range(csi_data.shape[1]):
            for ant in range(csi_data.shape[2]):
                filtered[:, sub, ant] = signal.filtfilt(b, a, csi_data[:, sub, ant])
        
        return filtered
    
    def segment_windows(self, csi_data, labels, window_size=1000, stride=500):
        """Segment into fixed-length windows"""
        windows = []
        window_labels = []
        
        for start in range(0, len(csi_data) - window_size + 1, stride):
            end = start + window_size
            windows.append(csi_data[start:end])
            # Majority voting for label
            window_label = np.bincount(labels[start:end]).argmax()
            window_labels.append(window_label)
        
        return np.array(windows), np.array(window_labels)
    
    def augment_data(self, csi_data, augmentation_config):
        """Data augmentation for training"""
        augmented = []
        
        for sample in csi_data:
            # Time shift
            if augmentation_config.get('time_shift', False):
                shift = np.random.randint(-10, 10)
                sample = np.roll(sample, shift, axis=0)
            
            # Amplitude scaling
            if augmentation_config.get('amplitude_scale', False):
                scale = np.random.uniform(0.8, 1.2)
                sample = sample * scale
            
            # Add noise
            if augmentation_config.get('add_noise', False):
                noise = np.random.normal(0, 0.01, sample.shape)
                sample = sample + noise
            
            augmented.append(sample)
        
        return np.array(augmented)
    
    def process_dataset(self, input_dir, output_dir):
        """Process entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_data = []
        all_labels = []
        
        # Process each file
        for file_path in tqdm(list(input_path.glob('*.mat'))):
            # Load data
            csi_complex = self.load_raw_csi(file_path)
            
            # Load labels (assuming label file exists)
            label_file = file_path.with_suffix('.labels')
            if label_file.exists():
                labels = np.load(label_file)
            else:
                # Default labels for testing
                labels = np.zeros(len(csi_complex))
            
            # Preprocessing pipeline
            csi_sanitized = self.phase_sanitization(csi_complex)
            csi_amplitude = np.abs(csi_sanitized)
            csi_normalized = self.amplitude_normalization(csi_amplitude)
            csi_filtered = self.temporal_filtering(csi_normalized)
            
            # Segmentation
            windows, window_labels = self.segment_windows(
                csi_filtered, labels,
                window_size=self.config['window_size'],
                stride=self.config['stride']
            )
            
            all_data.extend(windows)
            all_labels.extend(window_labels)
        
        # Convert to arrays
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        
        # Split into train/val/test
        n_samples = len(all_data)
        n_train = int(0.6 * n_samples)
        n_val = int(0.2 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Save processed data
        np.save(output_path / 'train_data.npy', all_data[train_idx])
        np.save(output_path / 'train_labels.npy', all_labels[train_idx])
        np.save(output_path / 'val_data.npy', all_data[val_idx])
        np.save(output_path / 'val_labels.npy', all_labels[val_idx])
        np.save(output_path / 'test_data.npy', all_data[test_idx])
        np.save(output_path / 'test_labels.npy', all_labels[test_idx])
        
        # Save statistics
        stats = {
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'n_classes': len(np.unique(all_labels)),
            'window_size': self.config['window_size'],
            'stride': self.config['stride'],
            'shape': all_data[0].shape
        }
        
        import json
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset processed successfully!")
        print(f"Train: {stats['n_train']}, Val: {stats['n_val']}, Test: {stats['n_test']}")
        print(f"Shape: {stats['shape']}, Classes: {stats['n_classes']}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess CSI data')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--stride', type=int, default=500)
    parser.add_argument('--sampling_rate', type=int, default=100)
    
    args = parser.parse_args()
    
    config = {
        'window_size': args.window_size,
        'stride': args.stride,
        'sampling_rate': args.sampling_rate
    }
    
    preprocessor = CSIPreprocessor(config)
    preprocessor.process_dataset(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()