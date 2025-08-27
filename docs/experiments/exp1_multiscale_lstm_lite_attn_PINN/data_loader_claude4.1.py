"""
Data loader for CSI-based HAR with support for public datasets
Supports: SignFi, synthetic data, and standard CSI format
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import h5py
from typing import Dict, List, Tuple, Optional
import random
from scipy import signal

class CSIDataset(Dataset):
    """Generic CSI dataset loader"""
    
    def __init__(self, 
                 data_path: str,
                 mode: str = 'train',
                 window_size: int = 100,
                 stride: int = 50,
                 normalize: bool = True,
                 augment: bool = False):
        """
        Args:
            data_path: Path to dataset
            mode: 'train', 'val', or 'test'
            window_size: Length of CSI window
            stride: Stride for sliding window
            normalize: Whether to normalize data
            augment: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.augment = augment and (mode == 'train')
        
        # Load data
        self.data, self.labels = self._load_data()
        
        # Create windows
        self.windows = self._create_windows()
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSI data and labels"""
        # Try different file formats
        if (self.data_path / f'{self.mode}_data.npy').exists():
            # NumPy format
            data = np.load(self.data_path / f'{self.mode}_data.npy')
            labels = np.load(self.data_path / f'{self.mode}_labels.npy')
        elif (self.data_path / f'{self.mode}.h5').exists():
            # HDF5 format
            with h5py.File(self.data_path / f'{self.mode}.h5', 'r') as f:
                data = f['data'][:]
                labels = f['labels'][:]
        else:
            # Generate synthetic data for testing
            print(f"Warning: Dataset not found at {self.data_path}, generating synthetic data")
            data, labels = self._generate_synthetic_data()
        
        return data, labels
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic CSI data for testing"""
        num_samples = 1000 if self.mode == 'train' else 200
        num_classes = 6
        time_steps = 1000
        num_subcarriers = 30
        num_antennas = 3
        
        data = []
        labels = []
        
        for _ in range(num_samples):
            # Random class
            label = random.randint(0, num_classes - 1)
            
            # Generate class-specific pattern
            base_freq = 0.5 + label * 0.2  # Different frequencies for different classes
            time = np.linspace(0, 10, time_steps)
            
            # Create CSI pattern
            csi = np.zeros((time_steps, num_subcarriers, num_antennas))
            
            for sub in range(num_subcarriers):
                for ant in range(num_antennas):
                    # Base signal
                    signal_real = np.sin(2 * np.pi * base_freq * time + sub * 0.1 + ant * 0.05)
                    
                    # Add class-specific modulation
                    if label == 0:  # Static
                        signal_real += np.random.normal(0, 0.1, time_steps)
                    elif label == 1:  # Walking
                        signal_real *= (1 + 0.3 * np.sin(2 * np.pi * 2 * time))
                    elif label == 2:  # Running
                        signal_real *= (1 + 0.5 * np.sin(2 * np.pi * 4 * time))
                    elif label == 3:  # Sitting down
                        signal_real[time_steps//2:] *= 0.5
                    elif label == 4:  # Standing up
                        signal_real[:time_steps//2] *= 0.5
                    else:  # Waving
                        signal_real *= (1 + 0.4 * np.sin(2 * np.pi * 3 * time))
                    
                    # Add noise
                    signal_real += np.random.normal(0, 0.05, time_steps)
                    
                    # Store amplitude (simplified - real data would be complex)
                    csi[:, sub, ant] = np.abs(signal_real)
            
            data.append(csi)
            labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def _create_windows(self) -> List[Tuple[int, int, int]]:
        """Create sliding windows over the data"""
        windows = []
        
        for idx, sample in enumerate(self.data):
            sample_length = len(sample)
            
            if sample_length < self.window_size:
                # Pad if too short
                windows.append((idx, 0, sample_length))
            else:
                # Sliding window
                for start in range(0, sample_length - self.window_size + 1, self.stride):
                    windows.append((idx, start, start + self.window_size))
        
        return windows
    
    def _normalize_csi(self, csi: np.ndarray) -> np.ndarray:
        """Normalize CSI data"""
        # Min-max normalization per antenna
        for ant in range(csi.shape[-1]):
            ant_data = csi[..., ant]
            min_val = ant_data.min()
            max_val = ant_data.max()
            if max_val > min_val:
                csi[..., ant] = (ant_data - min_val) / (max_val - min_val)
        return csi
    
    def _augment_data(self, csi: np.ndarray, label: int) -> np.ndarray:
        """Apply data augmentation"""
        # Time shift
        if random.random() < 0.5:
            shift = random.randint(-5, 5)
            csi = np.roll(csi, shift, axis=0)
        
        # Amplitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            csi = csi * scale
        
        # Add noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, csi.shape)
            csi = csi + noise
        
        # Random dropout (simulate missing packets)
        if random.random() < 0.3:
            dropout_mask = np.random.random(len(csi)) > 0.1
            csi[~dropout_mask] = 0
        
        return csi
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx, start, end = self.windows[idx]
        
        # Get CSI window
        csi = self.data[sample_idx][start:end].copy()
        label = self.labels[sample_idx]
        
        # Pad if necessary
        if len(csi) < self.window_size:
            padding = self.window_size - len(csi)
            csi = np.pad(csi, ((0, padding), (0, 0), (0, 0)), mode='constant')
        
        # Normalize
        if self.normalize:
            csi = self._normalize_csi(csi)
        
        # Augment
        if self.augment:
            csi = self._augment_data(csi, label)
        
        # Convert to tensor
        csi_tensor = torch.FloatTensor(csi)
        label_tensor = torch.LongTensor([label]).squeeze()
        
        return csi_tensor, label_tensor


class SignFiDataset(CSIDataset):
    """SignFi dataset specific loader"""
    
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load SignFi specific format"""
        # SignFi has specific structure
        # Try to load from standard location
        signfi_path = self.data_path / 'signfi_data'
        
        if signfi_path.exists():
            # Load SignFi format
            # This would need actual SignFi loading code
            pass
        
        # Fall back to parent implementation
        return super()._load_data()


def create_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """Create train, val, and test dataloaders"""
    
    data_path = config.get('data_path', './data')
    batch_size = config.get('batch_size', 32)
    window_size = config.get('window_size', 100)
    stride = config.get('stride', 50)
    num_workers = config.get('num_workers', 4)
    
    # Create datasets
    train_dataset = CSIDataset(
        data_path, 
        mode='train',
        window_size=window_size,
        stride=stride,
        normalize=True,
        augment=True
    )
    
    val_dataset = CSIDataset(
        data_path,
        mode='val',
        window_size=window_size,
        stride=window_size,  # No overlap for validation
        normalize=True,
        augment=False
    )
    
    test_dataset = CSIDataset(
        data_path,
        mode='test',
        window_size=window_size,
        stride=window_size,  # No overlap for test
        normalize=True,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test data loading
    config = {
        'data_path': './data',
        'batch_size': 4,
        'window_size': 100,
        'stride': 50,
        'num_workers': 0
    }
    
    dataloaders = create_dataloaders(config)
    
    # Test iteration
    for batch_idx, (data, labels) in enumerate(dataloaders['train']):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Labels shape {labels.shape}")
        if batch_idx >= 2:
            break