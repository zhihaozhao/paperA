import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

class RealCSIDataset(Dataset):
    def __init__(self, X, y, metadata=None):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
        self.metadata = metadata or {}
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), int(self.y[i])

class BenchmarkCSIDataset:
    """WiFi CSI Benchmark Dataset Loader with LOSO/LORO Support"""
    
    def __init__(self, benchmark_path: str = "benchmarks/WiFi-CSI-Sensing-Benchmark-main"):
        self.benchmark_path = Path(benchmark_path)
        self.X = None
        self.y = None
        self.subjects = None
        self.rooms = None
        self.metadata = {}
        
    def load_wifi_csi_benchmark(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load WiFi CSI benchmark dataset
        Returns: X, y, subjects, rooms, metadata
        """
        # Try multiple common data formats for WiFi CSI benchmarks
        try:
            import h5py
            data_files = list(self.benchmark_path.glob("*.h5")) + \
                        list(self.benchmark_path.glob("*.hdf5")) + \
                        list(self.benchmark_path.glob("data/*.h5"))
        except ImportError:
            data_files = []
            
        data_files += list(self.benchmark_path.glob("*.npz"))
                    
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.benchmark_path}")
            
        # Load the first available data file
        data_file = data_files[0]
        
        if data_file.suffix in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(data_file, 'r') as f:
                self.X = f['csi_data'][:]  # Shape: [N, T, F]
                self.y = f['labels'][:]    # Shape: [N,]
                self.subjects = f.get('subjects', np.zeros(len(self.y)))[:]
                self.rooms = f.get('rooms', np.zeros(len(self.y)))[:]
                
                # Load metadata if available
                if 'metadata' in f:
                    for key in f['metadata'].keys():
                        self.metadata[key] = f['metadata'][key][()]
                        
        elif data_file.suffix == '.npz':
            data = np.load(data_file)
            self.X = data['X']
            self.y = data['y'] 
            self.subjects = data.get('subjects', np.zeros(len(self.y)))
            self.rooms = data.get('rooms', np.zeros(len(self.y)))
            
        # Map labels to standard 4-class format if needed
        self.y = self._map_labels_to_standard(self.y)
        
        # Store basic metadata
        self.metadata.update({
            'n_samples': len(self.y),
            'n_subjects': len(np.unique(self.subjects)),
            'n_rooms': len(np.unique(self.rooms)),
            'sequence_length': self.X.shape[1],
            'n_features': self.X.shape[2],
            'class_distribution': np.bincount(self.y)
        })
        
        return self.X, self.y, self.subjects, self.rooms, self.metadata
    
    def _map_labels_to_standard(self, y: np.ndarray) -> np.ndarray:
        """Map labels to standard format: 0=Sitting, 1=Standing, 2=Walking, 3=Falling"""
        unique_labels = np.unique(y)
        if len(unique_labels) == 4 and set(unique_labels) == {0, 1, 2, 3}:
            return y  # Already in standard format
        
        # Create mapping (customize based on actual benchmark)
        label_map = {}
        for i, label in enumerate(sorted(unique_labels)):
            label_map[label] = i
            
        return np.array([label_map[label] for label in y])
    
    def create_loso_splits(self, subjects: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate LOSO cross-validation splits"""
        splits = []
        unique_subjects = np.unique(subjects)
        
        for test_subject in unique_subjects:
            train_idx = np.where(subjects != test_subject)[0]
            test_idx = np.where(subjects == test_subject)[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                
        return splits
    
    def create_loro_splits(self, rooms: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate LORO cross-validation splits"""
        splits = []
        unique_rooms = np.unique(rooms)
        
        for test_room in unique_rooms:
            train_idx = np.where(rooms != test_room)[0]
            test_idx = np.where(rooms == test_room)[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
                
        return splits

def get_real_loaders_loso(X, y, subjects, test_subj, batch=64):
    tr_idx = np.where(subjects != test_subj)[0]; te_idx = np.where(subjects == test_subj)[0]
    tr = RealCSIDataset(X[tr_idx], y[tr_idx]); te = RealCSIDataset(X[te_idx], y[te_idx])
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)

def get_real_loaders_loro(X, y, rooms, test_room, batch=64):
    """LORO version of the above function"""
    tr_idx = np.where(rooms != test_room)[0]; te_idx = np.where(rooms == test_room)[0]
    tr = RealCSIDataset(X[tr_idx], y[tr_idx]); te = RealCSIDataset(X[te_idx], y[te_idx])
    return DataLoader(tr, batch_size=batch, shuffle=True), DataLoader(te, batch_size=batch)

def get_sim2real_loaders(X, y, label_ratio=0.1, seed=0, batch=64):
    """Create limited labeled real data loaders for Sim2Real experiments"""
    rng = np.random.default_rng(seed)
    
    # Stratified sampling to maintain class balance
    labeled_idx = []
    for class_id in np.unique(y):
        class_idx = np.where(y == class_id)[0]
        n_class_labeled = max(1, int(len(class_idx) * label_ratio))
        labeled_idx.extend(rng.choice(class_idx, n_class_labeled, replace=False))
    
    labeled_idx = np.array(labeled_idx)
    unlabeled_idx = np.setdiff1d(np.arange(len(y)), labeled_idx)
    
    # Create train/test loaders
    train_ds = RealCSIDataset(X[labeled_idx], y[labeled_idx])
    test_ds = RealCSIDataset(X[unlabeled_idx], y[unlabeled_idx])
    
    return (DataLoader(train_ds, batch_size=batch, shuffle=True), 
            DataLoader(test_ds, batch_size=batch, shuffle=False))

def get_real_loaders(dataset="default", batch_size=64, seed=0, split_ratio=0.8):
    """
    Generic real data loader for cross-domain experiments
    This is a placeholder implementation that uses synthetic data for now
    
    TODO: Replace with actual benchmark data loading when WiFi-CSI-Sensing-Benchmark is integrated
    """
    # For now, fall back to synthetic data since benchmark integration is in progress
    from src.data_synth import get_synth_loaders
    return get_synth_loaders(batch=batch_size, difficulty="hard", seed=seed)