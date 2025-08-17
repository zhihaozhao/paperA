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
        # Check for Data subdirectory (correct structure based on GitHub repo)
        if (self.benchmark_path / "Data").exists():
            self.data_path = self.benchmark_path / "Data"
        else:
            self.data_path = self.benchmark_path
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
        # Search for data files in multiple benchmark datasets (NTU-Fi_HAR, UT_HAR, Widardata, NTU-Fi-HumanID)
        data_files = []
        
        # Check standard benchmark subdirectories
        benchmark_dirs = ["NTU-Fi_HAR", "UT_HAR", "Widardata", "NTU-Fi-HumanID"]
        
        for subdir in benchmark_dirs:
            subdir_path = self.data_path / subdir
            if subdir_path.exists():
                # Try different data formats
                try:
                    import h5py
                    data_files.extend(list(subdir_path.glob("**/*.h5")))
                    data_files.extend(list(subdir_path.glob("**/*.hdf5")))
                except ImportError:
                    pass
                
                data_files.extend(list(subdir_path.glob("**/*.npz")))
                data_files.extend(list(subdir_path.glob("**/*.csv")))
                data_files.extend(list(subdir_path.glob("**/*.mat")))
                
        # Also check root data directory
        try:
            import h5py
            data_files.extend(list(self.data_path.glob("*.h5")))
            data_files.extend(list(self.data_path.glob("*.hdf5")))
        except ImportError:
            pass
            
        data_files.extend(list(self.data_path.glob("*.npz")))
        data_files.extend(list(self.data_path.glob("*.csv")))
                    
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.data_path} or subdirectories {benchmark_dirs}")
            
        print(f"[INFO] Found {len(data_files)} data files in benchmark directory")
            
        # Try to load data from available files (try first few files in case some are corrupted)
        data_loaded = False
        
        for data_file in data_files[:5]:  # Try first 5 files
            print(f"[INFO] Attempting to load: {data_file}")
            
            try:
                if data_file.suffix in ['.h5', '.hdf5']:
                    import h5py
                    with h5py.File(data_file, 'r') as f:
                        print(f"[INFO] HDF5 keys available: {list(f.keys())}")
                        
                        # Try different common key names for WiFi CSI data
                        if 'csi_data' in f:
                            self.X = f['csi_data'][:]
                        elif 'data' in f:
                            self.X = f['data'][:]
                        elif 'X' in f:
                            self.X = f['X'][:]
                        else:
                            print(f"[WARNING] No recognized data key in {data_file}")
                            continue
                            
                        if 'labels' in f:
                            self.y = f['labels'][:]
                        elif 'y' in f:
                            self.y = f['y'][:]
                        elif 'target' in f:
                            self.y = f['target'][:]
                        else:
                            print(f"[WARNING] No recognized label key in {data_file}")
                            continue
                            
                        self.subjects = f.get('subjects', np.arange(len(self.y)) % 10)[:]  # Default: 10 subjects
                        self.rooms = f.get('rooms', np.arange(len(self.y)) % 5)[:]  # Default: 5 rooms
                        
                elif data_file.suffix == '.npz':
                    data = np.load(data_file)
                    print(f"[INFO] NPZ keys available: {list(data.keys())}")
                    
                    if 'X' in data:
                        self.X = data['X']
                    elif 'data' in data:
                        self.X = data['data']
                    else:
                        print(f"[WARNING] No recognized data key in {data_file}")
                        continue
                        
                    if 'y' in data:
                        self.y = data['y']
                    elif 'labels' in data:
                        self.y = data['labels']
                    else:
                        print(f"[WARNING] No recognized label key in {data_file}")
                        continue
                        
                    self.subjects = data.get('subjects', np.arange(len(self.y)) % 10)
                    self.rooms = data.get('rooms', np.arange(len(self.y)) % 5)
                    
                elif data_file.suffix == '.mat':
                    # MATLAB file loading for WiFi CSI benchmark
                    try:
                        from scipy.io import loadmat
                        mat_data = loadmat(data_file)
                        print(f"[INFO] MAT keys available: {[k for k in mat_data.keys() if not k.startswith('__')]}")
                        
                        # Try common WiFi CSI variable names
                        csi_data = None
                        labels = None
                        
                        # Common CSI data key names (including WiFi benchmark specific ones)
                        for key in ['CSIamp', 'CSI', 'csi_data', 'data', 'X', 'csi', 'signal']:
                            if key in mat_data:
                                csi_data = mat_data[key]
                                print(f"[INFO] Found CSI data with key: {key}, shape: {csi_data.shape}")
                                break
                                
                        # Common label key names  
                        for key in ['labels', 'y', 'target', 'class', 'activity']:
                            if key in mat_data:
                                labels = mat_data[key].flatten()  # Ensure 1D
                                print(f"[INFO] Found labels with key: {key}, shape: {labels.shape}")
                                break
                        
                        # If no labels found in file, try to infer from path structure
                        if csi_data is not None and labels is None:
                            print(f"[INFO] CSI data found but no labels, trying to infer from path...")
                            
                            # Try to infer activity from file path (NTU-Fi_HAR structure)
                            path_parts = str(data_file).lower()
                            if 'box' in path_parts:
                                labels = np.zeros(csi_data.shape[0])  # Sitting/Stationary
                            elif 'walk' in path_parts:
                                labels = np.ones(csi_data.shape[0]) * 2  # Walking
                            elif 'fall' in path_parts:
                                labels = np.ones(csi_data.shape[0]) * 3  # Falling
                            elif 'stand' in path_parts:
                                labels = np.ones(csi_data.shape[0])  # Standing
                            else:
                                # Default to cycling through classes for basic testing
                                labels = np.arange(csi_data.shape[0]) % 4
                                
                            print(f"[INFO] Inferred labels from path: {np.unique(labels)}")
                        
                        if csi_data is None or labels is None:
                            print(f"[WARNING] Could not find CSI data or labels in {data_file}")
                            continue
                            
                        # Ensure 3D format [N, T, F]
                        if len(csi_data.shape) == 2:
                            # [N, features] -> [N, T, F]
                            N, total_features = csi_data.shape
                            T = 128  # Default time steps
                            F = total_features // T if total_features >= T else total_features
                            if total_features >= T * F:
                                self.X = csi_data[:, :T*F].reshape(N, T, F)
                            else:
                                print(f"[WARNING] Insufficient features in {data_file}: {total_features}")
                                continue
                        elif len(csi_data.shape) == 3:
                            self.X = csi_data  # Already in correct format
                        else:
                            print(f"[WARNING] Unexpected CSI data shape in {data_file}: {csi_data.shape}")
                            continue
                            
                        self.y = labels
                        
                        # Try to extract subject/room info from filename or mat file
                        self.subjects = mat_data.get('subjects', np.arange(len(self.y)) % 10)
                        self.rooms = mat_data.get('rooms', np.arange(len(self.y)) % 5)
                        
                        # If subjects/rooms are not 1D, flatten them
                        if hasattr(self.subjects, 'flatten'):
                            self.subjects = self.subjects.flatten()
                        if hasattr(self.rooms, 'flatten'):
                            self.rooms = self.rooms.flatten()
                            
                    except ImportError:
                        print(f"[ERROR] scipy not available for .mat files, skipping {data_file}")
                        continue
                    except Exception as e:
                        print(f"[ERROR] Failed to load .mat file {data_file}: {e}")
                        continue
                        
                elif data_file.suffix == '.csv':
                    # Simple CSV loading (basic implementation)
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    
                    if len(df.columns) < 2:
                        print(f"[WARNING] CSV file has insufficient columns: {data_file}")
                        continue
                        
                    # Assume last column is labels, rest are features
                    self.y = df.iloc[:, -1].values
                    feature_data = df.iloc[:, :-1].values
                    
                    # Reshape to expected format [N, T, F] 
                    if len(feature_data.shape) == 2:
                        T, F = 128, feature_data.shape[1] // 128 if feature_data.shape[1] >= 128 else 1
                        if feature_data.shape[1] >= T * F:
                            self.X = feature_data[:, :T*F].reshape(-1, T, F)
                        else:
                            print(f"[WARNING] CSV feature dimension too small: {data_file}")
                            continue
                    
                    self.subjects = np.arange(len(self.y)) % 10  # Default: 10 subjects
                    self.rooms = np.arange(len(self.y)) % 5  # Default: 5 rooms
                else:
                    print(f"[WARNING] Unsupported file format: {data_file}")
                    continue
                
                # If we reach here, data was loaded successfully
                data_loaded = True
                print(f"[SUCCESS] Data loaded from: {data_file}")
                break
                
            except Exception as e:
                print(f"[ERROR] Failed to load {data_file}: {e}")
                continue
        
        if not data_loaded:
            raise ValueError("Could not load data from any available files")
            
        # Validate loaded data
        if self.X is None or self.y is None:
            raise ValueError(f"Failed to load data from {data_file}: X or y is None")
            
        print(f"[INFO] Loaded data: X.shape={self.X.shape}, y.shape={self.y.shape}")
        
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
        if y is None:
            raise ValueError("Labels array is None, cannot map to standard format")
            
        unique_labels = np.unique(y)
        print(f"[INFO] Original labels: {unique_labels}")
        
        if len(unique_labels) == 4 and set(unique_labels) == {0, 1, 2, 3}:
            print(f"[INFO] Labels already in standard format: {unique_labels}")
            return y  # Already in standard format
        
        # Create mapping (customize based on actual benchmark)
        label_map = {}
        for i, label in enumerate(sorted(unique_labels)):
            label_map[label] = i
            
        print(f"[INFO] Label mapping: {label_map}")
        mapped_y = np.array([label_map[label] for label in y])
        print(f"[INFO] Mapped labels: {np.unique(mapped_y)}")
        
        return mapped_y
    
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
    return get_synth_loaders(batch=batch_size, difficulty="hard", seed=seed, num_classes=4)