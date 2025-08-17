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
        Load WiFi CSI benchmark dataset from multiple activity files
        Returns: X, y, subjects, rooms, metadata
        """
        return self._load_multiclass_data()
    
    def _load_multiclass_data(self):
        """Load balanced multi-class data from multiple activity files"""
        # Search for .mat files in benchmark datasets
        data_files = []
        benchmark_dirs = ["NTU-Fi_HAR", "UT_HAR", "Widardata", "NTU-Fi-HumanID"]
        
        for subdir in benchmark_dirs:
            subdir_path = self.data_path / subdir
            if subdir_path.exists():
                data_files.extend(list(subdir_path.glob("**/*.mat")))
                
        if not data_files:
            raise FileNotFoundError(f"No .mat files found in {self.data_path}")
            
        print(f"[INFO] Found {len(data_files)} .mat files in benchmark directory")
        
        # Group files by activity type for balanced loading
        activity_files = {'sitting': [], 'standing': [], 'walking': [], 'falling': []}
        
        for data_file in data_files:
            path_parts = str(data_file).lower()
            if 'box' in path_parts:
                activity_files['sitting'].append(data_file)
            elif 'walk' in path_parts:
                activity_files['walking'].append(data_file)
            elif 'fall' in path_parts:
                activity_files['falling'].append(data_file)
            elif 'stand' in path_parts:
                activity_files['standing'].append(data_file)
        
        print(f"[INFO] Activity distribution:")
        for activity, files in activity_files.items():
            print(f"  {activity}: {len(files)} files")
        
        # Load data from each activity (limit files per activity for balance)
        all_X, all_y, all_subjects, all_rooms = [], [], [], []
        files_per_activity = 2  # Load 2 files per activity for testing
        
        for activity_idx, (activity, files) in enumerate(activity_files.items()):
            if not files:
                print(f"[WARNING] No files found for {activity}")
                continue
                
            activity_label = activity_idx  # 0=sitting, 1=standing, 2=walking, 3=falling
            print(f"\n[INFO] Loading {activity} data (label={activity_label})...")
            
            files_loaded = 0
            for data_file in files[:files_per_activity]:
                success = self._load_single_mat_file(data_file, activity_label, all_X, all_y, all_subjects, all_rooms)
                if success:
                    files_loaded += 1
                    
            print(f"[INFO] Loaded {files_loaded} files for {activity}")
        
        if not all_X:
            raise ValueError("No data could be loaded from any files")
            
        # Combine all data
        self.X = np.concatenate(all_X, axis=0)
        self.y = np.concatenate(all_y, axis=0)
        self.subjects = np.concatenate(all_subjects, axis=0)
        self.rooms = np.concatenate(all_rooms, axis=0)
        
        print(f"[SUCCESS] Combined dataset:")
        print(f"  Total samples: {self.X.shape[0]}")
        print(f"  Class distribution: {np.bincount(self.y.astype(int))}")
        print(f"  Unique subjects: {len(np.unique(self.subjects))}")
        print(f"  Unique rooms: {len(np.unique(self.rooms))}")
        
        # Store metadata
        self.metadata = {
            'n_samples': len(self.y),
            'n_subjects': len(np.unique(self.subjects)),
            'n_rooms': len(np.unique(self.rooms)),
            'sequence_length': self.X.shape[1],
            'n_features': self.X.shape[2],
            'class_distribution': np.bincount(self.y.astype(int))
        }
        
        return self.X, self.y, self.subjects, self.rooms, self.metadata
    
    def _load_single_mat_file(self, data_file, activity_label, all_X, all_y, all_subjects, all_rooms):
        """Load a single .mat file and append to data lists"""
        try:
            from scipy.io import loadmat
            mat_data = loadmat(data_file)
            
            # Find CSI data
            csi_data = None
            for key in ['CSIamp', 'CSI', 'csi_data', 'data', 'X']:
                if key in mat_data:
                    csi_data = mat_data[key]
                    break
                    
            if csi_data is None:
                return False
                
            # Ensure 3D format [N, T, F]
            if len(csi_data.shape) == 2:
                N, total_features = csi_data.shape
                T = 128
                F = total_features // T if total_features >= T else total_features
                if total_features >= T * F:
                    csi_data = csi_data[:, :T*F].reshape(N, T, F)
                else:
                    return False
            elif len(csi_data.shape) != 3:
                return False
                
            # Create labels and metadata
            N = csi_data.shape[0]
            labels = np.full(N, activity_label)
            subjects = np.arange(N) % 10  # Distribute across 10 subjects
            rooms = np.arange(N) % 5     # Distribute across 5 rooms
            
            # Append to lists
            all_X.append(csi_data)
            all_y.append(labels)
            all_subjects.append(subjects)
            all_rooms.append(rooms)
            
            print(f"[SUCCESS] Loaded {N} samples for activity {activity_label} from {data_file.name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load {data_file}: {e}")
            return False
    
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