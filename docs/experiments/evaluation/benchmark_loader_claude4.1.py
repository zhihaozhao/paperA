"""
Data loader for WiFi-CSI-Sensing-Benchmark datasets
Supports: NTU-Fi HAR, UT-HAR, Widar
"""

import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from typing import Dict, List, Tuple, Optional


class NTUFiDataset(Dataset):
    """NTU-Fi dataset loader for HAR and Human ID"""
    
    def __init__(self, 
                 root_dir: str, 
                 mode: str = 'train',
                 task: str = 'HAR',  # 'HAR' or 'HumanID'
                 modal: str = 'amp',  # 'amp' or 'phase'
                 transform=None):
        """
        Args:
            root_dir: Root directory containing NTU-Fi data
            mode: 'train' or 'test'
            task: 'HAR' for activity recognition or 'HumanID' for identification
            modal: 'amp' for amplitude or 'phase' for phase
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.task = task
        self.modal = f'CSI{modal}'
        self.transform = transform
        
        # Build data directory
        if task == 'HAR':
            data_dir = self.root_dir / 'NTU-Fi_HAR' / f'{mode}_{modal}'
        else:
            data_dir = self.root_dir / 'NTU-Fi-HumanID' / f'{mode}_{modal}'
        
        # Get all data files
        self.data_list = sorted(glob.glob(str(data_dir / '*/*.mat')))
        self.folder = sorted(glob.glob(str(data_dir / '*/')))
        self.category = {Path(f).name: i for i, f in enumerate(self.folder)}
        
        print(f"Loaded {len(self.data_list)} samples from {len(self.category)} classes")
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample_path = self.data_list[idx]
        
        # Get label from folder name
        class_name = Path(sample_path).parent.name
        y = self.category[class_name]
        
        # Load CSI data
        data = sio.loadmat(sample_path)[self.modal]
        
        # Normalize (using statistics from original code)
        x = (data - 42.3199) / 4.9802
        
        # Downsample: 2000 -> 500
        x = x[:, ::4]
        
        # Reshape to [3, 114, 500] (3 antennas, 114 subcarriers, 500 time steps)
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y]).squeeze()
        
        return x, y


class UTHARDataset(Dataset):
    """UT-HAR dataset loader"""
    
    def __init__(self, root_dir: str, mode: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory containing UT_HAR data
            mode: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir) / 'UT_HAR'
        self.mode = mode
        self.transform = transform
        
        # Load data and labels
        data_path = self.root_dir / 'data' / f'{mode}_data.csv'
        label_path = self.root_dir / 'label' / f'{mode}_label.csv'
        
        # Load numpy arrays
        with open(data_path, 'rb') as f:
            self.data = np.load(f)
            # Reshape to [samples, 1, 250, 90]
            self.data = self.data.reshape(len(self.data), 1, 250, 90)
            # Normalize
            self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        
        with open(label_path, 'rb') as f:
            self.labels = np.load(f)
        
        print(f"Loaded {len(self.data)} samples for {mode}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y]).squeeze()
        
        return x, y


class WidarDataset(Dataset):
    """Widar dataset loader"""
    
    def __init__(self, root_dir: str, mode: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory containing Widar data
            mode: 'train' or 'test'
        """
        self.root_dir = Path(root_dir) / 'Widardata' / mode
        self.mode = mode
        self.transform = transform
        
        # Get all CSV files
        self.data_list = sorted(glob.glob(str(self.root_dir / '*/*.csv')))
        self.folder = sorted(glob.glob(str(self.root_dir / '*/')))
        self.category = {Path(f).name: i for i, f in enumerate(self.folder)}
        
        print(f"Loaded {len(self.data_list)} samples from {len(self.category)} classes")
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample_path = self.data_list[idx]
        
        # Get label from folder name
        class_name = Path(sample_path).parent.name
        y = self.category[class_name]
        
        # Load CSV data
        x = np.genfromtxt(sample_path, delimiter=',')
        
        # Normalize (using statistics from original code)
        x = (x - 0.0025) / 0.0119
        
        # Reshape: 22,400 -> 22,20,20
        x = x.reshape(22, 20, 20)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y]).squeeze()
        
        return x, y


class UnifiedCSIDataset(Dataset):
    """Unified dataset wrapper for different CSI datasets"""
    
    def __init__(self, 
                 dataset_name: str,
                 root_dir: str,
                 mode: str = 'train',
                 **kwargs):
        """
        Args:
            dataset_name: 'ntu-fi-har', 'ntu-fi-id', 'ut-har', 'widar'
            root_dir: Root directory containing all datasets
            mode: 'train', 'val', or 'test'
        """
        self.dataset_name = dataset_name.lower()
        
        if 'ntu-fi-har' in self.dataset_name:
            self.dataset = NTUFiDataset(root_dir, mode, task='HAR', **kwargs)
            self.input_shape = (3, 114, 500)  # [antennas, subcarriers, time]
            self.num_classes = len(self.dataset.category)
            
        elif 'ntu-fi-id' in self.dataset_name:
            self.dataset = NTUFiDataset(root_dir, mode, task='HumanID', **kwargs)
            self.input_shape = (3, 114, 500)
            self.num_classes = len(self.dataset.category)
            
        elif 'ut-har' in self.dataset_name:
            self.dataset = UTHARDataset(root_dir, mode, **kwargs)
            self.input_shape = (1, 250, 90)  # [channel, time, features]
            self.num_classes = 7  # UT-HAR has 7 activities
            
        elif 'widar' in self.dataset_name:
            self.dataset = WidarDataset(root_dir, mode, **kwargs)
            self.input_shape = (22, 20, 20)  # [BVP, time, velocity]
            self.num_classes = len(self.dataset.category)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"Initialized {dataset_name} dataset:")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Num samples: {len(self.dataset)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def create_benchmark_dataloaders(
    dataset_name: str,
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for benchmark datasets
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to data directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    
    # Create datasets
    train_dataset = UnifiedCSIDataset(
        dataset_name, 
        data_path,
        mode='train',
        **kwargs
    )
    
    # Some datasets don't have validation split, use test as val
    try:
        val_dataset = UnifiedCSIDataset(
            dataset_name,
            data_path, 
            mode='val',
            **kwargs
        )
    except:
        val_dataset = UnifiedCSIDataset(
            dataset_name,
            data_path,
            mode='test',
            **kwargs
        )
    
    test_dataset = UnifiedCSIDataset(
        dataset_name,
        data_path,
        mode='test',
        **kwargs
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
        'test': test_loader,
        'info': {
            'input_shape': train_dataset.input_shape,
            'num_classes': train_dataset.num_classes
        }
    }


if __name__ == "__main__":
    # Test loading different datasets
    data_path = "./Data"  # Adjust to your data path
    
    datasets = ['ntu-fi-har', 'ut-har', 'widar']
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Testing {dataset_name}")
        print('='*50)
        
        try:
            dataloaders = create_benchmark_dataloaders(
                dataset_name,
                data_path,
                batch_size=4,
                num_workers=0
            )
            
            # Test one batch
            for batch_idx, (data, labels) in enumerate(dataloaders['train']):
                print(f"Data shape: {data.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Label values: {labels}")
                break
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")