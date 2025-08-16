"""
数据缓存管理器
提供多层级缓存：内存缓存 + 磁盘缓存
特别适用于sweep实验中的数据复用
"""

import hashlib
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import gc
import psutil
import numpy as np
import torch

class DataCacheManager:
    """数据缓存管理器"""
    
    _instance = None
    _memory_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    _cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    _max_memory_mb = 4096  # 最大内存使用量(MB)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_cache_key(cls, n: int, T: int, F: int, difficulty: str, seed: int,
                     sc_corr_rho: Optional[float] = None,
                     env_burst_rate: float = 0.0,
                     gain_drift_std: float = 0.0) -> str:
        """生成缓存键"""
        params_str = f"n={n}_T={T}_F={F}_diff={difficulty}_seed={seed}_rho={sc_corr_rho}_burst={env_burst_rate}_drift={gain_drift_std}"
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    @classmethod
    def get_memory_usage_mb(cls) -> float:
        """获取当前内存使用量(MB)"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @classmethod
    def estimate_data_size_mb(cls, X: np.ndarray, y: np.ndarray) -> float:
        """估算数据占用内存大小(MB)"""
        x_size = X.nbytes if hasattr(X, 'nbytes') else X.size * X.itemsize
        y_size = y.nbytes if hasattr(y, 'nbytes') else y.size * y.itemsize
        return (x_size + y_size) / 1024 / 1024
    
    @classmethod
    def cleanup_memory_cache(cls, target_free_mb: float = 1024):
        """清理内存缓存以释放空间"""
        if not cls._memory_cache:
            return
        
        print(f"[INFO] Cleaning memory cache to free ~{target_free_mb}MB...")
        
        # 按键排序（最近最少使用的简单代理）
        keys_to_remove = list(cls._memory_cache.keys())[:len(cls._memory_cache)//2]
        
        freed_mb = 0
        for key in keys_to_remove:
            if key in cls._memory_cache:
                X, y = cls._memory_cache[key]
                freed_mb += cls.estimate_data_size_mb(X, y)
                del cls._memory_cache[key]
                cls._cache_stats["evictions"] += 1
                
                if freed_mb >= target_free_mb:
                    break
        
        gc.collect()  # 强制垃圾回收
        print(f"[INFO] Freed {freed_mb:.1f}MB from memory cache")
    
    @classmethod
    def get_from_memory(cls, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从内存缓存获取数据"""
        if cache_key in cls._memory_cache:
            cls._cache_stats["hits"] += 1
            return cls._memory_cache[cache_key]
        
        cls._cache_stats["misses"] += 1
        return None
    
    @classmethod
    def put_to_memory(cls, cache_key: str, X: np.ndarray, y: np.ndarray):
        """将数据放入内存缓存"""
        data_size_mb = cls.estimate_data_size_mb(X, y)
        current_memory_mb = cls.get_memory_usage_mb()
        
        # 检查是否需要清理缓存
        if current_memory_mb + data_size_mb > cls._max_memory_mb:
            cls.cleanup_memory_cache(target_free_mb=data_size_mb * 2)
        
        # 存储数据（复制以避免外部修改）
        cls._memory_cache[cache_key] = (X.copy(), y.copy())
        print(f"[INFO] Cached to memory: {cache_key} ({data_size_mb:.1f}MB)")
    
    @classmethod
    def load_from_disk(cls, cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从磁盘加载缓存数据"""
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['X'], cached_data['y']
        except Exception as e:
            print(f"[WARNING] Failed to load disk cache {cache_path}: {e}")
            return None
    
    @classmethod 
    def save_to_disk(cls, cache_path: Path, X: np.ndarray, y: np.ndarray):
        """保存数据到磁盘缓存"""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {'X': X, 'y': y}
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[INFO] Cached to disk: {cache_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save disk cache: {e}")
    
    @classmethod
    def get_or_generate(cls, generator_func, cache_dir: str = "cache/synth_data", 
                       use_memory_cache: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取或生成数据（支持多级缓存）
        
        Args:
            generator_func: 数据生成函数
            cache_dir: 磁盘缓存目录
            use_memory_cache: 是否使用内存缓存
            **kwargs: 传递给生成函数和缓存键生成的参数
        
        Returns:
            (X, y): 数据集
        """
        # 生成缓存键
        cache_key = cls.get_cache_key(**{k: v for k, v in kwargs.items() 
                                       if k in ['n', 'T', 'F', 'difficulty', 'seed', 
                                               'sc_corr_rho', 'env_burst_rate', 'gain_drift_std']})
        
        # 1. 尝试内存缓存
        if use_memory_cache:
            data = cls.get_from_memory(cache_key)
            if data is not None:
                print(f"[INFO] Loaded from memory cache: {cache_key}")
                return data
        
        # 2. 尝试磁盘缓存
        cache_path = Path(cache_dir) / f"synth_data_{cache_key}.pkl"
        data = cls.load_from_disk(cache_path)
        if data is not None:
            print(f"[INFO] Loaded from disk cache: {cache_path}")
            X, y = data
            # 同时放入内存缓存
            if use_memory_cache:
                cls.put_to_memory(cache_key, X, y)
            return X, y
        
        # 3. 生成新数据
        print(f"[INFO] Generating new dataset: {kwargs}")
        X, y = generator_func(**kwargs)
        
        # 4. 保存到缓存
        cls.save_to_disk(cache_path, X, y)
        if use_memory_cache:
            cls.put_to_memory(cache_key, X, y)
        
        return X, y
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取缓存统计信息"""
        memory_usage_mb = cls.get_memory_usage_mb()
        cache_size = len(cls._memory_cache)
        
        # 计算缓存中的数据总大小
        cached_data_mb = 0
        for X, y in cls._memory_cache.values():
            cached_data_mb += cls.estimate_data_size_mb(X, y)
        
        return {
            "memory_hits": cls._cache_stats["hits"],
            "memory_misses": cls._cache_stats["misses"],
            "memory_evictions": cls._cache_stats["evictions"],
            "memory_hit_rate": cls._cache_stats["hits"] / max(1, cls._cache_stats["hits"] + cls._cache_stats["misses"]),
            "memory_cache_count": cache_size,
            "memory_usage_mb": memory_usage_mb,
            "cached_data_mb": cached_data_mb,
            "max_memory_mb": cls._max_memory_mb
        }
    
    @classmethod
    def print_stats(cls):
        """打印缓存统计信息"""
        stats = cls.get_stats()
        print("\n[CACHE STATS]")
        print(f"  Memory hits: {stats['memory_hits']}")
        print(f"  Memory misses: {stats['memory_misses']}")  
        print(f"  Hit rate: {stats['memory_hit_rate']:.2%}")
        print(f"  Cache entries: {stats['memory_cache_count']}")
        print(f"  Cached data: {stats['cached_data_mb']:.1f}MB")
        print(f"  Process memory: {stats['memory_usage_mb']:.1f}MB")
        print(f"  Memory limit: {stats['max_memory_mb']:.1f}MB")
    
    @classmethod
    def clear_memory_cache(cls):
        """清空内存缓存"""
        cls._memory_cache.clear()
        cls._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        gc.collect()
        print("[INFO] Memory cache cleared")
    
    @classmethod
    def set_memory_limit_mb(cls, limit_mb: int):
        """设置内存使用限制"""
        cls._max_memory_mb = limit_mb
        print(f"[INFO] Memory limit set to {limit_mb}MB")