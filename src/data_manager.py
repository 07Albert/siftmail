import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from .libraries import *

class DataManager:
    """Efficient data manager with caching and preprocessing pipeline"""
    
    def __init__(self, data_path: str = "spambase/spambase.csv", 
                 features_path: str = "spambase/features.txt",
                 cache_dir: str = "cache"):
        self.data_path = data_path
        self.features_path = features_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._data_cache = {}
        self._preprocessing_cache = {}
        
    def load_data(self, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """Load data with caching support"""
        cache_key = f"raw_data_{Path(self.data_path).stem}"
        
        if use_cache and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Load features
        with open(self.features_path, "r") as f:
            lines = f.readlines()
        raw = [x.strip().split(":") for x in lines]
        features = [x[0] for x in raw] + ["spam"]
        dataType = [x[1].strip()[:-1] for x in raw]
        
        # Load dataset
        df = pd.read_csv(self.data_path, names=features, index_col=False)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        
        result = (X, y, dataType, features)
        
        if use_cache:
            self._data_cache[cache_key] = result
            
        return result
    
    def get_preprocessing_pipeline(self, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get preprocessed data with caching"""
        cache_key = self._get_cache_key(config)
        
        if cache_key in self._preprocessing_cache:
            return self._preprocessing_cache[cache_key]
        
        X, y, _, _ = self.load_data()
        X_processed, y_processed = self._apply_preprocessing(X, y, config)
        
        self._preprocessing_cache[cache_key] = (X_processed, y_processed)
        return X_processed, y_processed
    
    def _get_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key from configuration"""
        return f"preprocessed_{hash(str(sorted(config.items())))}"
    
    def _apply_preprocessing(self, X: np.ndarray, y: np.ndarray, 
                           config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply preprocessing based on configuration"""
        # Data transformation
        if config.get('transformation') == 'minmax':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif config.get('transformation') == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif config.get('transformation') == 'normalize':
            X = normalize(X, norm='l1')
        
        # Feature selection
        if config.get('feature_selection') == 'variance':
            selector = VarianceThreshold(threshold=1)
            X = selector.fit_transform(X)
        elif config.get('feature_selection') == 'chi2':
            X = SelectKBest(chi2, k=28).fit_transform(X, y)
        elif config.get('feature_selection') == 'mutual_info':
            X = SelectKBest(mutual_info_classif, k=28).fit_transform(X, y)
        
        # Balancing
        if config.get('balancing') == 'smote':
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
        elif config.get('balancing') == 'random_under':
            rus = RandomUnderSampler(random_state=0)
            X, y = rus.fit_resample(X, y)
        
        return X, y
    
    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        self._preprocessing_cache.clear() 