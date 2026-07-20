"""
ML Preprocessor Module.

Prepares the segmented dataset for machine learning.
Handles train/val/test splits, removing leakage columns, and handling class imbalance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class MLPreprocessor:
    def __init__(self, target_col: str = 'TARGET'):
        self.target_col = target_col
        self.leakage_cols = ['SK_ID_CURR', 'CLIENT_ID', 'Cluster'] # Cluster shouldn't leak, but we use PCA components
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates features and target, drops identifier and leakage columns."""
        logger.info("Preparing data for ML...")
        df_clean = df.copy()
        
        # Drop identifiers
        drop_cols = [c for c in self.leakage_cols if c in df_clean.columns]
        if drop_cols:
            df_clean = df_clean.drop(columns=drop_cols)
            
        # Separate X and y
        y = df_clean[self.target_col]
        X = df_clean.drop(columns=[self.target_col])
        
        # Ensure only numeric columns (should already be encoded, but just in case)
        X = X.select_dtypes(include=[np.number])
        
        # Sanitize column names for XGBoost compatibility (no [, ], or <)
        import re
        clean_names = {col: re.sub(r'[\[\]<>]', '_', col) for col in X.columns}
        X = X.rename(columns=clean_names)
        
        self.feature_names = X.columns.tolist()
        
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.1, val_size: float = 0.1) -> Dict[str, Any]:
        """Performs a stratified 80/10/10 split."""
        logger.info(f"Performing Stratified Split (Train: {1-test_size-val_size:.2f}, Val: {val_size}, Test: {test_size})...")
        
        # First split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Now split the temp set into train and val
        # To get 10% of total as val, we need val_size / (1 - test_size)
        val_ratio = val_size / (1.0 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        logger.info(f"Split results: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """Applies oversampling techniques if requested."""
        logger.info(f"Applying class imbalance handling technique: {method}")
        
        if method == 'none' or method == 'scale_pos_weight':
            return X_train, y_train
            
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(random_state=42)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        else:
            raise ValueError(f"Unknown imbalance method: {method}")
            
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Ensure returned as DataFrame to preserve column names
        if not isinstance(X_resampled, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=self.feature_names)
            
        logger.info(f"Original shape: {X_train.shape}, Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
