"""
ML Trainer Module.

Handles benchmarking of candidate models, hyperparameter optimization via Optuna,
and final model training and persistence.
"""

import time
import os
import joblib
import logging
import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.ml.evaluator import MLEvaluator

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class MLTrainer:
    def __init__(self, model_dir: str = "models/trained/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.evaluator = MLEvaluator()
        self.best_model_name = None
        self.best_model = None

    def get_candidate_models(self) -> Dict[str, Any]:
        """Returns instantiated baseline models with fixed random seeds."""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=11),
            'LightGBM': LGBMClassifier(random_state=42, is_unbalance=True, n_jobs=-1, verbose=-1),
        }
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced', thread_count=-1)
        else:
            logger.warning("CatBoost not installed. Skipping CatBoost in benchmark.")
        return models

    def benchmark_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        """Trains all candidates on the baseline sample and evaluates them."""
        logger.info("Starting baseline benchmark of candidate models...")
        models = self.get_candidate_models()
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Training Time
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            # Inference Time
            start_infer = time.time()
            y_pred = model.predict(X_val)
            
            # Handle predict_proba differences
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = y_pred # Fallback
                
            infer_time = time.time() - start_infer
            
            # Evaluate
            self.evaluator.evaluate_model(name, y_val, y_pred, y_prob, train_time, infer_time)
            
        return self.evaluator.get_comparison_table()

    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 15) -> Any:
        """Runs Optuna optimization for a specific model."""
        logger.info(f"Starting Optuna Hyperparameter Optimization for {model_name}...")
        
        def objective(trial):
            if model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0)
                }
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
            
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0)
                }
                model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
                
            elif model_name == 'Random Forest':
                 params = {
                     'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                     'max_depth': trial.suggest_int('max_depth', 5, 20),
                     'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                 }
                 model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **params)
            else:
                # Fallback to default if not explicitly mapped
                model = self.get_candidate_models()[model_name]
                
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.predict(X_val)
                
            from sklearn.metrics import average_precision_score
            return average_precision_score(y_val, y_prob) # Optimize for PR-AUC

        study = optuna.create_study(direction="maximize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best PR-AUC for {model_name}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Re-instantiate with best params
        if model_name == 'XGBoost':
            best = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **study.best_params)
        elif model_name == 'LightGBM':
            best = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **study.best_params)
        elif model_name == 'Random Forest':
            best = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **study.best_params)
        else:
            best = self.get_candidate_models()[model_name]
            
        return best

    def train_final_model(self, model, model_name: str, X_full: pd.DataFrame, y_full: pd.Series):
        """Trains the final model on the complete dataset and saves it."""
        logger.info(f"Training final {model_name} model on full production dataset (Shape: {X_full.shape})...")
        model.fit(X_full, y_full)
        
        self.best_model = model
        self.best_model_name = model_name
        
        # Save timestamped and versioned
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_')}_v1_{timestamp}.joblib"
        joblib.dump(model, os.path.join(self.model_dir, filename))
        
        # Save standard best_model.joblib for Streamlit
        joblib.dump(model, os.path.join(self.model_dir, "best_model.joblib"))
        
        # Save feature list
        joblib.dump(X_full.columns.tolist(), os.path.join(self.model_dir, "model_features.joblib"))
        
        logger.info(f"Final model deployed to {self.model_dir}best_model.joblib")
