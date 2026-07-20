"""
ML Evaluator Module.

Handles model evaluation metrics (prioritizing imbalanced data metrics),
comparison tables, and saving evaluation reports.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class MLEvaluator:
    def __init__(self):
        self.results = []

    def evaluate_model(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, train_time: float, infer_time: float) -> Dict[str, Any]:
        """Calculates comprehensive metrics for a single model."""
        
        metrics = {
            'Model': model_name,
            'Training Time (s)': round(train_time, 2),
            'Inference Time (s)': round(infer_time, 4),
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'F1 Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else 0.0,
            'PR-AUC': round(average_precision_score(y_true, y_prob), 4) if y_prob is not None else 0.0
        }
        
        self.results.append(metrics)
        logger.info(f"{model_name} Evaluated. PR-AUC: {metrics['PR-AUC']:.4f}, Recall: {metrics['Recall']:.4f}")
        return metrics

    def get_comparison_table(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame of all evaluated models, ranked by PR-AUC."""
        if not self.results:
            return pd.DataFrame()
            
        df_results = pd.DataFrame(self.results)
        
        # Calculate a weighted score for selection: 40% PR-AUC, 30% Recall, 20% ROC-AUC, 10% F1
        df_results['Weighted_Score'] = (
            (df_results['PR-AUC'] * 0.4) +
            (df_results['Recall'] * 0.3) +
            (df_results['ROC-AUC'] * 0.2) +
            (df_results['F1 Score'] * 0.1)
        )
        
        # Rank based on the weighted score
        df_results = df_results.sort_values(by='Weighted_Score', ascending=False).reset_index(drop=True)
        df_results['Ranking'] = df_results.index + 1
        
        # Add business recommendation column based on rank
        def recommend(rank):
            if rank == 1: return "Champion Model"
            if rank == 2: return "Challenger Model"
            return "Discard"
            
        df_results['Business Recommendation'] = df_results['Ranking'].apply(recommend)
        
        return df_results

    def select_top_n_models(self, n: int = 2) -> List[str]:
        """Returns the names of the top N models based on the weighted ranking."""
        df = self.get_comparison_table()
        if df.empty:
            return []
        return df['Model'].head(n).tolist()
