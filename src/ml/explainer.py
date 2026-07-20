"""
Explainable AI (XAI) Module.

Provides SHAP-based global and local feature importance and visual analytics
to interpret machine learning predictions.
"""

import os
import shap
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class AIExplainer:
    def __init__(self, model: Any, feature_names: List[str]):
        """Initializes the SHAP explainer based on the provided model."""
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Determines the correct SHAP explainer type."""
        logger.info("Initializing SHAP Explainer...")
        try:
            # TreeExplainer is much faster for ensemble models like XGBoost/LightGBM/RF
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Successfully loaded TreeExplainer.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to general KernelExplainer.")
            # We defer KernelExplainer initialization until we have background data
            self.explainer = None

    def _ensure_explainer(self, background_data: pd.DataFrame = None):
        if self.explainer is None:
            if background_data is None:
                raise ValueError("KernelExplainer requires background data. Pass background_data to generate_global_importance.")
            # Use kmeans to summarize background data for performance
            summary = shap.kmeans(background_data, 10)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, summary)
            logger.info("Successfully loaded KernelExplainer.")

    def get_local_explanation(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates SHAP values for a single applicant and identifies
        the top positive (risk drivers) and negative (protective factors).
        """
        self._ensure_explainer(input_df) # Pass input as background just in case, though Kernel requires larger sets usually
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(input_df)
        
        # Handle binary classification lists (some versions/models return a list [class0, class1])
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # We want SHAP values for class 1 (Default)
            
        # Also need base value
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[1]
        else:
            base_value = self.explainer.expected_value
            
        shap_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Value': input_df.iloc[0].values,
            'SHAP_Value': shap_values[0]
        })
        
        # Sort by impact
        top_risk_drivers = shap_df[shap_df['SHAP_Value'] > 0].sort_values(by='SHAP_Value', ascending=False).head(5)
        top_protective = shap_df[shap_df['SHAP_Value'] < 0].sort_values(by='SHAP_Value', ascending=True).head(5)
        
        return {
            'base_value': float(base_value),
            'shap_values': shap_values[0],
            'top_risk_drivers': top_risk_drivers.to_dict(orient='records'),
            'top_protective': top_protective.to_dict(orient='records')
        }

    def generate_waterfall_plot(self, input_df: pd.DataFrame):
        """Returns a Matplotlib figure for a SHAP Waterfall plot."""
        import matplotlib.pyplot as plt
        
        self._ensure_explainer(input_df)
        
        # For Waterfall we need an Explanation object
        shap_values = self.explainer(input_df)
        
        # Select class 1 if multi-output
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]
            
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        
        return fig
