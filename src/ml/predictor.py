"""
ML Predictor Module.

Provides reusable prediction functions for Streamlit integration and API exposure.
"""

import os
import joblib
import pandas as pd
import numpy as np
import logging

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class MLPredictor:
    def __init__(self, model_dir: str = "models/trained/"):
        self.model_path = os.path.join(model_dir, "best_model.joblib")
        self.features_path = os.path.join(model_dir, "model_features.joblib")
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Loads the serialized model and feature list."""
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.features_path)
            logger.info("Successfully loaded ML model for inference.")
        else:
            logger.warning(f"Model or features not found in {self.model_path}. Predictor is disabled.")

    def _prepare_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensures the input dataframe matches the expected feature schema perfectly."""
        if self.feature_names is None:
            raise ValueError("Model features not loaded.")
            
        df = data.copy()
        
        # Add missing columns (e.g. from OHE categories) with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        # Select and order exactly as training
        return df[self.feature_names]

    def predict_default(self, data: pd.DataFrame) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        X = self._prepare_input(data)
        return self.model.predict(X)

    def predict_probability(self, data: pd.DataFrame) -> np.ndarray:
        """Returns the probability of default (class 1)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        X = self._prepare_input(data)
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X) # Fallback

    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame with appended predictions and probabilities."""
        df_out = data.copy()
        df_out['Predicted_Default'] = self.predict_default(data)
        df_out['Default_Probability'] = self.predict_probability(data)
        return df_out

    def prediction_summary(self, data: pd.DataFrame) -> dict:
        """Generates a summary of the prediction for dashboard display."""
        prob = self.predict_probability(data)[0]
        pred = 1 if prob >= 0.5 else 0
        
        # Determine confidence
        if prob > 0.8: confidence = "High Risk"
        elif prob > 0.5: confidence = "Moderate Risk"
        elif prob > 0.2: confidence = "Moderate Safety"
        else: confidence = "High Safety"
        
        return {
            "prediction": "Default" if pred == 1 else "Repay",
            "probability": float(prob),
            "confidence_band": confidence
        }
