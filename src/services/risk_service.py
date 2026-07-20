"""
Risk management service.
Handles real-time machine learning predictions using the trained XGBoost/Logistic Regression models.
"""
import joblib
import os
import pandas as pd
import numpy as np

class RiskService:
    """Orchestrates risk assessment workflows."""
    _instance = None
    _model = None
    _features = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RiskService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Load the ML model and feature list."""
        model_path = os.path.join("models", "trained", "best_model.joblib")
        features_path = os.path.join("models", "trained", "model_features.joblib")
        
        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                self._model = joblib.load(model_path)
                self._features = joblib.load(features_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("Model files not found. Risk scoring unavailable.")

    def assess_loan_application(self, application_data: dict) -> dict:
        """
        Assess risk for a new loan application using the trained ML model.
        
        Args:
            application_data (dict): Data for the loan application.
            
        Returns:
            dict: Assessment results including risk score and recommendation.
        """
        if self._model is None or self._features is None:
            return {"error": "Machine learning model is currently unavailable."}
            
        try:
            # Create a dataframe with NaN for all required features
            df = pd.DataFrame(np.nan, index=[0], columns=self._features, dtype=np.float64)
            
            # Map incoming data to the feature columns if they exist
            for key, value in application_data.items():
                if key in self._features:
                    try:
                        df.at[0, key] = float(value) if pd.notnull(value) else np.nan
                    except (ValueError, TypeError):
                        pass
            
            # The pipeline handles imputation (fillna) and scaling automatically
            # Predict probability of default (class 1)
            probability = self._model.predict_proba(df)[0][1]
            risk_score = round(probability * 100, 1)
            
            if risk_score > 60:
                recommendation = "High Risk - Manual Review Required"
            elif risk_score > 30:
                recommendation = "Moderate Risk - Proceed with Caution"
            else:
                recommendation = "Low Risk - Auto Approval Eligible"
                
            return {
                "risk_score": risk_score,
                "probability": probability,
                "recommendation": recommendation
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
