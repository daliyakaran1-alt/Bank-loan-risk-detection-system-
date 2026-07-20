"""
Model Registry and Governance Module.

Logs every prediction made by the system to ensure auditability, compliance,
and longitudinal monitoring.
"""

import os
import pandas as pd
from datetime import datetime
import uuid
import logging
from typing import Dict, Any

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class ModelRegistry:
    def __init__(self, log_dir: str = "models/logs/"):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "prediction_audit_log.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        """Creates the CSV log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            columns = [
                "Prediction_ID",
                "Timestamp",
                "Model_Version",
                "Predicted_Probability",
                "Risk_Score",
                "Decision",
                "Input_Features_JSON"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_file, index=False)
            logger.info(f"Initialized Model Governance Log at {self.log_file}")

    def log_prediction(self, model_version: str, prob: float, score: int, decision: str, features: Dict[str, Any]) -> str:
        """
        Logs a single prediction and returns a unique tracking ID.
        """
        pred_id = f"PRD-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()
        
        import json
        
        # Filter features to essential items to save space, or log all
        # Converting numpy types to native python for JSON serialization
        clean_features = {}
        for k, v in features.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_features[k] = v.item()
            else:
                clean_features[k] = v
                
        record = {
            "Prediction_ID": pred_id,
            "Timestamp": timestamp,
            "Model_Version": model_version,
            "Predicted_Probability": round(prob, 4),
            "Risk_Score": score,
            "Decision": decision,
            "Input_Features_JSON": json.dumps(clean_features)
        }
        
        df = pd.DataFrame([record])
        df.to_csv(self.log_file, mode='a', header=False, index=False)
        
        return pred_id

    def get_audit_trail(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves the most recent predictions from the governance log."""
        if os.path.exists(self.log_file):
            df = pd.read_csv(self.log_file)
            return df.tail(limit).sort_values(by="Timestamp", ascending=False)
        return pd.DataFrame()
