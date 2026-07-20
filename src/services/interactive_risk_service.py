import joblib
import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.db.models import Applicant
from datetime import datetime


class InteractiveRiskService:
    """Orchestrates risk assessment for new manual applications using the retrained model."""
    _instance = None
    _model = None
    _features = None
    _encoders = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InteractiveRiskService, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        model_path = os.path.join("models", "trained", "interactive_model.joblib")
        features_path = os.path.join("models", "trained", "interactive_features.joblib")
        encoders_path = os.path.join("models", "trained", "interactive_encoders.joblib")

        try:
            if os.path.exists(model_path):
                self._model = joblib.load(model_path)
            if os.path.exists(features_path):
                self._features = joblib.load(features_path)
            if os.path.exists(encoders_path):
                self._encoders = joblib.load(encoders_path)
        except Exception as e:
            print(f"Error loading interactive model: {e}")

    def assess_new_application(self, data: dict) -> dict:
        """Predict risk for a new applicant using the retrained extended model."""
        if self._model is None or self._features is None:
            return {"error": "Model unavailable. Please retrain."}

        try:
            # Build feature dict in exact order
            row = {}
            row['AMT_INCOME_TOTAL'] = float(data.get('income', 0))
            row['AGE'] = int(data.get('age', 30))
            row['AMT_CREDIT'] = float(data.get('loan_amount', 0))
            row['CREDIT_SCORE'] = float(data.get('credit_score', 650))
            row['EMPLOYMENT_YEARS'] = float(data.get('employment_years', 5))
            row['CNT_CHILDREN'] = int(data.get('children', 0))

            # Categorical features — encode them using saved LabelEncoders
            cat_map = {
                'NAME_EDUCATION_TYPE': data.get('education', 'Secondary / secondary special'),
                'CODE_GENDER': data.get('gender', 'F'),
                'FLAG_OWN_CAR': data.get('owns_car', 'N'),
                'FLAG_OWN_REALTY': data.get('owns_realty', 'Y'),
                'NAME_FAMILY_STATUS': data.get('family_status', 'Married'),
                'NAME_HOUSING_TYPE': data.get('housing_type', 'House / apartment'),
            }

            if self._encoders:
                for col, val in cat_map.items():
                    le = self._encoders.get(col)
                    if le is not None:
                        # Handle unseen labels gracefully
                        if val in le.classes_:
                            row[col] = le.transform([val])[0]
                        else:
                            row[col] = 0  # fallback
                    else:
                        row[col] = 0
            else:
                for col in cat_map:
                    row[col] = 0

            # Build dataframe in exact feature order, then extract values for sklearn
            df = pd.DataFrame([row])[self._features]

            # Predict (use .values to avoid feature name warnings)
            probability = self._model.predict_proba(df.values)[0][1]
            risk_score = round(probability * 100, 1)

            if risk_score > 60:
                recommendation = "Reject Loan"
                risk_level = "HIGH"
            elif risk_score > 30:
                recommendation = "Manual Review"
                risk_level = "MEDIUM"
            else:
                recommendation = "Approve Loan"
                risk_level = "LOW"

            # XAI — Explainable AI reasoning
            reasons = []
            income = float(data.get('income', 0))
            credit = float(data.get('credit_score', 650))
            loan = float(data.get('loan_amount', 0))
            age = int(data.get('age', 30))
            emp = float(data.get('employment_years', 5))

            lti = loan / max(income, 1)
            if credit < 500:
                reasons.append("Credit Score is critically low (< 500)")
            elif credit < 600:
                reasons.append("Credit Score is below average (< 600)")
            elif credit > 750:
                reasons.append("Excellent Credit Score (> 750)")

            if lti > 6:
                reasons.append(f"Loan-to-Income ratio is very high ({lti:.1f}x)")
            elif lti > 3:
                reasons.append(f"Loan-to-Income ratio is elevated ({lti:.1f}x)")
            elif lti <= 2:
                reasons.append(f"Healthy Loan-to-Income ratio ({lti:.1f}x)")

            if age < 25:
                reasons.append("Young age group carries higher statistical risk")
            elif age > 55:
                reasons.append("Mature age group — stable repayment history typical")

            if emp < 1:
                reasons.append("Very short employment history (< 1 year)")
            elif emp > 10:
                reasons.append("Strong employment stability (10+ years)")

            if data.get('owns_realty', 'N') == 'Y':
                reasons.append("Owns real estate — positive collateral signal")
            
            if int(data.get('children', 0)) > 3:
                reasons.append("Large number of dependents increases financial burden")

            if not reasons:
                reasons.append("Overall financial profile is within normal parameters")

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "reasons": reasons
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Prediction failed: {str(e)}"}

    def save_application(self, db: Session, data: dict, result: dict, user_id: int):
        """Save a completed application to the database."""
        try:
            app = Applicant(
                user_id=user_id,
                name=data.get('name', 'Unknown'),
                age=int(data.get('age', 0)),
                gender=data.get('gender', 'F'),
                family_status=data.get('family_status', 'Single / not married'),
                children=int(data.get('children', 0)),
                income=float(data.get('income', 0)),
                loan_amount=float(data.get('loan_amount', 0)),
                credit_score=float(data.get('credit_score', 0)),
                owns_car=data.get('owns_car', 'N'),
                owns_realty=data.get('owns_realty', 'N'),
                housing_type=data.get('housing_type', 'House / apartment'),
                employment_years=float(data.get('employment_years', 0)),
                education=data.get('education', 'Secondary / secondary special'),
                risk_score=result.get('risk_score', 0),
                risk_level=result.get('risk_level', 'UNKNOWN'),
                recommendation=result.get('recommendation', 'N/A'),
                reasons=result.get('reasons', []),
                created_at=datetime.utcnow()
            )
            db.add(app)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Error saving application: {e}")
            return False
