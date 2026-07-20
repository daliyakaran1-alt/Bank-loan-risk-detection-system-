"""
Enterprise Risk Score Engine.

Converts ML probabilities into standardized Risk Scores (300-900)
and implements a multi-variate Decision Engine for loan underwriting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class EnterpriseRiskEngine:
    def __init__(self):
        # Configure thresholds
        self.score_min = 300
        self.score_max = 900
        self.risk_categories = [
            (850, 900, "Excellent"),
            (750, 849, "Very Low Risk"),
            (650, 749, "Low Risk"),
            (550, 649, "Medium Risk"),
            (450, 549, "High Risk"),
            (300, 449, "Very High Risk")
        ]

    def calculate_risk_score(self, default_probability: float) -> int:
        """
        Calibrates default probability to a 300-900 scale.
        Linear mapping where Prob=0 is 900, Prob=1 is 300.
        """
        prob = max(0.0, min(1.0, float(default_probability)))
        score = int(self.score_max - (prob * (self.score_max - self.score_min)))
        return max(self.score_min, min(self.score_max, score))

    def get_risk_category(self, score: int) -> str:
        """Returns the category label for a given score."""
        for min_val, max_val, label in self.risk_categories:
            if min_val <= score <= max_val:
                return label
        return "Unknown"

    def evaluate_decision(self, score: int, prob: float, features: Dict[str, Any]) -> str:
        """
        Multi-variate Decision Engine.
        Uses score, probability, and specific engineered features to make a business recommendation.
        """
        # Feature extraction with safe defaults
        lti_ratio = features.get("Loan_to_Income_Ratio", 0)
        burden = features.get("Credit_Burden_Score", 0)
        stability = features.get("Financial_Stability_Index", 0)
        cluster = features.get("Cluster", "Unknown")

        if score >= 750:
            if lti_ratio > 4.0 or burden > 40:
                return "Approve with Conditions"
            return "Approve"
            
        elif score >= 650:
            if stability < 1 or burden > 40:
                return "Manual Review"
            return "Approve"
            
        elif score >= 550:
            if lti_ratio > 3.5 and burden > 30:
                return "Reduce Loan Amount"
            return "Manual Review"
            
        elif score >= 450:
            if stability >= 2:
                return "Require Additional Documents"
            return "Reject"
            
        else:
            return "Reject"

    def generate_explanation(self, score: int, prob: float, decision: str, features: Dict[str, Any]) -> str:
        """Generates dynamic, human-readable explanations suitable for bank officers."""
        income = features.get("AMT_INCOME_TOTAL", 0)
        loan = features.get("AMT_CREDIT", 0)
        burden = features.get("Credit_Burden_Score", 0)
        stability = features.get("Financial_Stability_Index", 0)
        
        paragraphs = []
        
        # Paragraph 1: Core assessment
        cat = self.get_risk_category(score)
        paragraphs.append(
            f"The applicant generated a FinSight Risk Score of {score} ({cat}), representing an estimated default probability of {prob:.2%}."
        )
        
        # Paragraph 2: Repayment capacity
        if stability >= 2 and burden < 30:
            paragraphs.append(
                "The applicant demonstrates strong repayment capability due to a high Financial Stability Index and low credit burden."
            )
        elif burden > 40:
            paragraphs.append(
                f"However, the credit burden is severely high ({burden:.1f}% of income), indicating stretched household finances."
            )
        elif income < 50000 and loan > 200000:
            paragraphs.append(
                "The requested credit amount is significantly misaligned with the reported annual income, increasing default exposure."
            )
        else:
            paragraphs.append(
                "The applicant presents a standard financial profile with moderate risk indicators."
            )
            
        # Paragraph 3: Decision driver
        paragraphs.append(f"Overall system recommendation: **{decision}**.")
        
        return "\n\n".join(paragraphs)

    def generate_business_recommendations(self, decision: str, score: int) -> Dict[str, str]:
        """Provides specific strategic recommendations based on the decision."""
        recs = {
            "Approve": {
                "Strategy": "Standard onboarding and immediate fulfillment.",
                "Interest Rate": "Prime (Lowest Tier)",
                "Monitoring": "Annual review."
            },
            "Approve with Conditions": {
                "Strategy": "Require collateral verification or co-signer before fulfillment.",
                "Interest Rate": "Standard (Middle Tier)",
                "Monitoring": "Bi-annual review."
            },
            "Manual Review": {
                "Strategy": "Escalate to Senior Underwriter for holistic file assessment.",
                "Interest Rate": "TBD by Underwriter",
                "Monitoring": "Quarterly review if approved."
            },
            "Reduce Loan Amount": {
                "Strategy": "Counter-offer with a 20-30% reduction in principal to align with LTI limits.",
                "Interest Rate": "Standard (Middle Tier)",
                "Monitoring": "Bi-annual review."
            },
            "Require Additional Documents": {
                "Strategy": "Halt processing until tax returns and asset proofs are verified.",
                "Interest Rate": "Sub-prime (High Tier)",
                "Monitoring": "Monthly review if approved."
            },
            "Reject": {
                "Strategy": "Issue adverse action notice compliant with fair lending regulations.",
                "Interest Rate": "N/A",
                "Monitoring": "N/A"
            }
        }
        
        return recs.get(decision, {
            "Strategy": "Review file manually.",
            "Interest Rate": "TBD",
            "Monitoring": "TBD"
        })
