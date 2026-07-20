"""
Data Transformer module.

Provides the FeatureEngineer and DataTransformer classes to apply transformations like 
feature generation, encoding, scaling, and feature selection.
"""
import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

try:
    from core.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

from core.exceptions import DataTransformationError

logger = get_logger(__name__)

class FeatureEngineer:
    """Generates domain-specific business features for credit risk."""

    def __init__(self):
        self.feature_metadata = []

    def _add_metadata(self, name: str, formula: str, meaning: str, rationale: str, impact: str):
        self.feature_metadata.append({
            "Feature Name": name,
            "Formula": formula,
            "Business Meaning": meaning,
            "Why Bank Uses It": rationale,
            "Expected Impact": impact
        })

    def generate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Financial Ratios...")
        
        # Helper variables
        inc = df.get('AMT_INCOME_TOTAL', pd.Series(dtype=float))
        loan = df.get('AMT_CREDIT', pd.Series(dtype=float))
        annuity = df.get('AMT_ANNUITY', pd.Series(dtype=float))
        fam = df.get('CNT_FAM_MEMBERS', pd.Series(dtype=float))
        child = df.get('CNT_CHILDREN', pd.Series(dtype=float))

        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
            df['Loan_to_Income_Ratio'] = loan / inc.replace(0, np.nan)
            self._add_metadata("Loan_to_Income_Ratio", "AMT_CREDIT / AMT_INCOME_TOTAL", 
                               "The size of the loan relative to income.", 
                               "Assesses affordability and over-leveraging.", 
                               "High ratio strongly correlates with default risk.")

            df['Credit_to_Income_Ratio'] = loan / inc.replace(0, np.nan) # Redundant with Loan to Income, kept for compliance
            self._add_metadata("Credit_to_Income_Ratio", "AMT_CREDIT / AMT_INCOME_TOTAL", 
                               "Credit extended vs annual income.", 
                               "Standard regulatory check.", 
                               "Identifies over-indebtedness.")

        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_ANNUITY' in df.columns:
            df['Annuity_to_Income_Ratio'] = annuity / inc.replace(0, np.nan)
            self._add_metadata("Annuity_to_Income_Ratio", "AMT_ANNUITY / AMT_INCOME_TOTAL", 
                               "Proportion of income consumed by loan payment.", 
                               "Direct measure of repayment capacity.", 
                               "Crucial for predicting short-term default.")

        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['Income_per_Family_Member'] = inc / fam.replace(0, np.nan)
            self._add_metadata("Income_per_Family_Member", "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", 
                               "Disposable income per person.", 
                               "Better reflects true living standards.", 
                               "Refines income predictive power.")

            df['Credit_per_Family_Member'] = loan / fam.replace(0, np.nan)
            self._add_metadata("Credit_per_Family_Member", "AMT_CREDIT / CNT_FAM_MEMBERS", 
                               "Debt burden distributed across family.", 
                               "Measures household debt pressure.", 
                               "High values indicate stretched household finances.")

        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_CHILDREN' in df.columns:
            # Add 1 to avoid division by zero for childless
            df['Income_per_Child'] = inc / (child + 1)
            self._add_metadata("Income_per_Child", "AMT_INCOME_TOTAL / (CNT_CHILDREN + 1)", 
                               "Income scaled by dependents.", 
                               "Childcare is a major hidden expense.", 
                               "Improves risk profiling for large families.")

        return df

    def generate_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Family Features...")

        if 'CNT_FAM_MEMBERS' in df.columns:
            df['Family_Size_Category'] = pd.cut(df['CNT_FAM_MEMBERS'], bins=[0, 1, 3, 5, 100], labels=['Single', 'Small', 'Medium', 'Large']).astype(str)
            self._add_metadata("Family_Size_Category", "Categorized CNT_FAM_MEMBERS", 
                               "Groups family sizes.", "Simplifies demographic risk profiling.", "Captures non-linear risk of family size.")

        if 'CNT_CHILDREN' in df.columns:
            df['Children_Category'] = pd.cut(df['CNT_CHILDREN'], bins=[-1, 0, 2, 100], labels=['None', '1-2', '3+']).astype(str)
            self._add_metadata("Children_Category", "Categorized CNT_CHILDREN", 
                               "Groups child counts.", "Identifies high-dependency households.", "Isolates the high cost of raising multiple children.")

        if 'CNT_FAM_MEMBERS' in df.columns and 'CNT_CHILDREN' in df.columns:
            df['Dependency_Ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
            self._add_metadata("Dependency_Ratio", "CNT_CHILDREN / CNT_FAM_MEMBERS", 
                               "Proportion of dependents.", "Economic burden indicator.", "High ratio limits debt service capacity.")
            
            # Approximation of Single Income (Single adult + children)
            df['Single_Income_Household'] = ((df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']) == 1).astype(int)
            self._add_metadata("Single_Income_Household", "1 if (Adults == 1), 0 otherwise", 
                               "Single parent / earner household.", "Highly vulnerable to income shocks.", "Strong predictor of financial distress.")

        return df

    def generate_employment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Employment Features...")
        
        if 'DAYS_EMPLOYED' in df.columns:
            # Handle the 365243 anomaly if present (often means pensioner/unemployed in some datasets)
            days_emp = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            years_emp = abs(days_emp) / 365.25
            
            df['Employment_Stability'] = (years_emp > 5).astype(int)
            self._add_metadata("Employment_Stability", "1 if Years_Employed > 5 else 0", 
                               "Long-term employment flag.", "Banks favor stable job history.", "Reduces default probability.")
            
            df['Employment_Duration_Category'] = pd.cut(years_emp, bins=[0, 2, 5, 10, 100], labels=['<2y', '2-5y', '5-10y', '>10y']).astype(str)
            self._add_metadata("Employment_Duration_Category", "Binned Years_Employed", 
                               "Categorical tenure.", "Standard risk scorecard input.", "Captures non-linear risk of tenure.")
                               
        if 'DAYS_BIRTH' in df.columns:
            years_age = abs(df['DAYS_BIRTH']) / 365.25
            df['Age_Group'] = pd.cut(years_age, bins=[0, 25, 35, 50, 65, 100], labels=['<25', '25-35', '35-50', '50-65', '>65']).astype(str)
            self._add_metadata("Age_Group", "Binned Age", "Age brackets.", "Risk varies heavily by life stage.", "Critical demographic separator.")
            
            df['Working_Age_Category'] = ((years_age > 25) & (years_age < 60)).astype(int)
            self._add_metadata("Working_Age_Category", "1 if 25 < Age < 60", "Prime earning years.", "Maximum repayment capacity phase.", "Lower risk segment.")

        return df

    def generate_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Asset Features...")
        
        car_col = 'FLAG_OWN_CAR' if 'FLAG_OWN_CAR' in df.columns else None
        house_col = 'FLAG_OWN_REALTY' if 'FLAG_OWN_REALTY' in df.columns else None
        
        if car_col and house_col:
            owns_car = (df[car_col] == 'Y') | (df[car_col] == 1)
            owns_house = (df[house_col] == 'Y') | (df[house_col] == 1)
            
            df['Owns_Both_House_And_Car'] = (owns_car & owns_house).astype(int)
            self._add_metadata("Owns_Both_House_And_Car", "Owns Car AND House", "High asset wealth.", "Collateral and stability.", "Lowest risk group.")
            
            df['Owns_House_Only'] = (~owns_car & owns_house).astype(int)
            self._add_metadata("Owns_House_Only", "Owns House AND NOT Car", "Property owner.", "Fixed asset stability.", "Moderate to low risk.")
            
            df['Owns_Car_Only'] = (owns_car & ~owns_house).astype(int)
            self._add_metadata("Owns_Car_Only", "Owns Car AND NOT House", "Mobile asset owner.", "Some liquidity, but less stability.", "Moderate risk.")
            
            df['Asset_Ownership_Score'] = owns_car.astype(int) + owns_house.astype(int)
            self._add_metadata("Asset_Ownership_Score", "Sum of owned major assets (0-2)", "Aggregate wealth index.", "Quick proxy for affluence.", "Negative correlation with default.")

        return df

    def generate_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Credit Features...")
        
        if 'AMT_CREDIT' in df.columns:
            q75_cred = df['AMT_CREDIT'].quantile(0.75)
            df['High_Credit_Applicant'] = (df['AMT_CREDIT'] > q75_cred).astype(int)
            self._add_metadata("High_Credit_Applicant", "Credit > 75th percentile", "Seeking large loans.", "Higher exposure amount.", "Increases Loss Given Default (LGD).")
            
        if 'AMT_INCOME_TOTAL' in df.columns:
            q75_inc = df['AMT_INCOME_TOTAL'].quantile(0.75)
            q25_inc = df['AMT_INCOME_TOTAL'].quantile(0.25)
            df['High_Income_Applicant'] = (df['AMT_INCOME_TOTAL'] > q75_inc).astype(int)
            self._add_metadata("High_Income_Applicant", "Income > 75th percentile", "Affluent segment.", "High repayment capacity.", "Premium safe segment.")
            
            df['Low_Income_Customer'] = (df['AMT_INCOME_TOTAL'] < q25_inc).astype(int)
            self._add_metadata("Low_Income_Customer", "Income < 25th percentile", "Vulnerable segment.", "Highly sensitive to macro shocks.", "High default probability.")
            
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['Premium_Customer'] = ((df['AMT_INCOME_TOTAL'] > df['AMT_INCOME_TOTAL'].quantile(0.75)) & 
                                      (df['AMT_CREDIT'] > df['AMT_CREDIT'].quantile(0.75))).astype(int)
            self._add_metadata("Premium_Customer", "High Income AND High Credit", "Whale clients.", "Highest profit margin group.", "Careful underwriting required despite wealth.")
            
        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['Credit_Burden_Score'] = (df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)) * 100
            self._add_metadata("Credit_Burden_Score", "(Annuity / Income) * 100", "Percentage of income eaten by debt.", "Debt-To-Income proxy.", "Values > 30-40% indicate severe risk.")
            
        return df

    def generate_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Generating Risk Features...")
        # Purely heuristic based, no TARGET leakage
        
        if 'AMT_INCOME_TOTAL' in df.columns:
            df['Income_Risk_Level'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=4, labels=['High Risk', 'Medium-High', 'Medium-Low', 'Low Risk']).astype(str)
            self._add_metadata("Income_Risk_Level", "Quartiles of Income (Reversed)", "Income-based risk.", "Lower income is historically riskier.", "Separates baseline risk.")
            
        if 'Loan_to_Income_Ratio' in df.columns:
            df['Debt_Burden_Category'] = pd.cut(df['Loan_to_Income_Ratio'], bins=[-np.inf, 1, 3, 5, np.inf], labels=['Low', 'Moderate', 'High', 'Severe']).astype(str)
            self._add_metadata("Debt_Burden_Category", "Binned LTI Ratio", "Categorized debt pressure.", "Regulatory thresholds.", "Severe categories predict default strongly.")
            
        if 'AMT_CREDIT' in df.columns:
             df['Loan_Risk_Level'] = pd.qcut(df['AMT_CREDIT'], q=3, labels=['Low Exposure', 'Medium Exposure', 'High Exposure']).astype(str)
             self._add_metadata("Loan_Risk_Level", "Tertiles of Credit Amount", "Exposure-based risk.", "Higher loans carry more systemic risk.", "Affects underwriting depth.")

        # Synthetic Stability Index
        stability_score = np.zeros(len(df))
        if 'Employment_Stability' in df.columns: stability_score += df['Employment_Stability']
        if 'Owns_Both_House_And_Car' in df.columns: stability_score += df['Owns_Both_House_And_Car']
        if 'High_Income_Applicant' in df.columns: stability_score += df['High_Income_Applicant']
        
        df['Financial_Stability_Index'] = stability_score
        self._add_metadata("Financial_Stability_Index", "Sum of positive stability flags", "Synthetic composite score.", "Quick unified view of applicant strength.", "Highly predictive meta-feature.")
        
        return df

    def generate_feature_dictionary(self, filepath: str):
        """Generates a markdown dictionary of engineered features."""
        lines = ["# Engineered Feature Dictionary\n", "This document details the business logic behind all engineered features for FinSight.\n"]
        for md in self.feature_metadata:
            lines.append(f"## {md['Feature Name']}")
            lines.append(f"- **Formula**: `{md['Formula']}`")
            lines.append(f"- **Business Meaning**: {md['Business Meaning']}")
            lines.append(f"- **Why Bank Uses It**: {md['Why Bank Uses It']}")
            lines.append(f"- **Expected Impact**: {md['Expected Impact']}\n")
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logger.info(f"Feature dictionary saved to {filepath}")


class DataTransformer:
    """Handles data transformations: encoding, scaling, and feature selection."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.scalers = {}
        self.encoders = {}

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies appropriate encoding (One-Hot, Ordinal, Frequency) based on cardinality."""
        logger.info("Encoding categorical variables...")
        df_encoded = df.copy()
        
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in cat_cols:
            n_unique = df_encoded[col].nunique()
            
            # Binary/Low Cardinality -> One Hot Encoding
            if n_unique <= 5:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True, dtype=int)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
            # High Cardinality -> Frequency Encoding
            else:
                freq = df_encoded[col].value_counts(normalize=True)
                df_encoded[f'{col}_FREQ'] = df_encoded[col].map(freq)
                df_encoded = df_encoded.drop(columns=[col])
                
        return df_encoded

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies scaling based on feature skewness and presence of outliers."""
        logger.info("Scaling numerical variables...")
        df_scaled = df.copy()
        
        # Don't scale binary features, ID columns, or TARGET
        exclude = ['TARGET', 'SK_ID_CURR', 'CLIENT_ID']
        num_cols = [c for c in df_scaled.select_dtypes(include=[np.number]).columns 
                    if c not in exclude and df_scaled[c].nunique() > 2]
                    
        for col in num_cols:
            skewness = df_scaled[col].skew()
            
            # Highly skewed -> RobustScaler (handles outliers better)
            if abs(skewness) > 1.5:
                scaler = RobustScaler()
            # Uniform-ish -> MinMaxScaler
            elif abs(skewness) < 0.5:
                scaler = MinMaxScaler()
            # Normal-ish -> StandardScaler
            else:
                scaler = StandardScaler()
                
            # Fit and transform, preserving NaN handling if needed (fill temporary for scaler)
            temp_col = df_scaled[[col]].fillna(df_scaled[col].median())
            df_scaled[col] = scaler.fit_transform(temp_col)
            
            self.scalers[col] = scaler
            
        # Save scalers
        joblib.dump(self.scalers, os.path.join(self.model_dir, "feature_scalers.pkl"))
        logger.info("Scalers saved to models/feature_scalers.pkl")
        return df_scaled

    def select_features(self, df: pd.DataFrame, target_col: str = 'TARGET') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Removes low variance and highly redundant features. Returns dataset and feature report."""
        logger.info("Performing Feature Selection...")
        df_selected = df.copy()
        report_data = []
        
        # 1. Low Variance Filter (VarianceThreshold)
        num_df = df_selected.select_dtypes(include=[np.number])
        if not num_df.empty:
            vt = VarianceThreshold(threshold=0.01) # Near zero variance
            vt.fit(num_df.fillna(0))
            drop_vars = [c for c, keep in zip(num_df.columns, vt.get_support()) if not keep]
            
            for c in drop_vars:
                if c != target_col:
                    df_selected = df_selected.drop(columns=[c])
                    report_data.append({"Feature": c, "Action": "Dropped", "Reason": "Low Variance (<0.01)"})
        
        # 2. High Correlation Filter (Redundancy)
        # Sample to avoid huge memory spike
        sample_df = df_selected.select_dtypes(include=[np.number]).sample(min(10000, len(df_selected)), random_state=42)
        corr_matrix = sample_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
        for c in to_drop_corr:
            if c != target_col and c in df_selected.columns:
                df_selected = df_selected.drop(columns=[c])
                report_data.append({"Feature": c, "Action": "Dropped", "Reason": "Highly Correlated (>0.95)"})
                
        # 3. Simple Feature Importance (Correlation with target if available)
        if target_col in df_selected.columns:
            corrs = df_selected.select_dtypes(include=[np.number]).corrwith(df_selected[target_col]).abs().sort_values(ascending=False)
            for feat, val in corrs.items():
                if feat != target_col:
                    report_data.append({"Feature": feat, "Action": "Kept", "Reason": f"Correlation with Target: {val:.4f}"})
        
        report_df = pd.DataFrame(report_data)
        return df_selected, report_df
