"""
Segmentation Engine Module.

Handles Unsupervised Machine Learning for Customer Segmentation,
Dimensionality Reduction (PCA), and Risk Profiling Persona generation.
"""

import logging
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, Tuple, List

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class SegmentationEngine:
    """Enterprise Customer Segmentation and Persona Generation Engine."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model = None
        self.pca_model = None
        self.personas = {}

    def _select_numeric_features(self, df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
        """Selects only numeric features and drops target/ids."""
        num_df = df.select_dtypes(include=[np.number])
        cols_to_drop = [c for c in exclude_cols if c in num_df.columns]
        return num_df.drop(columns=cols_to_drop)

    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 3, variance_threshold: float = 0.95) -> pd.DataFrame:
        """Applies PCA to reduce dimensionality while preserving variance."""
        logger.info(f"Applying PCA dimensionality reduction...")
        num_df = self._select_numeric_features(df, exclude_cols=["TARGET", "SK_ID_CURR", "CLIENT_ID"])
        
        # Fit PCA
        pca = PCA(n_components=variance_threshold) if variance_threshold else PCA(n_components=n_components)
        pca_result = pca.fit_transform(num_df.fillna(0))
        self.pca_model = pca
        
        logger.info(f"PCA reduced dimensions to {pca.n_components_} preserving {sum(pca.explained_variance_ratio_):.2%} variance.")
        
        # Create PCA DataFrame
        pca_cols = [f'PCA_Component_{i+1}' for i in range(pca.n_components_)]
        df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=df.index)
        
        # Keep only top 3 for visualization compatibility even if variance requires more
        if pca.n_components_ > 3:
            df_pca['PCA_Component_1'] = pca_result[:, 0]
            df_pca['PCA_Component_2'] = pca_result[:, 1]
            df_pca['PCA_Component_3'] = pca_result[:, 2]
            
        return df_pca

    def evaluate_clusters(self, data: np.ndarray, k_range: range = range(3, 8)) -> Dict[int, Dict[str, float]]:
        """Evaluates clustering algorithms across a range of K using multiple metrics."""
        logger.info("Evaluating optimal number of clusters...")
        results = {}
        
        # Sample for performance if data is huge
        sample_data = data
        if len(data) > 20000:
            np.random.seed(42)
            indices = np.random.choice(len(data), 20000, replace=False)
            sample_data = data[indices]

        for k in k_range:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
            labels = kmeans.fit_predict(sample_data)
            
            sil = silhouette_score(sample_data, labels)
            ch = calinski_harabasz_score(sample_data, labels)
            db = davies_bouldin_score(sample_data, labels)
            
            results[k] = {
                'Silhouette': sil,
                'Calinski-Harabasz': ch,
                'Davies-Bouldin': db
            }
            logger.info(f"K={k} -> Silhouette: {sil:.3f}, CH: {ch:.0f}, DB: {db:.3f}")
            
        return results

    def fit_predict_clusters(self, df: pd.DataFrame, df_pca: pd.DataFrame, n_clusters: int = 5, algorithm: str = 'kmeans') -> pd.DataFrame:
        """Fits the chosen algorithm and assigns cluster labels."""
        logger.info(f"Fitting {algorithm.upper()} with {n_clusters} clusters...")
        
        data = df_pca.values
        
        if algorithm == 'kmeans':
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            
        labels = model.fit_predict(data)
        self.best_model = model
        
        df_result = df.copy()
        df_result['Cluster'] = labels
        df_result['Cluster'] = df_result['Cluster'].astype(str)
        
        # Merge PCA back in for dashboard
        for col in ['PCA_Component_1', 'PCA_Component_2', 'PCA_Component_3']:
            if col in df_pca.columns:
                df_result[col] = df_pca[col]
                
        # Save model
        joblib.dump(self.best_model, os.path.join(self.model_dir, 'clustering_model.pkl'))
        
        return df_result

    def _safe_mean(self, group, col_name, default=0):
        if col_name in group:
            return group[col_name].mean()
        return default

    def generate_personas(self, df_clustered: pd.DataFrame) -> Dict[str, Any]:
        """Automatically generates business personas based on cluster centroids."""
        logger.info("Generating Business Personas...")
        
        # Calculate global medians/means for comparison
        global_inc = self._safe_mean(df_clustered, 'AMT_INCOME_TOTAL', default=150000)
        global_cred = self._safe_mean(df_clustered, 'AMT_CREDIT', default=500000)
        global_age = abs(self._safe_mean(df_clustered, 'DAYS_BIRTH', default=-15000)) / 365.25
        global_def = self._safe_mean(df_clustered, 'TARGET', default=0.08)
        
        grouped = df_clustered.groupby('Cluster')
        personas = {}
        
        for cluster, group in grouped:
            c_inc = self._safe_mean(group, 'AMT_INCOME_TOTAL')
            c_cred = self._safe_mean(group, 'AMT_CREDIT')
            c_age = abs(self._safe_mean(group, 'DAYS_BIRTH')) / 365.25 if 'DAYS_BIRTH' in group else 40
            c_def = self._safe_mean(group, 'TARGET')
            count = len(group)
            
            # Heuristics for naming
            if c_inc > global_inc * 1.5 and c_cred > global_cred * 1.2:
                name = "Premium Borrowers"
                desc = "High-income individuals seeking large credit lines."
                rec = "Offer premium banking services and large asset-backed loans."
            elif c_def > global_def * 1.5:
                name = "High Risk Borrowers"
                desc = "Vulnerable segment with significantly elevated default rates."
                rec = "Implement stringent underwriting, require collateral, or deny credit."
            elif c_age < global_age * 0.85 and c_inc >= global_inc * 0.8:
                name = "Young Professionals"
                desc = "Younger applicants with stable or growing income."
                rec = "Target with first-home mortgages and credit builder products."
            elif c_def < global_def * 0.6:
                name = "Financially Stable Customers"
                desc = "Highly reliable borrowers with low default probability."
                rec = "Provide pre-approved personal loans and retention offers."
            else:
                name = "Mainstream Credit Consumers"
                desc = "Average income and credit profiles representing the core customer base."
                rec = "Standard underwriting and cross-selling of consumer loans."
                
            # Append cluster ID to make unique in case rules overlap
            name = f"{name} (C{cluster})"
            
            personas[cluster] = {
                "Cluster Name": name,
                "Customer Count": count,
                "Average Income": c_inc,
                "Average Credit": c_cred,
                "Average Age": c_age,
                "Default Rate": c_def,
                "Key Characteristics": desc,
                "Bank Recommendation": rec
            }
            
        self.personas = personas
        return personas

    def generate_persona_markdown(self, filepath: str):
        """Export personas to Markdown."""
        lines = ["# Customer Segmentation Personas\n"]
        for c_id, details in self.personas.items():
            lines.append(f"## {details['Cluster Name']}")
            lines.append(f"- **Customer Count**: {details['Customer Count']:,}")
            lines.append(f"- **Average Income**: {details['Average Income']:,.2f}")
            lines.append(f"- **Average Credit**: {details['Average Credit']:,.2f}")
            lines.append(f"- **Average Age**: {details['Average Age']:.1f} years")
            lines.append(f"- **Default Rate**: {details['Default Rate']:.2%}")
            lines.append(f"- **Key Characteristics**: {details['Key Characteristics']}")
            lines.append(f"- **Bank Recommendation**: {details['Bank Recommendation']}\n")
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logger.info(f"Customer Personas saved to {filepath}")

    def profile_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a composite Overall Risk Profile for each customer."""
        df_risk = df.copy()
        logger.info("Calculating Overall Risk Profile...")
        
        # Synthetic risk calculation combining existing burdens and cluster defaults
        # Assume cluster defaults are mapped back
        cluster_defaults = {k: v['Default Rate'] for k, v in self.personas.items()}
        
        if 'Cluster' in df_risk.columns and cluster_defaults:
            df_risk['Cluster_Risk_Score'] = df_risk['Cluster'].map(cluster_defaults)
        else:
            df_risk['Cluster_Risk_Score'] = 0.08 # fallback global avg
            
        # Combine metrics if they exist (standardized)
        base_risk = df_risk['Cluster_Risk_Score'] * 10
        
        if 'Financial_Stability_Index' in df_risk.columns:
            # Higher stability lowers risk
            base_risk = base_risk - (df_risk['Financial_Stability_Index'] * 0.1)
            
        if 'Credit_Burden_Score' in df_risk.columns:
            # Higher burden increases risk
            base_risk = base_risk + (df_risk['Credit_Burden_Score'] * 0.05)
            
        # Normalize to 0-100 scale
        df_risk['Overall_Risk_Profile'] = ((base_risk - base_risk.min()) / (base_risk.max() - base_risk.min())) * 100
        df_risk['Overall_Risk_Profile'] = df_risk['Overall_Risk_Profile'].fillna(50.0).clip(0, 100)
        
        df_risk['Risk_Category'] = pd.qcut(df_risk['Overall_Risk_Profile'], q=4, labels=False, duplicates='drop').astype(str)
        # Map labels manually if bin count changes, or just use pd.cut if qcut drops bins
        bins = pd.qcut(df_risk['Overall_Risk_Profile'], q=4, duplicates='drop', retbins=True)[1]
        labels = ['Low', 'Medium', 'High', 'Severe'][:len(bins)-1]
        df_risk['Risk_Category'] = pd.cut(df_risk['Overall_Risk_Profile'], bins=bins, labels=labels, include_lowest=True).astype(str)
        
        return df_risk
