"""
Mock Data Loader.

Provides a synthetic or cached dataset for UI testing and analytics rendering.
"""

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import os
import streamlit as st

@st.cache_data
def load_real_data(nrows=None) -> pd.DataFrame:
    """Load the actual enterprise dataset."""
    path_segmented = "data/processed/application_train_segmented.csv"
    path_cleaned = "data/processed/application_train_cleaned.csv"
    
    if os.path.exists(path_segmented):
        return pd.read_csv(path_segmented, nrows=nrows)
    elif os.path.exists(path_cleaned):
        return pd.read_csv(path_cleaned, nrows=nrows)
    else:
        st.error("Enterprise dataset not found! Please ensure data processing pipeline has been run.")
        return pd.DataFrame()
