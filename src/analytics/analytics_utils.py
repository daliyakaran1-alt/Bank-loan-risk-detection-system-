"""
Utility functions for the Analytics Engine.

Provides common data manipulation, safe math, and formatting routines.
"""

import logging
from typing import Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform division safely handling zero or null denominators."""
    try:
        if denominator is None or denominator == 0:
            return default
        result = numerator / denominator
        if pd.isna(result) or np.isinf(result):
            return default
        return float(result)
    except Exception:
        return default

def safe_mean(series: pd.Series, default: float = 0.0) -> float:
    """Calculate mean safely."""
    if series is None or series.empty:
        return default
    val = series.mean()
    return float(val) if pd.notna(val) else default

def safe_median(series: pd.Series, default: float = 0.0) -> float:
    """Calculate median safely."""
    if series is None or series.empty:
        return default
    val = series.median()
    return float(val) if pd.notna(val) else default

def format_currency(value: float) -> str:
    """Format a number as INR currency (e.g., ₹1.5L)."""
    if pd.isna(value) or value is None:
        return "₹0"
    
    abs_val = abs(value)
    if abs_val >= 10_000_000:
        return f"₹{value / 10_000_000:.2f}Cr"
    elif abs_val >= 100_000:
        return f"₹{value / 100_000:.2f}L"
    elif abs_val >= 1_000:
        return f"₹{value / 1_000:.2f}K"
    else:
        return f"₹{value:,.0f}"

def format_percentage(value: float) -> str:
    """Format a ratio as a percentage string (e.g., 0.15 -> '15.00%')."""
    if pd.isna(value) or value is None:
        return "0.00%"
    return f"{value * 100:.2f}%"

def bin_numerical_column(df: pd.DataFrame, column: str, bins: int = 5, labels: Optional[List[str]] = None) -> pd.Series:
    """Bin a numerical column into categories."""
    if column not in df.columns or df[column].empty:
        return pd.Series(dtype=str)
    
    try:
        return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True).astype(str)
    except Exception as e:
        logger.warning(f"Failed to bin column {column}: {str(e)}")
        return pd.Series(dtype=str)
