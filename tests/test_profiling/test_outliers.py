import pytest
import pandas as pd
import numpy as np
from src.profiling.outliers import OutlierAnalysis, OutlierResult
from core.exceptions import ProfilingError

def test_outlier_analysis_iqr():
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100).tolist() + [10.0, -10.0],
        'B': ['cat'] * 102
    })
    analyzer = OutlierAnalysis(df)
    result = analyzer.analyze(method='iqr')
    
    assert isinstance(result, OutlierResult)
    assert 'A' in result.affected_columns
    assert 'B' not in result.affected_columns
    assert result.column_details['A'].count >= 2
    assert result.method_used == 'iqr'

def test_outlier_analysis_zscore():
    df = pd.DataFrame({
        'A': [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]
    })
    analyzer = OutlierAnalysis(df)
    result = analyzer.analyze(method='zscore', threshold=2.0)
    
    assert 'A' in result.affected_columns
    assert 100 in df['A'][result.column_details['A'].indices].values

def test_outlier_analysis_modified_zscore():
    df = pd.DataFrame({
        'A': [1, 2, 2, 3, 3, 3, 4, 4, 5, 100]
    })
    analyzer = OutlierAnalysis(df)
    result = analyzer.analyze(method='modified_zscore', threshold=3.5)
    
    assert 'A' in result.affected_columns
    assert 100 in df['A'][result.column_details['A'].indices].values

def test_outlier_analysis_empty_df():
    analyzer = OutlierAnalysis(pd.DataFrame())
    with pytest.raises(ProfilingError):
        analyzer.analyze()

def test_outlier_analysis_invalid_method():
    df = pd.DataFrame({'A': [1, 2, 3]})
    analyzer = OutlierAnalysis(df)
    with pytest.raises(ProfilingError):
        analyzer.analyze(method='invalid_method')
