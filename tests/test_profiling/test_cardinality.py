import pytest
import pandas as pd
from src.profiling.cardinality import CardinalityAnalysis, CardinalityResult
from core.exceptions import ProfilingError

def test_cardinality_analysis():
    df = pd.DataFrame({
        'high_card': [f'val_{i}' for i in range(100)],
        'low_card': ['A', 'A', 'B', 'B', 'B'] * 20,
        'num_col': [1] * 100
    })
    
    analyzer = CardinalityAnalysis(df)
    result = analyzer.analyze(high_cardinality_threshold=0.9, low_cardinality_threshold=10)
    
    assert isinstance(result, CardinalityResult)
    assert 'high_card' in result.column_details
    assert 'low_card' in result.column_details
    assert 'num_col' not in result.column_details
    
    assert result.column_details['high_card'].is_high_cardinality is True
    assert result.column_details['low_card'].is_low_cardinality is True
    
    assert 'high_card' in result.high_cardinality_columns
    assert 'low_card' in result.low_cardinality_columns

def test_cardinality_analysis_empty_df():
    analyzer = CardinalityAnalysis(pd.DataFrame())
    with pytest.raises(ProfilingError):
        analyzer.analyze()
