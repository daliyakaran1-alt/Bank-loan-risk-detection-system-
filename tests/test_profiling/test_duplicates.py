"""
Tests for DuplicateAnalysis.
"""
import pytest
import pandas as pd
from profiling.duplicates import DuplicateAnalysis, DuplicateResult
from core.exceptions import ProfilingError

def test_duplicate_analysis_basic():
    """Test DuplicateAnalysis calculates metrics correctly."""
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [1, 1, 3]})
    analysis = DuplicateAnalysis(df)
    result = analysis.analyze()
    
    assert isinstance(result, DuplicateResult)
    assert result.total_duplicates == 1
    assert result.duplicate_percentage > 0.0

def test_potential_duplicate_columns():
    """Test identification of duplicate columns."""
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3], 'col3': [4, 5, 6]})
    analysis = DuplicateAnalysis(df)
    result = analysis.analyze()
    
    assert 'col1' in result.potential_duplicate_columns or 'col2' in result.potential_duplicate_columns
    
def test_duplicate_analysis_error_handling():
    """Test DuplicateAnalysis raises ProfilingError on failure."""
    analysis = DuplicateAnalysis(None) # type: ignore
    with pytest.raises(ProfilingError):
        analysis.analyze()
