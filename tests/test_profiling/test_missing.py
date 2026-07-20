"""
Tests for MissingValueAnalysis.
"""
import pytest
import pandas as pd
from profiling.missing import MissingValueAnalysis, MissingResult
from core.exceptions import ProfilingError

def test_missing_analysis_basic():
    """Test MissingValueAnalysis calculates metrics correctly."""
    df = pd.DataFrame({'a': [1, None, 3], 'b': [None, None, 3]})
    analysis = MissingValueAnalysis(df)
    result = analysis.analyze()
    
    assert isinstance(result, MissingResult)
    assert result.total_missing == 3
    assert result.missing_by_column['b'] == 2
    assert result.column_ranking[0] == 'b'

def test_missing_analysis_empty():
    """Test MissingValueAnalysis handles empty datasets."""
    df = pd.DataFrame()
    analysis = MissingValueAnalysis(df)
    result = analysis.analyze()
    
    assert result.total_missing == 0
    assert len(result.missing_by_column) == 0

def test_missing_analysis_error_handling():
    """Test MissingValueAnalysis raises ProfilingError on failure."""
    analysis = MissingValueAnalysis(None) # type: ignore
    with pytest.raises(ProfilingError):
        analysis.analyze()
