"""
Tests for DataQuality.
"""
import pytest
import pandas as pd
from profiling.quality import DataQuality, QualityResult
from core.exceptions import ProfilingError

def test_data_quality_basic():
    """Test DataQuality calculates metrics correctly on a basic dataset."""
    df = pd.DataFrame({'a': [1, None, 3], 'b': [1, 2, 2]})
    quality = DataQuality(df)
    result = quality.evaluate()
    
    assert isinstance(result, QualityResult)
    assert result.missing_ratio > 0.0
    assert result.duplicate_ratio == 0.0

def test_data_quality_empty():
    """Test DataQuality handles empty datasets."""
    df = pd.DataFrame()
    quality = DataQuality(df)
    result = quality.evaluate()
    
    assert result.health_score == 0.0
    assert result.missing_ratio == 0.0
    assert result.duplicate_ratio == 0.0

def test_data_quality_error_handling():
    """Test DataQuality raises ProfilingError on failure."""
    stats = DataQuality(None) # type: ignore
    with pytest.raises(ProfilingError):
        stats.evaluate()
