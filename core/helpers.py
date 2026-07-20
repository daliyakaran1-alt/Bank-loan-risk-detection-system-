"""
General utility functions and helpers.
Contains reusable logic for date formatting, string manipulation, etc.
"""

from typing import Any

def safe_cast(val: Any, to_type: type, default: Any = None) -> Any:
    """
    Safely casts a value to a given type, returning a default on failure.
    """
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default
