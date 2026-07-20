"""
Date Utilities.
Helper functions for handling dates, times, and timezones.
"""
from datetime import datetime, timezone

def get_current_timestamp() -> datetime:
    """
    Returns the current UTC timestamp.
    """
    return datetime.now(timezone.utc)
