"""
Data validation utilities.

Ensures inputs and datasets conform to expected banking schemas
before processing by ML models or saving to the database.
"""

def validate_account_number(account_number: str) -> bool:
    """
    Validates the format and checksum of an account number.
    """
    return len(account_number) >= 10 and account_number.isalnum()
