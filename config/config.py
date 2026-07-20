"""
Core configuration loader.

Responsible for parsing environment variables, loading secrets,
and orchestrating hierarchical configuration states.
"""

import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Base configuration class.
    """
    APP_ENV = os.getenv("APP_ENV", "development")
    DEBUG = os.getenv("APP_DEBUG", "true").lower() == "true"
    
    @classmethod
    def load(cls):
        """
        Validates and loads required configuration variables.
        """
        logger.debug("Loading application configuration.")
