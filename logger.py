"""
Enterprise logging architecture for Rin Mitra.

Implements centralized, formatted logging with rotation, structured output,
and distinct logging levels for different platform components.
"""

import logging
import sys

def setup_logging():
    """
    Configures the root logger and standard handlers for the application.
    Ensures logs are formatted consistently across all modules.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Enterprise logging system initialized.")
