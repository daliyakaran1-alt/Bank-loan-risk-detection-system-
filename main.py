"""
Entry point for the Rin Mitra Enterprise Banking Analytics Platform.

This module initializes the platform environment and launches the core application.
"""

import sys
import logging
from app import run_app

logger = logging.getLogger(__name__)

def main():
    """
    Main execution function.
    Initializes environment variables, validates system state, and starts the application.
    """
    try:
        logger.info("Initializing Rin Mitra Enterprise Banking Analytics Platform")
        run_app()
    except Exception as e:
        logger.critical(f"Critical failure during application startup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
