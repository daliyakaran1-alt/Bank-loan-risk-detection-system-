"""
Middleware layer.

Handles request interception, session management, auditing, and pre-processing
for incoming user actions or API requests.
"""

import logging

logger = logging.getLogger(__name__)

def audit_action(action_name: str, user_id: str):
    """
    Logs an audited action to the enterprise audit trail.
    """
    logger.info(f"AUDIT: User {user_id} performed {action_name}")
