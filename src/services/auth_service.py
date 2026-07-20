"""
Authentication Service.
Handles User login, roles, password hashing, and session management for Flask.
"""
import bcrypt
from sqlalchemy.orm import Session
from src.db import crud
from src.db.database import SessionLocal, engine
from src.db.models import Base

def init_db_and_auth():
    """Ensures database exists and default admin user is present."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = crud.get_user_by_username(db, "admin")
        if not admin:
            crud.create_user(
                db=db,
                username="admin",
                email="admin@finsight.local",
                raw_password="admin",
                role="Administrator",
                full_name="System Administrator"
            )
            # Create some dummy users for other roles
            crud.create_user(db, "exec", "exec@finsight.local", "exec", "Executive", "Chief Executive")
            crud.create_user(db, "loanofficer", "lo@finsight.local", "loanofficer", "Loan Officer", "Jane Officer")
            crud.create_user(db, "risk", "risk@finsight.local", "risk", "Risk Analyst", "John Risk")
    finally:
        db.close()

def authenticate_user(db: Session, username: str, raw_password: str) -> dict:
    """
    Authenticates a user and returns their data if successful, otherwise None.
    """
    user = crud.get_user_by_username_or_email(db, username)
    if not user:
        return None
        
    # Check password
    is_valid = bcrypt.checkpw(raw_password.encode('utf-8'), user.hashed_password.encode('utf-8'))
    if is_valid:
        crud.log_audit(db, "LOGIN_SUCCESS", f"User logged in from Web UI", user.id)
        return {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "full_name": user.full_name
        }
        
    crud.log_audit(db, "LOGIN_FAILED", f"Failed login attempt for username: {username}")
    return None
