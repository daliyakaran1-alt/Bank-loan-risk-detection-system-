"""
CRUD Operations for Database Interaction.
"""
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import bcrypt
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import json

from src.db.models import User, Customer, Prediction, RiskAssessment, Report, AuditLog, ModelRegistry
from src.db.database import engine, Base

try:
    from core.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_username_or_email(db: Session, login_id: str) -> Optional[User]:
    return db.query(User).filter((User.username == login_id) | (User.email == login_id)).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, username: str, email: str, raw_password: str, role: str, full_name: str = "") -> User:
    hashed = bcrypt.hashpw(raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = User(username=username, email=email, hashed_password=hashed, role=role, full_name=full_name)
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        raise ValueError(f"User with username '{username}' or email '{email}' already exists.")

def log_audit(db: Session, action: str, details: str = "", user_id: int = None):
    log_entry = AuditLog(action=action, details=details, user_id=user_id)
    db.add(log_entry)
    db.commit()

def create_customer(db: Session, name: str, annual_income: float, employment_years: float, age_years: float, owns_car: bool, owns_house: bool) -> Customer:
    customer_id = f"CUST-{uuid.uuid4().hex[:8].upper()}"
    customer = Customer(
        customer_id=customer_id,
        name=name,
        annual_income=annual_income,
        employment_years=employment_years,
        age_years=age_years,
        owns_car=owns_car,
        owns_house=owns_house
    )
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer

def get_customer_by_cid(db: Session, cid: str) -> Optional[Customer]:
    return db.query(Customer).filter(Customer.customer_id == cid).first()

def create_prediction(db: Session, customer_id: int, loan_amount: float, annuity_amount: float, prob: float, input_features: Dict[str, Any], model_id: int = None) -> Prediction:
    pred_id = f"PRD-{uuid.uuid4().hex[:8].upper()}"
    
    # clean numpy types for json
    clean_features = {}
    for k, v in input_features.items():
        if hasattr(v, 'item'):
            clean_features[k] = v.item()
        else:
            clean_features[k] = v
            
    prediction = Prediction(
        prediction_id=pred_id,
        customer_id=customer_id,
        model_id=model_id,
        loan_amount=loan_amount,
        annuity_amount=annuity_amount,
        probability_of_default=prob,
        input_features=clean_features
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

def create_risk_assessment(db: Session, prediction_id: int, score: int, category: str, decision: str, explanation: str, recommendations: Dict[str, str]) -> RiskAssessment:
    assessment = RiskAssessment(
        prediction_id=prediction_id,
        risk_score=score,
        risk_category=category,
        decision=decision,
        explanation=explanation,
        recommendations_json=recommendations
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment

def get_prediction_full(db: Session, pred_id: str) -> Optional[Prediction]:
    return db.query(Prediction).filter(Prediction.prediction_id == pred_id).first()

def search_predictions(db: Session, query: str) -> List[Prediction]:
    # Very basic search just by ID for now
    return db.query(Prediction).filter(Prediction.prediction_id.ilike(f"%{query}%")).all()

def search_customers(db: Session, query: str) -> List[Customer]:
    return db.query(Customer).filter(
        (Customer.customer_id.ilike(f"%{query}%")) | 
        (Customer.name.ilike(f"%{query}%"))
    ).all()
