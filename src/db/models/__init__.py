"""
Database Models for FinSight Platform.
All SQLAlchemy ORM models are defined here.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from src.db.database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="Viewer")
    full_name = Column(String, default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True)
    name = Column(String)
    annual_income = Column(Float)
    employment_years = Column(Float)
    age_years = Column(Float)
    owns_car = Column(Boolean, default=False)
    owns_house = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    model_id = Column(Integer, nullable=True)
    loan_amount = Column(Float)
    annuity_amount = Column(Float)
    probability_of_default = Column(Float)
    input_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    risk_score = Column(Integer)
    risk_category = Column(String)
    decision = Column(String)
    explanation = Column(Text)
    recommendations_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    format_type = Column(String)
    file_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String)
    details = Column(Text)
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    version = Column(String)
    accuracy = Column(Float)
    file_path = Column(String)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Applicant(Base):
    """New model for the Phase 2 dynamic portfolio system."""
    __tablename__ = "applicants"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    family_status = Column(String)
    children = Column(Integer)
    income = Column(Float)
    loan_amount = Column(Float)
    credit_score = Column(Float)
    owns_car = Column(String)
    owns_realty = Column(String)
    housing_type = Column(String)
    employment_years = Column(Float)
    education = Column(String)
    risk_score = Column(Float)
    risk_level = Column(String)
    recommendation = Column(String)
    reasons = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
