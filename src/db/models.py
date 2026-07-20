"""
SQLAlchemy Models for the FinSight Enterprise Platform.
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from src.db.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, nullable=False) # Administrator, Executive, Loan Officer, Risk Analyst, Viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    audit_logs = relationship("AuditLog", back_populates="user")
    reports = relationship("Report", back_populates="author")

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True, nullable=False) # e.g. CUST-1234
    name = Column(String)
    annual_income = Column(Float)
    employment_years = Column(Float)
    age_years = Column(Float)
    owns_car = Column(Boolean)
    owns_house = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="customer")

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String, unique=True, index=True, nullable=False)
    algorithm = Column(String)
    training_date = Column(DateTime, default=datetime.utcnow)
    feature_list = Column(JSON)
    metrics_json = Column(JSON)
    is_active = Column(Boolean, default=False)
    
    predictions = relationship("Prediction", back_populates="model")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True, nullable=False) # e.g. PRD-ABCD
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("model_registry.id"))
    
    loan_amount = Column(Float)
    annuity_amount = Column(Float)
    input_features = Column(JSON)
    
    probability_of_default = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="predictions")
    model = relationship("ModelRegistry", back_populates="predictions")
    risk_assessment = relationship("RiskAssessment", back_populates="prediction", uselist=False)

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    
    risk_score = Column(Integer, nullable=False) # 300-900
    risk_category = Column(String, nullable=False) # Excellent, High Risk, etc.
    decision = Column(String, nullable=False) # Approve, Reject, Manual Review
    explanation = Column(Text)
    recommendations_json = Column(JSON)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="risk_assessment")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    report_type = Column(String, nullable=False) # Loan Assessment, Architecture Audit, etc.
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="reports")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String, nullable=False)
    details = Column(Text)
    
    user = relationship("User", back_populates="audit_logs")
