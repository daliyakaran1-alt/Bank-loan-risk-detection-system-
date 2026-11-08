"""
Application Configuration.
Loads environment variables and sets up global settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database Configuration
    # Default to a local SQLite if POSTGRES_URL is not provided, to prevent hard crashes during local dev
    # However, production expects PostgreSQL.
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_DB = f"sqlite:///{os.path.join(_BASE_DIR, 'finsight_dev.db')}"
    DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DB)
    
    # Ensure psycopg2 is used for postgres URLs if specified as postgres:// instead of postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
        
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-default-key-change-in-prod")
    
    # ML Models
    MODEL_DIR = os.getenv("MODEL_DIR", "models/trained/")
    BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.joblib")
    
    # Reporting
    REPORT_DIR = os.getenv("REPORT_DIR", "reports/")

config = Config()
