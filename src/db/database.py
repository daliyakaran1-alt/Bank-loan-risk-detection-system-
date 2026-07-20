import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import config

# Set up the engine
# check_same_thread=False is needed for SQLite in Streamlit. It's ignored by Postgres.
connect_args = {}
if "sqlite" in config.DATABASE_URL:
    connect_args["check_same_thread"] = False

engine = create_engine(
    config.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Enables connection pooling health checks
    pool_recycle=3600    # Recycle connections after an hour
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Generator for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
