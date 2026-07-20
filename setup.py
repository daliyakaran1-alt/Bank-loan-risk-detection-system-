"""
Setup script for Rin Mitra platform distribution.
"""

from setuptools import setup, find_packages

setup(
    name="rin_mitra",
    version="0.1.0",
    description="Enterprise Banking Analytics Platform",
    author="Rin Mitra Team",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "shap",
        "plotly",
    ],
    python_requires=">=3.9",
)
