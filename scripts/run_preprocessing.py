"""
Runner script for Phase 5 Data Preprocessing.
Executes the data cleaning pipeline on the application_train dataset.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import DataPreprocessor
from logger import setup_logging

def main():
    setup_logging()
    
    input_file = "data/raw/application_train.csv"
    output_file = "data/processed/application_train_cleaned.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return

    print("--- Starting Phase 5: Data Cleaning & Preprocessing ---")
    
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline(file_path=input_file, output_path=output_file)
    
    print("--- Phase 5 Completed successfully! ---")

if __name__ == "__main__":
    main()
