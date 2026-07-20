"""
Train a lightweight interactive ML model for manual loan application inputs.
Uses features that map directly to the form fields users fill in.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_interactive_model():
    print("Loading dataset...")
    df = pd.read_csv('data/processed/application_train_cleaned.csv')
    
    # --- Feature Engineering ---
    # Age from DAYS_BIRTH
    df['AGE'] = (df['DAYS_BIRTH'] / -365).astype(int)
    
    # Credit Score from EXT_SOURCE_2 (scaled to 300-850 range)
    df['CREDIT_SCORE'] = 300 + (df['EXT_SOURCE_2'].fillna(0.5) * 550)
    
    # Employment Years from DAYS_EMPLOYED
    df['EMPLOYMENT_YEARS'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x == 365243 else max(0, x / -365.0))
    
    # --- Select Features ---
    feature_cols = [
        'AMT_INCOME_TOTAL', 'AGE', 'AMT_CREDIT', 'CREDIT_SCORE', 'EMPLOYMENT_YEARS',
        'CNT_CHILDREN'
    ]
    cat_cols = ['NAME_EDUCATION_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
    
    # Label encode categorical features
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        feature_cols.append(col)
    
    # Drop rows with NaN in our selected features
    df = df.dropna(subset=feature_cols + ['TARGET'])
    
    X = df[feature_cols].values
    y = df['TARGET'].values
    
    print(f"Training set size: {len(X)} rows, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"Default rate: {y.mean()*100:.1f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    # Save
    os.makedirs('models/trained', exist_ok=True)
    joblib.dump(model, 'models/trained/interactive_model.joblib')
    joblib.dump(feature_cols, 'models/trained/interactive_features.joblib')
    joblib.dump(encoders, 'models/trained/interactive_encoders.joblib')
    
    print("Saved: interactive_model.joblib, interactive_features.joblib, interactive_encoders.joblib")
    
    # Quick test
    test_input = X_test[0:1]
    prob = model.predict_proba(test_input)[0][1]
    print(f"Test prediction probability: {prob*100:.1f}%")

if __name__ == '__main__':
    train_interactive_model()
