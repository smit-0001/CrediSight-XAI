import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- CONFIGURATION ---
DATA_FILENAME = 'heloc_dataset_v1.csv' 
TARGET_COLUMN = 'RiskPerformance'
ARTIFACTS_DIR = 'artifacts'
PREPROCESSOR_FILENAME = os.path.join(ARTIFACTS_DIR, 'preprocessor_pipeline.joblib')
MODEL_FILENAME = os.path.join(ARTIFACTS_DIR, 'logistic_model.joblib')

def main():
    # 1. Load and Prepare Data
    print(f"Loading real data from '{DATA_FILENAME}'...")
    
    try:
        df = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILENAME}' not found.")
        print("Please update the 'DATA_FILENAME' variable in train.py")
        return

    print(f"Data loaded successfully with {len(df)} rows.")

    # Standardize all "missing" values to np.nan
    missing_placeholders = [-7, -8, -9]
    df.replace(missing_placeholders, np.nan, inplace=True)

    # --- Target Variable Preparation ---
    if TARGET_COLUMN not in df.columns:
         print(f"Error: Target column '{TARGET_COLUMN}' not found.")
         return
    
    # Map target variable (Bad=1, Good=0). 1 is the class we want to predict.
    df['Default'] = df[TARGET_COLUMN].map({'Bad': 1, 'Good': 0})
    
    if df['Default'].isnull().any():
        print("Warning: Found unmapped or NaN values in 'RiskPerformance'. Dropping these rows.")
        df.dropna(subset=['Default'], inplace=True)
    
    df['Default'] = df['Default'].astype(int)

    X = df.drop([TARGET_COLUMN, 'Default'], axis=1)
    y = df['Default']
    
    # --- Feature Preparation ---
    # Identify non-numeric columns in features. The pipeline can only handle numbers.
    numeric_cols = X.select_dtypes(include=np.number).columns
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    
    if len(non_numeric_cols) > 0:
        print(f"Warning: Non-numeric columns found in features and will be dropped: {list(non_numeric_cols)}")
    
    # We will ONLY use the numeric columns for this minimal model
    X_numeric = X[numeric_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # 2. Define Preprocessing Pipeline
    print("Defining preprocessing pipeline for numeric features...")
    
    # This pipeline will ONLY be applied to the numeric columns
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handles np.nan
        ('scaler', StandardScaler())                     # Scales data
    ])
    
    # 3. Train Preprocessor and Model
    print("Training preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Training LogisticRegression model...")
    # Add class_weight='balanced' to help with imbalanced data ("Bad" is likely the minority)
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train_processed, y_train)
    
    # 4. Save Artifacts
    print(f"Saving artifacts to '{ARTIFACTS_DIR}' directory...")
    joblib.dump(preprocessor, PREPROCESSOR_FILENAME)
    joblib.dump(model, MODEL_FILENAME)
    
    print("\n--- Artifacts Saved Successfully ---")
    print(f"1. {PREPROCESSOR_FILENAME}")
    print(f"2. {MODEL_FILENAME}")
    print("----------------------------------")

if __name__ == "__main__":
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    main()