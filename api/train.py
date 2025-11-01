import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb # <-- IMPORT XGBOOST
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_FILENAME = 'heloc_dataset_v1.csv' 
TARGET_COLUMN = 'RiskPerformance'
ARTIFACTS_DIR = 'artifacts'
PREPROCESSOR_FILENAME = os.path.join(ARTIFACTS_DIR, 'preprocessor_pipeline.joblib')

# --- NEW MODEL FILENAME ---
MODEL_FILENAME = os.path.join(ARTIFACTS_DIR, 'xgb_model.json') # <-- NEW FILENAME

def main():
    # 1. Load and Prepare Data
    print(f"Loading real data from '{DATA_FILENAME}'...")
    try:
        df = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILENAME}' not found.")
        return

    print(f"Data loaded successfully with {len(df)} rows.")

    missing_placeholders = [-7, -8, -9]
    df.replace(missing_placeholders, np.nan, inplace=True)

    # --- Target Variable Preparation ---
    df['Default'] = df[TARGET_COLUMN].map({'Bad': 1, 'Good': 0})
    if df['Default'].isnull().any():
        print("Warning: Found unmapped or NaN values in 'RiskPerformance'. Dropping rows.")
        df.dropna(subset=['Default'], inplace=True)
    df['Default'] = df['Default'].astype(int)

    X = df.drop([TARGET_COLUMN, 'Default'], axis=1)
    y = df['Default']

    numeric_cols = X.select_dtypes(include=np.number).columns
    X_numeric = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # 2. Define and FIT Preprocessing Pipeline
    # We still need this for the API!
    print("Defining and training preprocessing pipeline...")
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # IMPORTANT: We still need to save the preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_FILENAME)
    print(f"Preprocessor saved to {PREPROCESSOR_FILENAME}")

    # 3. Train XGBoost Model
    print("Training XGBoost model...")

    # --- Handle Class Imbalance (as per your original plan) ---
    # scale_pos_weight = count(negative_class) / count(positive_class)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_json=True
        #, base_score=0.5
    )

    model.fit(X_train_processed, y_train)

    # 4. Save Artifacts
    print(f"Saving XGBoost model to '{MODEL_FILENAME}'...")
    model.save_model(MODEL_FILENAME) # <-- NEW SAVING METHOD

    print("\n--- Iteration 2 Artifacts Saved Successfully ---")
    print(f"1. {PREPROCESSOR_FILENAME} (unchanged)")
    print(f"2. {MODEL_FILENAME} (NEW XGBoost model)")
    print("--------------------------------------------------")

if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    main()