import joblib
import pandas as pd
import xgboost as xgb
import shap
import re
import os
from typing import List
from fastapi import HTTPException
from .models import CreditAppFeatures, FeatureExplanation

# --- 1. Artifacts Dictionary ---
# This will be imported by main.py to get access to the loaded models
artifacts = {
    "preprocessor": None,
    "model_logistic": None,
    "model_xgb": None,
    "feature_order": None,
    "explainer_xgb": None
}

# --- 2. Artifact Loading Function ---
def load_artifacts():
    print("Loading all artifacts...")
    
    preprocessor_path = "artifacts/preprocessor_pipeline.joblib"
    logistic_model_path = "artifacts/logistic_model.joblib"
    xgb_model_path = "artifacts/xgb_model.json"
    fixed_xgb_model_path = "artifacts/xgb_model_fixed.json"

    try:
        # Load Preprocessor
        artifacts["preprocessor"] = joblib.load(preprocessor_path)
        artifacts["feature_order"] = artifacts["preprocessor"].feature_names_in_
        print(f"Feature order: {artifacts['feature_order']}")

        # Load Logistic Regression Model
        artifacts["model_logistic"] = joblib.load(logistic_model_path)
        print("Logistic Regression model loaded.")

        # --- Hot-patch the broken XGBoost model file ---
        print(f"Attempting to hot-patch {xgb_model_path}...")
        try:
            with open(xgb_model_path, 'r') as f:
                model_content = f.read()
            
            fixed_content, num_replacements = re.subn(
                r'("base_score":\s*)"\[5E-1\]"', 
                r'\1 "0.5"',  # Replaces with a valid JSON string
                model_content
            )

            if num_replacements > 0:
                print(f"Successfully hot-patched 'base_score' (found {num_replacements} instance(s)).")
            else:
                print("Warning: 'base_score' hot-patch did not find the target string. The file might already be clean.")

            with open(fixed_xgb_model_path, 'w') as f:
                f.write(fixed_content)

        except Exception as e:
            print(f"CRITICAL ERROR during model hot-patch: {e}")
            raise
        
        # --- Load Models from fixed file ---
        artifacts["model_xgb"] = xgb.XGBClassifier()
        artifacts["model_xgb"].load_model(fixed_xgb_model_path)
        print(f"XGBoost model loaded from {fixed_xgb_model_path}.")

        # --- Create SHAP Explainer ---
        print("Creating SHAP Explainer for XGBoost model...")
        artifacts["explainer_xgb"] = shap.TreeExplainer(artifacts["model_xgb"])
        print("SHAP Explainer created.")

        print("All artifacts loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: An artifact file was not found. {e}")
        # In a real app, you might want to exit
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        # This will catch any other errors during the process

# --- 3. ML Helper Functions ---

def _preprocess_data(features: CreditAppFeatures) -> pd.DataFrame:
    """Internal helper function to preprocess raw JSON data."""
    if not artifacts["preprocessor"]:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded.")
    
    data_dict = features.model_dump()
    data_df = pd.DataFrame([data_dict], columns=artifacts["feature_order"])
    
    return artifacts["preprocessor"].transform(data_df)

def _get_prediction(features: CreditAppFeatures, model):
    """Internal helper function to run the prediction pipeline."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    transformed_data = _preprocess_data(features)
    
    proba = model.predict_proba(transformed_data)
    prob_default = proba[0][1]
    
    return prob_default

def _generate_summary(base_value: float, explanations: List[FeatureExplanation]) -> str:
    """
    Translates the raw SHAP values into a human-readable summary.
    """
    # Calculate the final prediction (in log-odds) by summing the base value and all contributions
    all_shap_values = [ex.shap_value for ex in explanations]
    final_log_odds = base_value + sum(all_shap_values)
    
    # Check if the final prediction is higher (High Risk) or lower (Low Risk) than the average
    is_high_risk = final_log_odds > base_value

    if is_high_risk:
        # --- HIGH-RISK LOGIC (The one we already have) ---
        drivers = [ex for ex in explanations if ex.shap_value > 0][:3]
        mitigator = min((ex for ex in explanations if ex.shap_value < 0), key=lambda x: x.shap_value, default=None)
        
        if not drivers:
            return "The prediction is slightly above average, but no single strong risk factor was identified."

        # Format the list with 'and' for the last item
        driver_names = [d.feature for d in drivers]
        if len(driver_names) == 1:
            drivers_str = driver_names[0]
        elif len(driver_names) == 2:
            drivers_str = " and ".join(driver_names)
        else:
            drivers_str = ", ".join(driver_names[:-1]) + ", and " + driver_names[-1]

        summary = f"This is a high-risk profile, primarily driven by: {drivers_str}."

        if mitigator:
            summary += f" While factors like {mitigator.feature} were a positive, "
            summary += "it was not enough to offset the primary risk factors."
        
        return summary

    else:
        # --- LOW-RISK LOGIC (New) ---
        drivers = [ex for ex in explanations if ex.shap_value < 0][:3] # Negative drivers are good
        risk_factor = max((ex for ex in explanations if ex.shap_value > 0), key=lambda x: x.shap_value, default=None)

        if not drivers:
             return "The prediction is in line with the average; no significant factors were identified."

        # Format the list with 'and' for the last item
        driver_names = [d.feature for d in drivers]
        if len(driver_names) == 1:
            drivers_str = driver_names[0]
        elif len(driver_names) == 2:
            drivers_str = " and ".join(driver_names)
        else:
            drivers_str = ", ".join(driver_names[:-1]) + ", and " + driver_names[-1]

        summary = f"This is a low-risk profile, primarily due to positive factors like: {drivers_str}."

        if risk_factor:
            summary += f" A minor risk was noted ({risk_factor.feature}), but it was offset by the strong positive factors."
        
        return summary