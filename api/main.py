import joblib
import pandas as pd
import xgboost as xgb
import shap  # <-- 1. Import SHAP
from fastapi import FastAPI, HTTPException
from typing import List
from .models import CreditAppFeatures, PredictionResponse, ExplanationResponse # <-- 2. Import new model
# from .models import CreditAppFeatures, PredictionResponse, ExplanationResponse, FeatureExplanation

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="CrediSight XAI Engine (XGB + SHAP)",
    description="API for credit risk prediction and explanation.",
    version="0.3.0"
)

# --- 2. Load Artifacts On Startup ---
artifacts = {
    "preprocessor": None,
    "model_logistic": None,
    "model_xgb": None,
    "feature_order": None,
    "explainer_xgb": None  # <-- 3. Add a spot for our SHAP explainer
}

@app.on_event("startup")
async def load_artifacts():
    print("Loading all artifacts...")
    
    preprocessor_path = "artifacts/preprocessor_pipeline.joblib"
    logistic_model_path = "artifacts/logistic_model.joblib"
    xgb_model_path = "artifacts/xgb_model.json"
    
    try:
        # Load Preprocessor (shared)
        artifacts["preprocessor"] = joblib.load(preprocessor_path)
        artifacts["feature_order"] = artifacts["preprocessor"].feature_names_in_
        print(f"Feature order: {artifacts['feature_order']}")

        # Load Logistic Regression Model
        artifacts["model_logistic"] = joblib.load(logistic_model_path)
        print("Logistic Regression model loaded.")

        # Load XGBoost Model
        artifacts["model_xgb"] = xgb.XGBClassifier()
        artifacts["model_xgb"].load_model(xgb_model_path)
        print("XGBoost model loaded.")

        # --- 4. Create and load SHAP Explainer ---
        print("Creating SHAP Explainer for XGBoost model...")
        model_booster = artifacts["model_xgb"].get_booster()
        # artifacts["explainer_xgb"] = shap.Explainer(model_booster)
        artifacts["explainer_xgb"] = shap.TreeExplainer(model_booster, feature_perturbation="interventional")

        print("SHAP Explainer created.")

        print("All artifacts loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: An artifact file was not found. {e}")
        
# --- 3. Refactored Helper Functions ---

def _preprocess_data(features: CreditAppFeatures) -> pd.DataFrame:
    """
    Internal helper function to preprocess raw JSON data.
    """
    if not artifacts["preprocessor"]:
        raise HTTPException(status_code=500, detail="Preprocessor not loaded.")
    
    # Convert Pydantic model to a DataFrame, enforcing the correct column order
    data_dict = features.model_dump()
    data_df = pd.DataFrame([data_dict], columns=artifacts["feature_order"])
    
    # Apply preprocessing (imputation, scaling)
    return artifacts["preprocessor"].transform(data_df)

def _get_prediction(features: CreditAppFeatures, model):
    """
    Internal helper function to run the prediction pipeline.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Use our new helper to get the processed data
    transformed_data = _preprocess_data(features)
    
    # Make prediction
    proba = model.predict_proba(transformed_data)
    prob_default = proba[0][1]  # Get the probability for class '1' (Default)
    
    return prob_default

# --- 4. Define API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the CrediSight XAI API. Go to /docs for documentation."}

# --- Prediction Endpoints ---
@app.post("/predict/logistic", response_model=PredictionResponse, tags=["Prediction"])
async def predict_logistic(features: CreditAppFeatures):
    prob_default = _get_prediction(features, artifacts["model_logistic"])
    return PredictionResponse(prob_default=prob_default)

@app.post("/predict/xgb", response_model=PredictionResponse, tags=["Prediction"])
async def predict_xgb(features: CreditAppFeatures):
    prob_default = _get_prediction(features, artifacts["model_xgb"])
    return PredictionResponse(prob_default=prob_default)

# --- 5. NEW Explanation Endpoint ---
@app.post("/explain/xgb", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_xgb(features: CreditAppFeatures):
    """
    Get a full SHAP explanation for a single prediction from the XGBoost model.
    """
    if not artifacts["explainer_xgb"]:
        raise HTTPException(status_code=500, detail="SHAP Explainer not loaded.")

    # 1. Preprocess the data (same as prediction)
    transformed_data = _preprocess_data(features)
    
    # 2. Get SHAP values
    # The explainer returns a complex 'Explanation' object
    shap_explanation = artifacts["explainer_xgb"](transformed_data)
    
    # We want the values for the first sample [0], all features [:], and Class 1 [:, 1]
    # This is because the explainer gives values for both 'Good' (0) and 'Bad' (1)
    shap_values_for_class_1 = shap_explanation.values[0, :, 1]
    
    # The base_value (average prediction for Class 1)
    base_value = shap_explanation.base_values[0, 1]

    # 3. Format the response
    feature_names = artifacts["feature_order"]
    explanations = []
    for name, value in zip(feature_names, shap_values_for_class_1):
        explanations.append(FeatureExplanation(feature=name, shap_value=float(value)))
    
    # Sort explanations by the absolute magnitude of their SHAP value, descending
    # This puts the most "impactful" features at the top.
    explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
    
    return ExplanationResponse(base_value=float(base_value), explanations=explanations)