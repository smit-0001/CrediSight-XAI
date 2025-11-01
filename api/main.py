from fastapi import FastAPI, HTTPException
from .models import CreditAppFeatures, PredictionResponse, ExplanationResponse, FeatureExplanation
from . import service  # <-- Import our new service.py file

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="CrediSight XAI Engine (Refactored)",
    description="API for credit risk prediction and explanation.",
    version="0.4.0"
)

# --- 2. Load Artifacts On Startup ---
# This just calls the function from our service.py file
@app.on_event("startup")
async def startup_event():
    service.load_artifacts()

# --- 3. Define API Endpoints ---
# These endpoints now just handle routing and call the service logic.

@app.get("/")
async def root():
    return {"message": "Welcome to the CrediSight XAI API. Go to /docs for documentation."}

# --- Prediction Endpoints ---
@app.post("/predict/logistic", response_model=PredictionResponse, tags=["Prediction"])
async def predict_logistic(features: CreditAppFeatures):
    """
    Run a prediction using the **Logistic Regression** model.
    """
    prob_default = service._get_prediction(features, service.artifacts["model_logistic"])
    return PredictionResponse(prob_default=prob_default)

@app.post("/predict/xgb", response_model=PredictionResponse, tags=["Prediction"])
async def predict_xgb(features: CreditAppFeatures):
    """
    Run a prediction using the **XGBoost** model.
    """
    prob_default = service._get_prediction(features, service.artifacts["model_xgb"])
    return PredictionResponse(prob_default=prob_default)

# --- Explanation Endpoint ---
@app.post("/explain/xgb", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_xgb(features: CreditAppFeatures):
    """
    Get a full SHAP explanation for a single prediction from the XGBoost model.
    """
    if not service.artifacts["explainer_xgb"]:
        raise HTTPException(status_code=500, detail="SHAP Explainer not loaded.")

    # 1. Preprocess the data
    transformed_data = service._preprocess_data(features)
    
    # 2. Get SHAP values
    shap_explanation = service.artifacts["explainer_xgb"](transformed_data)
    
    shap_values_for_class_1 = shap_explanation.values[0]
    base_value = shap_explanation.base_values[0]

    # 3. Format the response
    feature_names = service.artifacts["feature_order"]
    explanations = []
    for name, value in zip(feature_names, shap_values_for_class_1):
        explanations.append(FeatureExplanation(feature=name, shap_value=float(value)))
    
    explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
    
    # 4. Generate the summary
    summary_text = service._generate_summary(float(base_value), explanations)

    # 5. Return the full response
    return ExplanationResponse(
        base_value=float(base_value), 
        explanations=explanations,
        summary=summary_text
    )