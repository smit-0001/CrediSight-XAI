from pydantic import BaseModel, Field
from typing import Optional, List # <-- Make sure List is imported

class PredictionResponse(BaseModel):
    """
    The JSON response for a prediction request.
    """
    prob_default: float

class CreditAppFeatures(BaseModel):
    """
    The input features for a prediction request.
    Pydantic will automatically validate these types.
    'Optional[float] = None' means it can accept a number or 'null'.
    """
    ExternalRiskEstimate: Optional[float] = None
    MSinceOldestTradeOpen: Optional[float] = None
    MSinceMostRecentTradeOpen: Optional[float] = None
    AverageMInFile: Optional[float] = None
    NumSatisfactoryTrades: Optional[float] = None
    NumTrades60Ever2DerogPubRec: Optional[float] = None
    NumTrades90Ever2DerogPubRec: Optional[float] = None
    PercentTradesNeverDelq: Optional[float] = None
    MSinceMostRecentDelq: Optional[float] = None
    MaxDelq2PublicRecLast12M: Optional[float] = None
    MaxDelqEver: Optional[float] = None
    NumTotalTrades: Optional[float] = None
    NumTradesOpeninLast12M: Optional[float] = None
    PercentInstallTrades: Optional[float] = None
    MSinceMostRecentInqexcl7days: Optional[float] = None
    NumInqLast6M: Optional[float] = None
    NumInqLast6Mexcl7days: Optional[float] = None
    NetFractionRevolvingBurden: Optional[float] = None
    NetFractionInstallBurden: Optional[float] = None
    NumRevolvingTradesWBalance: Optional[float] = None
    NumInstallTradesWBalance: Optional[float] = None
    NumBank2NatlTradesWHighUtilization: Optional[float] = None
    PercentTradesWBalance: Optional[float] = None

class FeatureExplanation(BaseModel):
    """
    A single feature's contribution to the prediction.
    """
    feature: str
    shap_value: float

class ExplanationResponse(BaseModel):
    """
    The full SHAP explanation response.
    """
    base_value: float
    explanations: List[FeatureExplanation]