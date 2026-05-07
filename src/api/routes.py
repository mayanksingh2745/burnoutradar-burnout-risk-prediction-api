import pandas as pd
from fastapi import APIRouter, Depends
from src.api.schemas import PredictionRequest, PredictionResponse, ExplanationResponse
from src.api.dependencies import get_model_service, ModelService

router = APIRouter()

@router.get("/health", tags=["System"])
def health_check(service: ModelService = Depends(get_model_service)):
    status = "ok" if service.model else "model_missing"
    return {"status": status, "message": "BurnoutRadar API is running"}

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_burnout(request: PredictionRequest, service: ModelService = Depends(get_model_service)):
    """
    Takes in remote worker features and returns a burnout risk prediction.
    """
    # Convert request to dataframe
    df = pd.DataFrame([request.model_dump()])
    
    # Predict
    prediction = service.predict(df)
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    
    return PredictionResponse(
        burnout_risk_prediction=prediction,
        risk_label=risk_label
    )

@router.post("/explain", response_model=ExplanationResponse, tags=["Prediction"])
def explain_prediction(request: PredictionRequest, service: ModelService = Depends(get_model_service)):
    """
    Returns SHAP feature importances to explain the prediction for the given features.
    """
    df = pd.DataFrame([request.model_dump()])
    
    # Get prediction and explanation
    prediction = service.predict(df)
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    base_value, shap_values = service.explain(df)
    
    return ExplanationResponse(
        burnout_risk_prediction=prediction,
        risk_label=risk_label,
        shap_values=shap_values,
        base_value=base_value
    )
