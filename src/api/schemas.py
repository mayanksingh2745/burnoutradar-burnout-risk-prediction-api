from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the remote worker")
    hours_worked: float = Field(..., ge=0, description="Hours worked per week")
    stress_level: int = Field(..., ge=1, le=10, description="Self-reported stress level (1-10)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours per night")
    support_level: int = Field(..., ge=1, le=5, description="Perceived support level (1-5)")

class PredictionResponse(BaseModel):
    burnout_risk_prediction: int = Field(..., description="1 if high risk, 0 if low risk")
    risk_label: str = Field(..., description="High Risk or Low Risk")

class ExplanationResponse(BaseModel):
    burnout_risk_prediction: int
    risk_label: str
    shap_values: dict = Field(..., description="SHAP feature importances for this prediction")
    base_value: float = Field(..., description="SHAP base value")
