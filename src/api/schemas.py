from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the remote worker")
    hours_worked: float = Field(..., ge=0, description="Hours worked per week")
    work_situation_rating: int = Field(..., ge=1, le=5, description="Self-rated current work situation (1=worst, 5=best)")
    work_control: int = Field(..., ge=1, le=5, description="Perceived control over work situation (1=none, 5=a lot)")
    work_intensive: int = Field(..., ge=1, le=5, description="How intensively you work at your job (1=never, 5=all the time)")
    work_demands_combine: int = Field(..., ge=1, le=5, description="Difficulty combining work demands (1=never, 5=all the time)")
    coworker_support: int = Field(..., ge=1, le=4, description="Coworker help/support level (1=a lot, 4=not at all)")
    supervisor_support: int = Field(..., ge=1, le=4, description="Supervisor help/support level (1=a lot, 4=not at all)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours on workdays")
    trouble_falling_asleep: int = Field(..., ge=1, le=5, description="How often trouble falling asleep (1=never, 5=almost always)")
    self_health: int = Field(..., ge=1, le=5, description="Self-evaluated health compared to peers (1=a lot better, 5=a lot worse)")
    days_unable_work: int = Field(..., ge=0, le=30, description="Days unable to work due to health in past 30 days")
    days_cutback_work: int = Field(..., ge=0, le=30, description="Days cut back on work due to health in past 30 days")

class PredictionResponse(BaseModel):
    burnout_risk_prediction: int = Field(..., description="1 if high risk, 0 if low risk")
    risk_label: str = Field(..., description="High Risk or Low Risk")

class ExplanationResponse(BaseModel):
    burnout_risk_prediction: int
    risk_label: str
    shap_values: dict = Field(..., description="SHAP feature importances for this prediction")
    base_value: float = Field(..., description="SHAP base value")
