from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    # Demographics
    age: int = Field(..., ge=18, le=100, description="Age of the worker")
    sex: int = Field(..., ge=1, le=2, description="Sex (1=Male, 2=Female)")
    # Work conditions
    hours_worked: float = Field(..., ge=0, description="Hours worked per week at main job")
    work_situation_rating: int = Field(..., ge=1, le=5, description="Current work situation rating (1=worst, 5=best)")
    work_control: int = Field(..., ge=1, le=5, description="Control over work situation (1=none, 5=a lot)")
    work_effort: int = Field(..., ge=1, le=5, description="Thought/effort put into work (1=none, 5=a lot)")
    work_intensive: int = Field(..., ge=1, le=5, description="Work intensively at job (1=never, 5=all the time)")
    work_skill_level: int = Field(..., ge=1, le=5, description="Work demands high skill level (1=never, 5=all the time)")
    work_choice_how: int = Field(..., ge=1, le=5, description="Choice in how to do work tasks (1=never, 5=all the time)")
    work_choice_what: int = Field(..., ge=1, le=5, description="Choice in what tasks to do (1=never, 5=all the time)")
    work_decisions: int = Field(..., ge=1, le=5, description="Say in work decisions (1=never, 5=all the time)")
    work_demands_combine: int = Field(..., ge=1, le=5, description="Difficulty combining work demands (1=never, 5=all the time)")
    work_absorbed: int = Field(..., ge=1, le=5, description="So involved in work forget time (1=never, 5=all the time)")
    coworker_support: int = Field(..., ge=1, le=4, description="Coworker help/support (1=a lot, 4=not at all)")
    coworker_listen: int = Field(..., ge=1, le=4, description="Coworker listens to work problems (1=a lot, 4=not at all)")
    supervisor_support: int = Field(..., ge=1, le=4, description="Supervisor help/support (1=a lot, 4=not at all)")
    supervisor_listen: int = Field(..., ge=1, le=4, description="Supervisor listens to work problems (1=a lot, 4=not at all)")
    pride_work: int = Field(..., ge=1, le=5, description="Feel pride for work at job (1=never, 5=all the time)")
    others_respect_work: int = Field(..., ge=1, le=5, description="Others respect my work (1=never, 5=all the time)")
    # Work-Family Spillover
    neg_work_to_family: float = Field(..., description="Negative Work to Family Spillover scale score")
    neg_family_to_work: float = Field(..., description="Negative Family to Work Spillover scale score")
    pos_work_to_family: float = Field(..., description="Positive Work to Family Spillover scale score")
    pos_family_to_work: float = Field(..., description="Positive Family to Work Spillover scale score")
    # Sleep
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep on workdays")
    trouble_falling_asleep: int = Field(..., ge=1, le=5, description="Trouble falling asleep (1=never, 5=almost always)")
    # Health
    self_health: int = Field(..., ge=1, le=5, description="Health compared to peers (1=much better, 5=much worse)")
    days_unable_work: int = Field(..., ge=0, le=30, description="Days unable to work due to health (past 30 days)")
    days_cutback_work: int = Field(..., ge=0, le=30, description="Days cut back on work due to health (past 30 days)")
    # Social Support
    coworker_support_scale: float = Field(..., description="Coworker Support scale score")
    supervisor_support_scale: float = Field(..., description="Supervisor Support scale score")
    family_support: float = Field(..., description="Support from Family scale score")
    friend_support: float = Field(..., description="Support from Friends scale score")
    spouse_support: float = Field(..., description="Support from Spouse/Partner scale score")

class PredictionResponse(BaseModel):
    burnout_risk_prediction: int = Field(..., description="1 if high risk, 0 if low risk")
    risk_label: str = Field(..., description="High Risk or Low Risk")

class ExplanationResponse(BaseModel):
    burnout_risk_prediction: int
    risk_label: str
    shap_values: dict = Field(..., description="SHAP feature importances for this prediction")
    base_value: float = Field(..., description="SHAP base value")
