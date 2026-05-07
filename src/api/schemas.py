from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    # Example fields: Adjust these based on your actual dataset features
    designation: int = Field(..., description="Employee's designation level")
    resource_allocation: float = Field(..., description="Amount of resources allocated to employee")
    mental_fatigue_score: float = Field(..., description="Mental fatigue score from 0.0 to 10.0")
    
    class Config:
        json_schema_extra = {
            "example": {
                "designation": 2,
                "resource_allocation": 5.0,
                "mental_fatigue_score": 6.5
            }
        }

class PredictionResponse(BaseModel):
    burnout_risk: float = Field(..., description="Predicted burnout risk score (0 to 1)")
    risk_category: str = Field(..., description="Categorical risk: Low, Medium, or High")
