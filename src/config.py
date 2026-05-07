import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BurnoutRadar API"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths
    MODEL_PATH: str = os.path.join("models", "model.pkl")
    SCALER_PATH: str = os.path.join("models", "scaler.pkl")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
