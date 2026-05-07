import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_validation_error():
    # Missing required fields
    response = client.post("/predict", json={"age": 30})
    assert response.status_code == 422

def test_predict_endpoint_mock(monkeypatch):
    # Mock the dependency to not require an actual trained model
    from src.api.dependencies import ModelService, get_model_service
    import pandas as pd
    
    class MockModelService(ModelService):
        def __init__(self):
            self.model = True # Just so it's not None
            
        def predict(self, df: pd.DataFrame):
            return 1 # Mock high risk
            
        def explain(self, df: pd.DataFrame):
            return 0.5, {"age": 0.1, "stress_level": 0.4}

    app.dependency_overrides[get_model_service] = MockModelService
    
    payload = {
        "age": 35,
        "hours_worked": 50,
        "stress_level": 8,
        "sleep_hours": 5,
        "support_level": 2
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["burnout_risk_prediction"] == 1
    assert response.json()["risk_label"] == "High Risk"
    
    # Test explain
    response_explain = client.post("/explain", json=payload)
    assert response_explain.status_code == 200
    assert "shap_values" in response_explain.json()
    assert response_explain.json()["base_value"] == 0.5
    
    # Remove override
    app.dependency_overrides = {}
