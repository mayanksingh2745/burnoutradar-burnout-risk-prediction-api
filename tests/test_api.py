import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# Full sample payload matching the 33-feature schema
SAMPLE_PAYLOAD = {
    "age": 35,
    "sex": 1,
    "hours_worked": 45,
    "work_situation_rating": 3,
    "work_control": 3,
    "work_effort": 4,
    "work_intensive": 4,
    "work_skill_level": 4,
    "work_choice_how": 3,
    "work_choice_what": 3,
    "work_decisions": 3,
    "work_demands_combine": 3,
    "work_absorbed": 3,
    "coworker_support": 2,
    "coworker_listen": 2,
    "supervisor_support": 2,
    "supervisor_listen": 2,
    "pride_work": 4,
    "others_respect_work": 4,
    "neg_work_to_family": 2.5,
    "neg_family_to_work": 2.0,
    "pos_work_to_family": 3.0,
    "pos_family_to_work": 3.5,
    "sleep_hours": 6.5,
    "trouble_falling_asleep": 3,
    "self_health": 2,
    "days_unable_work": 2,
    "days_cutback_work": 3,
    "coworker_support_scale": 3.0,
    "supervisor_support_scale": 3.0,
    "family_support": 3.5,
    "friend_support": 3.0,
    "spouse_support": 3.5
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_validation_error():
    # Missing required fields
    response = client.post("/predict", json={"age": 30})
    assert response.status_code == 422

def test_predict_endpoint_mock(monkeypatch):
    from src.api.dependencies import ModelService, get_model_service
    import pandas as pd
    
    class MockModelService(ModelService):
        def __init__(self):
            self.model = True
            
        def predict(self, df: pd.DataFrame):
            return 1
            
        def explain(self, df: pd.DataFrame):
            return 0.5, {"age": 0.1, "hours_worked": 0.4}

    app.dependency_overrides[get_model_service] = MockModelService
    
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["burnout_risk_prediction"] == 1
    assert response.json()["risk_label"] == "High Risk"
    
    # Test explain
    response_explain = client.post("/explain", json=SAMPLE_PAYLOAD)
    assert response_explain.status_code == 200
    assert "shap_values" in response_explain.json()
    assert response_explain.json()["base_value"] == 0.5
    
    # Remove override
    app.dependency_overrides = {}
