import os
import glob
import mlflow.sklearn
import shap
from fastapi import HTTPException
import pandas as pd

class ModelService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.load_model()

    def load_model(self):
        """Loads the latest model from mlruns directory if available."""
        mlruns_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'mlruns')
        if not os.path.exists(mlruns_dir):
            print("Warning: mlruns directory not found. Model is not loaded.")
            return
            
        try:
            # Find the most recently created model in mlruns
            model_paths = glob.glob(os.path.join(mlruns_dir, "*", "*", "artifacts", "voting_ensemble_model"))
            if not model_paths:
                print("Warning: No trained model found in mlruns.")
                return
                
            # Sort by modification time to get the latest
            latest_model_path = max(model_paths, key=os.path.getmtime)
            
            # Load the model
            self.model = mlflow.sklearn.load_model(f"file://{latest_model_path}")
            
            # Note: For SHAP on ensembles, we might need a KernelExplainer or just use an underlying model.
            # Using a generic Explainer for demonstration. It expects the prediction function.
            # SHAP initialization can be deferred or done here if training data background is available.
            
            print(f"Model successfully loaded from {latest_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, features_df: pd.DataFrame):
        if not self.model:
            raise HTTPException(status_code=503, detail="Model is not loaded or not trained yet.")
        return int(self.model.predict(features_df)[0])
        
    def explain(self, features_df: pd.DataFrame):
        if not self.model:
            raise HTTPException(status_code=503, detail="Model is not loaded or not trained yet.")
        
        # Creating a SHAP Explainer dynamically
        # Since it's a voting classifier, we use KernelExplainer. It's slow but works for black-box models.
        # Ideally, we should pass a background dataset. For this mockup, we just use the instance itself as background.
        try:
            # Suppress output from KernelExplainer
            explainer = shap.KernelExplainer(self.model.predict, features_df)
            shap_values = explainer.shap_values(features_df)
            
            # format the response
            features_dict = features_df.iloc[0].to_dict()
            explanation = {k: float(v) for k, v in zip(features_dict.keys(), shap_values[0])}
            return float(explainer.expected_value), explanation
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {str(e)}")

# Create a singleton instance
model_service = ModelService()

def get_model_service() -> ModelService:
    return model_service
