import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.ml.data_processing import load_and_preprocess_data

def train_model():
    """
    Train a Voting Ensemble model and track with MLflow.
    """
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Set up MLflow experiment
    mlflow.set_experiment("Burnout_Risk_Prediction")
    
    with mlflow.start_run():
        print("Training Voting Ensemble...")
        
        # Define base models
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Create Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lr', lr_model)
            ],
            voting='soft'
        )
        
        # Train model
        voting_clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = voting_clf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Evaluation -> F1 Score: {f1:.2f}, Accuracy: {acc:.2f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.sklearn.log_model(voting_clf, "voting_ensemble_model")
        
        print("Model saved and tracked via MLflow.")

if __name__ == "__main__":
    # Ensure mlruns directory gets created in the project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    train_model()
