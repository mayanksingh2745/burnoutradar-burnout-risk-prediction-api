from src.ml.data_processing import load_and_preprocess_data

def test_data_loading():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    assert len(X_train) > 1000
    assert len(X_test) > 200
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Check if features are present
    expected_features = ['age', 'hours_worked', 'stress_level', 'sleep_hours', 'support_level']
    assert list(X_train.columns) == expected_features
