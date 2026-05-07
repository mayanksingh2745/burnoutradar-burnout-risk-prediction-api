from src.ml.data_processing import load_and_preprocess_data

def test_data_loading():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    assert len(X_train) > 500
    assert len(X_test) > 100
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Check expected number of features (33)
    assert X_train.shape[1] == 33
    
    # Check key features are present
    expected_features = ['age', 'hours_worked', 'sleep_hours', 'coworker_support']
    for feat in expected_features:
        assert feat in X_train.columns, f"Missing feature: {feat}"
    
    # Check target is binary
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
