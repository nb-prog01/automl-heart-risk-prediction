from model import train_model, predict_proba

def test_model_training():
    model, acc, X_test, y_test = train_model()
    assert model is not None
    assert acc > 0.5   # sanity check

def test_prediction_range():
    model, _, X_test, _ = train_model()
    features = X_test.iloc[0].tolist()
    prob = predict_proba(features)
    assert 0.0 <= prob <= 1.0
