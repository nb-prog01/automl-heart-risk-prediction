from fastapi.testclient import TestClient
from api import app
from model import train_model

client = TestClient(app)

def test_predict_endpoint():
    # Train model first (so model.pkl exists)
    model, _, X_test, _ = train_model()
    
    # Take one row of features
    features = X_test.iloc[0].tolist()
    print(f"Test features: {features}")

    response = client.post("/predict", json={"features": features})
    
    #sample test feature [1.0, 39.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 195.0, 106.0, 70.0, 26.97, 80.0, 77.0]

    assert response.status_code == 200
    data = response.json()
    print("RESPONSE: ",data)
    assert "heart_disease_probability" in data
    assert 0.0 <= data["heart_disease_probability"] <= 1.0
