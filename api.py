from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_proba

app = FastAPI()

class PatientData(BaseModel):
    features: list

@app.get("/",include_in_schema=False)
def read_root():
    return {"message": "AutoML Heart Disease Risk Prediction API"}

@app.post("/predict")
def get_prediction(data: PatientData):
    prob = predict_proba(data.features)
    return {"heart_disease_probability": prob}
