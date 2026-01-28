from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict

app = FastAPI(title="Calorie Estimation API")

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(req: PredictionRequest):
    result = predict(req.features)
    return {"prediction": int(result)}


#Creates the web application
#Holds all routes (@app.get, @app.post, etc.)
#Is what uvicorn runs
#No app → no API → underline.

#import path: app.api:app, uvicorn app.api:app --reload

