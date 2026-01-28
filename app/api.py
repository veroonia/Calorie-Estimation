# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import predict_from_image

app = FastAPI(title="Calorie Estimation API")

class ImageRequest(BaseModel):
    image_path: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: ImageRequest):
    try:
        prediction = predict_from_image(req.image_path)
        return {"prediction": prediction}
    except ValueError as e:
        # Return as 400 Bad Request for errors like missing image or feature mismatch
        raise HTTPException(status_code=400, detail=str(e))