from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import os
from app.model import predict_from_image
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

templates = Jinja2Templates(directory="app/templates")

app = FastAPI(title="Calorie Estimation API")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predict_from_image(temp_path)
        return prediction

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
