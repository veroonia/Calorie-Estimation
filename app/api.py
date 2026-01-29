from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import os
from app.model import predict_from_image

app = FastAPI(title="Calorie Estimation API")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict-image")
def predict_image(file: UploadFile = File(...)):
    print("API: request received")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        print("API: saving file")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("API: file saved, calling model")
        prediction = predict_from_image(temp_path)
        print("API: model returned")

        return {"prediction": prediction}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)