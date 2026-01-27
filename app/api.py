from fastapi import FastAPI

app = FastAPI(title="Calorie Estimation API")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/predict")
def predict():
    return {"calories": 123}



#Creates the web application
#Holds all routes (@app.get, @app.post, etc.)
#Is what uvicorn runs
#No app → no API → underline.

#import path: app.api:app, uvicorn app.api:app --reload

