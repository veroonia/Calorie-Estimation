from joblib import load

MODEL_PATH = "models/rf_model.joblib"

model = load(MODEL_PATH)
def predict(features):
    return model.predict([features])[0]