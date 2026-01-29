# app/model.py
import cv2
import numpy as np
from joblib import load
from skimage.feature import local_binary_pattern
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALORIES_PATH = os.path.join(BASE_DIR, "Calories.csv")

calories_df = pd.read_csv(CALORIES_PATH)

label_map = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit"
}

# Load your existing RandomForest model
MODEL_PATH = os.path.join("models", "rf_model.joblib")
rf_model = load(MODEL_PATH)

def enhance_clahe(bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE enhancement to improve contrast."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def lbp_features(img):
    """Extract LBP texture features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return hist

def extract_features(img):
    """
    Extract color and texture features from image:
    - HSV histogram (512 features: 8x8x8)
    - LAB histogram (512 features: 8x8x8)
    - LBP texture histogram (9 features)
    Total: 1033 features
    """
    # Resize and enhance
    img = cv2.resize(img, (128, 128))
    img = enhance_clahe(img)

    # HSV histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # LAB histogram
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    lab_hist = cv2.normalize(lab_hist, lab_hist).flatten()

    # LBP texture histogram
    lbp_hist = lbp_features(img)

    # Combine features
    features = np.concatenate([hsv_hist, lab_hist, lbp_hist])
    return features.reshape(1, -1)

def predict_from_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    features = extract_features(img)

    if features.shape[1] != rf_model.n_features_in_:
        raise ValueError(
            f"Expected {rf_model.n_features_in_} features, got {features.shape[1]}"
        )

    # Predict class
    pred_label = int(rf_model.predict(features)[0])
    food_category = label_map[pred_label]

    # Lookup calories per 100g
    calories_per_100g = calories_df.loc[
        calories_df["food_category"] == food_category,
        "calories_per_100g"
    ].values[0]

    return {
        "food_category": food_category,
        "calories_per_100g": int(calories_per_100g)
    }
