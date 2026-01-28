import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_features_from_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    return hist
