# Food Image Calorie Estimation

A Python project that **estimates the calorie content of food from images** using image processing and machine learning. The system enhances images, extracts color and texture features, classifies food types, and provides calorie estimates based on a reference dataset.

---

## Features

- **Image Enhancement:** Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast and visibility of food textures.  
- **Image Restoration:** Uses Non-Local Means (NLM) denoising to reduce noise while preserving important details.  
- **Feature Extraction:**  
  - HSV and LAB color histograms capture color information.  
  - Local Binary Patterns (LBP) capture texture details.  
- **Machine Learning Models:** Trains and evaluates Random Forest, SVM, and Logistic Regression classifiers for food recognition.  
- **Calorie Estimation:** Maps predicted food categories to calories per 100g from a CSV dataset and calculates estimated calories for a given portion.  
- **Evaluation:** Provides accuracy and classification reports for each model.

---

## Tech Stack

- **Python 3**  
- **OpenCV** – Image enhancement and denoising  
- **scikit-image** – Feature extraction (LBP, HOG)  
- **scikit-learn** – Machine learning models and evaluation  
- **NumPy & Pandas** – Data handling  
- **Joblib** – Model persistence  

---

## Dataset

- [Food-11 Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) – Contains 11 food categories including Bread, Dairy, Dessert, Meat, and more.  
- **Calories.csv** – Custom CSV mapping each food category to its calories per 100g.  

---

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/veroonia/Calorie-Estimation.git
cd Calorie-Estimation
