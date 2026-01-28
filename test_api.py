# test_api.py
import requests

url = "http://127.0.0.1:8000/predict"
data = {"image_path": "C:/Users/veron/Desktop/Calorie-Estimation/bread.jpg"}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()
    print("Prediction:", response.json())
except requests.exceptions.HTTPError as errh:
    print("HTTP Error:", errh, response.json())
except requests.exceptions.RequestException as err:
    print("Request Error:", err)
