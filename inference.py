"""
FastAPI server for serving the trained ML model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define input schema
class InputData(BaseModel):
    features: list[float]

# Load model
MODEL_PATH = "results/models/best_model.joblib"
model = joblib.load(MODEL_PATH)

# Initialize app
app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    features_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}
