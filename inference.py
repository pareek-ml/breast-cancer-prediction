"""
FastAPI server for serving the trained ML model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn


# Define input schema
class InputData(BaseModel):
    features: list[float]


# Load model
MODEL_PATH = "results/models/best_model.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
selector = bundle["selector"]
feature_names = bundle["feature_names"]

# Initialize app
app = FastAPI()


@app.post("/predict")
def predict(data: InputData):
    """
    Pass the following features to the array:
    ['radius1', 'texture1', 'perimeter1', 'area1', 'compactness1', 'concavity1',
      'concave_points1', 'radius2', 'perimeter2', 'area2', 'concave_points2', 'radius3',
        'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3',
          'concave_points3', 'symmetry3']
    """
    input_array = np.array(data.features).reshape(1, -1)
    input_scaled = bundle["scaler"].transform(input_array)
    input_selected = bundle["selector"].transform(input_scaled)
    prediction = bundle["model"].predict(input_selected)
    return {"prediction": prediction.tolist()}


@app.get("/ping")
def ping():
    return {"status": "ok"}


# if __name__ == "__main__":
#     uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)


