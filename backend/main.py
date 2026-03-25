from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://pcun89.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once (fast)
with open("embedded_model.json") as f:
    model = json.load(f)

mean = np.array(model["scaler"]["mean"])
scale = np.array(model["scaler"]["scale"])
coef = np.array(model["coefficients"])
intercept = model["intercept"]

# --------------------------
# Data Structure
# --------------------------


class Features(BaseModel):
    values: list[float]


@app.get("/")
def root():
    return {"message": "House Price API running"}


@app.post("/predict")
def predict(data: Features):
    x = np.array(data.values)

    # Normalize
    x_scaled = (x - mean) / scale

    # Dot product
    prediction = float(np.dot(x_scaled, coef) + intercept)

    return {"prediction": prediction}
