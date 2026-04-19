from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List, Dict
import pandas as pd
import os


UPLOAD_DIR = "data"

# Create folder if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Iris MLOps API")

# Load model at startup
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "prediction": int(prediction),
        "class_name": CLASS_NAMES[prediction],
        "confidence": round(float(proba[prediction]), 4)
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file to disk
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Read it with pandas
    df = pd.read_csv(file_path)

    return {
        "message": "File saved successfully",
        "filename": file.filename,
        "columns": df.columns.tolist()
    }