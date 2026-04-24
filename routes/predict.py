from fastapi import APIRouter
import numpy as np

from schemas.iris import IrisFeatures
from services.model_service import get_model, CLASS_NAMES

router = APIRouter()

@router.post("/predict")
def predict(features: IrisFeatures):
    clf = get_model()

    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    prediction = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    return {
        "prediction": int(prediction),
        "class_name": CLASS_NAMES[prediction],
        "confidence": round(float(proba[prediction]), 4)
    }