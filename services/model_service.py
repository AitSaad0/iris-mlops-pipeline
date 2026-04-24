import os
import pickle
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

model = None
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. POST /train first."
            )
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model


def train_model(file_path: str):
    global model

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    data = pd.read_csv(file_path)

    if "Id" in data.columns:
        data = data.drop("Id", axis=1)

    data["Species"] = data["Species"].astype("category").cat.codes

    X = data.drop("Species", axis=1).values
    y = data["Species"].values

    clf = RandomForestClassifier()
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    model = clf

    return len(data)