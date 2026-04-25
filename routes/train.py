import os
from fastapi import APIRouter, HTTPException
from app.services.model_service import train_model
from app.core.config import UPLOAD_DIR

router = APIRouter()

@router.post("/train")
def train(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    rows = train_model(file_path)

    return {
        "message": "Model trained",
        "rows_used": rows
    }