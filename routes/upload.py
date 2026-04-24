from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": os.path.exists("model.pkl")
    }