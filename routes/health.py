from fastapi import APIRouter, UploadFile, File
from services.data_service import save_file

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    result = save_file(file)
    return {
        "message": "File saved",
        **result
    }