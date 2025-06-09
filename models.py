from fastapi import UploadFile
from pydantic import BaseModel

class ImageUpload(BaseModel):
    file: UploadFile

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float