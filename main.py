from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from predict import predict_image
from models import ImageUpload, PredictionResponse
from auth import get_current_user
import uvicorn

app = FastAPI(title="Maize Leaf Classification API", description="API for classifying maize leaf images using a CNN model")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: ImageUpload):
    try:
        prediction, confidence = await predict_image(file)
        return PredictionResponse(class_name=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello, {current_user}!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)