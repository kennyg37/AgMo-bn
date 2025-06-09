import numpy as np
import cv2
from tensorflow.keras.models import load_model
from config import MODEL_PATH, CLASS_NAMES
from models import ImageUpload
import io
from fastapi import HTTPException

async def predict_image(file: ImageUpload):
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise HTTPException(status_code=400, detail=f"Invalid file format. Only {', '.join(allowed_extensions)} are supported.")
    
    # Read and preprocess image
    try:
        contents = await file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image. Ensure the file is a valid image.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Make prediction
    try:
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class_idx])
        prediction = CLASS_NAMES[predicted_class_idx]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return prediction, confidence