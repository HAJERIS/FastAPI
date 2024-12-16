from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import torch

app = FastAPI()

# Load your YOLOv8 model
model = YOLO("best.pt")  # Adjust path if needed

@app.get("/")
def home():
    return {"message": "YOLOv8 Plant Diagnosis API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Make prediction
    results = model.predict(image, conf=0.5)  # Adjust confidence if needed

    # Process results
    detections = []
    for result in results[0].boxes:
        class_id = int(result.cls)
        confidence = float(result.conf)
        bbox = result.xyxy.tolist()  # [x_min, y_min, x_max, y_max]
        detections.append({
            "class_id": class_id,
            "confidence": confidence,
            "bbox": bbox
        })

    return {"detections": detections}