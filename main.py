from fastapi import FastAPI, UploadFile, File
import shutil
import os
import random

app = FastAPI()

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.get("/")
def home():
    return {"message": "Backend is working 🚀"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Dummy detection (temporary)
    confidence = random.randint(70, 99)

    result = {
        "filename": file.filename,
        "status": "Deepfake" if confidence > 80 else "Real",
        "confidence": confidence,
        "reason": "Initial model check (demo)"
    }

    return result