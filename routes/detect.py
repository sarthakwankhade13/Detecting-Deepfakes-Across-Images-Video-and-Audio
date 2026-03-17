from fastapi import APIRouter, UploadFile, File
import shutil, os
from services.detector import detect_fake
from db import SessionLocal, Result

router = APIRouter()

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detect_fake(file_path)

    # Save to DB
    db = SessionLocal()
    db_result = Result(**result)
    db.add(db_result)
    db.commit()
    db.close()

    return result


@router.get("/results")
def get_results():
    db = SessionLocal()
    data = db.query(Result).all()
    db.close()
    return data