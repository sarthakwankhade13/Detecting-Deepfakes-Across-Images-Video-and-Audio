import cv2
import numpy as np
from PIL import Image
from deepfake_system import predict_frames_batch


def detect_fake(file_path: str) -> dict:
    """
    Runs real deepfake detection on an image file.
    Returns a result dict compatible with the DB schema.
    """
    try:
        img = Image.open(file_path).convert("RGB")
        frame = np.array(img)
        results = predict_frames_batch([frame])
        label, score = results[0]

        status = "Deepfake" if label.lower() == "fake" else "Real"
        confidence = round(score * 100, 2)

        return {
            "filename": file_path.split("\\")[-1].split("/")[-1],
            "status": status,
            "confidence": confidence,
            "reason": f"Model: prithivMLmods/Deep-Fake-Detector-Model | Label: {label}"
        }

    except Exception as e:
        return {
            "filename": file_path.split("\\")[-1].split("/")[-1],
            "status": "Error",
            "confidence": 0,
            "reason": str(e)
        }
