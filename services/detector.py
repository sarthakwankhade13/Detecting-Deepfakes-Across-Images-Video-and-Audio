import random

def detect_fake(file_path):
    confidence = random.randint(70, 99)

    return {
        "filename": file_path.split("/")[-1],
        "status": "Deepfake" if confidence > 80 else "Real",
        "confidence": confidence,
        "reason": "Initial model check (demo)"
    }