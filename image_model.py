from transformers import pipeline
from PIL import Image
import torch

device = 0 if torch.cuda.is_available() else -1

image_detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-Model",
    device=device
)

def predict_frame(frame_rgb):

    img = Image.fromarray(frame_rgb)

    result = image_detector(img)[0]

    label = result["label"]
    score = float(result["score"])

    return label, score


def predict_frames_batch(frames):

    images = [Image.fromarray(f) for f in frames]

    outputs = image_detector(images)

    results = []

    for o in outputs:
        label = o[0]["label"]
        score = float(o[0]["score"])
        results.append((label, score))

    return results