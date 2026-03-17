import cv2
import torch
import numpy as np
from PIL import Image

# ================= IMAGE MODEL =================

_image_detector = None

def get_image_detector():
    global _image_detector
    if _image_detector is None:
        from transformers import pipeline
        device_pipe = 0 if torch.cuda.is_available() else -1
        _image_detector = pipeline(
            "image-classification",
            model="prithivMLmods/Deep-Fake-Detector-Model",
            device=device_pipe
        )
    return _image_detector


def predict_frame(frame):
    """Single frame prediction"""
    results = predict_frames_batch([frame])
    return results[0]  # (label, score)


def predict_frames_batch(frames):
    detector = get_image_detector()
    images = [Image.fromarray(f) for f in frames]
    outputs = detector(images)
    results = []
    for o in outputs:
        label = o[0]["label"]
        score = float(o[0]["score"])
        results.append((label, score))
    return results


# ================= VIDEO MODEL =================

FRAME_SAMPLE_RATE = 8
BATCH_SIZE = 16


def detect_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    fake_votes = 0
    real_votes = 0
    confidences = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SAMPLE_RATE != 0:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if len(frames) == BATCH_SIZE:
            for label, score in predict_frames_batch(frames):
                confidences.append(score)
                if label.lower() == "fake":
                    fake_votes += 1
                else:
                    real_votes += 1
            frames = []

    if frames:
        for label, score in predict_frames_batch(frames):
            confidences.append(score)
            if label.lower() == "fake":
                fake_votes += 1
            else:
                real_votes += 1

    cap.release()

    if not confidences:
        return "ERROR", 0

    final_conf = float(np.mean(confidences))
    verdict = "FAKE" if fake_votes > real_votes else "REAL"
    return verdict, final_conf


# ================= AUDIO MODEL =================

_audio_model = None
_audio_processor = None

def get_audio_model():
    global _audio_model, _audio_processor
    if _audio_model is None:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base").to(device)
        _audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    return _audio_model, _audio_processor


CHUNK_DURATION = 3


def detect_audio(path):
    import librosa
    audio_model, audio_processor = get_audio_model()
    device = next(audio_model.parameters()).device

    audio, sr = librosa.load(path, sr=16000)
    chunk_size = CHUNK_DURATION * 16000
    scores = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < 16000:
            continue
        inputs = audio_processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = audio_model(**inputs).logits
        prob = torch.softmax(logits, dim=1)
        scores.append(float(prob.max()))

    if not scores:
        return "ERROR", 0

    final_conf = float(np.mean(scores))
    verdict = "FAKE" if final_conf > 0.5 else "REAL"
    return verdict, final_conf
