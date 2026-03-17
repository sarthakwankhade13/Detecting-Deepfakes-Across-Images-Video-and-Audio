import cv2
import librosa
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ================= IMAGE MODELS (ENSEMBLE) =================

device_pipe = 0 if torch.cuda.is_available() else -1

model1 = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
    device=device_pipe
)

model2 = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-Model",
    device=device_pipe
)


def preprocess_artifact(img):
    """
    Simple artifact emphasis preprocessing
    """
    img = np.array(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return Image.fromarray(img)


def predict_frames_batch(frames):

    images = [Image.fromarray(f) for f in frames]
    images_art = [preprocess_artifact(img) for img in images]

    # RUN MODELS SIMULTANEOUSLY
    out1 = model1(images)
    out2 = model2(images)
    out3 = model2(images_art)

    results = []

    for i in range(len(images)):

        score1 = float(out1[i][0]["score"])
        score2 = float(out2[i][0]["score"])
        score3 = float(out3[i][0]["score"])

        # weighted fusion
        final_score = 0.4 * score1 + 0.3 * score2 + 0.3 * score3

        label = "FAKE" if final_score > 0.5 else "REAL"

        results.append((label, final_score))

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

            results = predict_frames_batch(frames)

            for label, score in results:

                confidences.append(score)

                if label.lower() == "fake":
                    fake_votes += 1
                else:
                    real_votes += 1

            frames = []

    if len(frames) > 0:

        results = predict_frames_batch(frames)

        for label, score in results:

            confidences.append(score)

            if label.lower() == "fake":
                fake_votes += 1
            else:
                real_votes += 1

    cap.release()

    if len(confidences) == 0:
        return "ERROR", 0

    final_conf = float(np.mean(confidences))

    verdict = "FAKE" if fake_votes > real_votes else "REAL"

    return verdict, final_conf


# ================= AUDIO MODEL =================

MODEL_NAME = "facebook/wav2vec2-base"

device_audio = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(device_audio)
audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

CHUNK_DURATION = 3


def detect_audio(path):

    audio, sr = librosa.load(path, sr=16000)

    chunk_size = CHUNK_DURATION * 16000

    scores = []

    for i in range(0, len(audio), chunk_size):

        chunk = audio[i:i + chunk_size]

        if len(chunk) < 16000:
            continue

        inputs = audio_processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device_audio) for k, v in inputs.items()}

        with torch.no_grad():
            logits = audio_model(**inputs).logits

        prob = torch.softmax(logits, dim=1)

        scores.append(float(prob.max()))

    if len(scores) == 0:
        return "ERROR", 0

    final_conf = float(np.mean(scores))

    verdict = "FAKE" if final_conf > 0.5 else "REAL"

    return verdict, final_conf