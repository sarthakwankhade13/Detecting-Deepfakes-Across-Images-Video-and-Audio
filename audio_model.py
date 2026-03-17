import librosa
import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_NAME = "facebook/wav2vec2-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

CHUNK_DURATION = 3


def detect_audio(path):

    audio, sr = librosa.load(path, sr=16000)

    chunk_size = CHUNK_DURATION * 16000

    scores = []

    for i in range(0, len(audio), chunk_size):

        chunk = audio[i:i + chunk_size]

        if len(chunk) < 16000:
            continue

        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        prob = torch.softmax(logits, dim=1)

        scores.append(float(prob.max()))

    if len(scores) == 0:
        return "ERROR", 0

    final_conf = float(np.mean(scores))

    verdict = "FAKE" if final_conf > 0.5 else "REAL"

    return verdict, final_conf