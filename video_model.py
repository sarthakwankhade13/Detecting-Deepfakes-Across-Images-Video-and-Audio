import cv2
import numpy as np
from image_model import predict_frames_batch

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

    # remaining frames
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