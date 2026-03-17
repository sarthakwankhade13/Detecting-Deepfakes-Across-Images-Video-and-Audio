import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

from deepfake_system import predict_frame, detect_video, detect_audio


st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎭 Deepfake Detection Tester")

media_type = st.selectbox(
    "Select Media Type",
    ["Image", "Video", "Audio"]
)

uploaded_file = st.file_uploader("Upload File")


if uploaded_file is not None:

    # save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    # ================= IMAGE =================
    if media_type == "Image":

        image = Image.open(file_path)
        frame = np.array(image)

        label, score = predict_frame(frame)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("Prediction")

        if label.lower() == "fake":
            st.error(f"FAKE ❌  | Confidence: {score:.3f}")
        else:
            st.success(f"REAL ✅ | Confidence: {score:.3f}")

    # ================= VIDEO =================
    elif media_type == "Video":

        st.video(uploaded_file)

        with st.spinner("Analyzing Video Frames..."):

            verdict, conf = detect_video(file_path)

        st.subheader("Prediction")

        if verdict == "FAKE":
            st.error(f"FAKE ❌ | Confidence: {conf:.3f}")
        elif verdict == "REAL":
            st.success(f"REAL ✅ | Confidence: {conf:.3f}")
        else:
            st.warning("Could not analyze video")

    # ================= AUDIO =================
    elif media_type == "Audio":

        st.audio(uploaded_file)

        with st.spinner("Analyzing Audio..."):

            verdict, conf = detect_audio(file_path)

        st.subheader("Prediction")

        if verdict == "FAKE":
            st.error(f"FAKE ❌ | Confidence: {conf:.3f}")
        elif verdict == "REAL":
            st.success(f"REAL ✅ | Confidence: {conf:.3f}")
        else:
            st.warning("Could not analyze audio")