import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

from image_model import predict_frame
from video_model import detect_video
from audio_model import detect_audio


st.title("Deepfake Detection Tester")

media_type = st.selectbox(
    "Select Media Type",
    ["Image", "Video", "Audio"]
)

uploaded_file = st.file_uploader("Upload File")

if uploaded_file is not None:

    # save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if media_type == "Image":

        image = Image.open(tfile.name)
        frame = np.array(image)

        result = predict_frame(frame)[0]

        st.image(image)
        st.write("Result:", result["label"])
        st.write("Confidence:", result["score"])

    elif media_type == "Video":

        st.video(uploaded_file)

        conf = detect_video(tfile.name)

        st.write("Video Confidence:", conf)

    elif media_type == "Audio":

        st.audio(uploaded_file)

        result = detect_audio(tfile.name)

        st.write(result)