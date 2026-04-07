import streamlit as st
import cv2
import numpy as np
from detector import StoneCrackDetector
from PIL import Image

st.title("Stone Crack Detection App 🪨")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    detector = StoneCrackDetector()

    cracks, pre = detector.detect_cracks(image)
    metrics = detector.calculate_crack_metrics(cracks)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.subheader("Detected Cracks")
    st.image(cracks, clamp=True)

    st.subheader("Metrics")
    st.write(metrics)
