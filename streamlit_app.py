# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request

# CHANGE THIS LINE AFTER UPLOADING YOUR MODEL
MODEL_URL = "https://drive.google.com/uc?id=1crpqkA6Y1Beu0fgaXEs-pAriltoAey3A&export=download"
MODEL_PATH = "deepfake_model.h5"

st.set_page_config(page_title="Deepfake Detector", page_icon="Detective", layout="centered")

# === DOWNLOAD MODEL ONCE ===
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model (~200MB)... First time only!"):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("Model ready!")
            except:
                st.error("Download failed. Check your internet or link.")
                return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        st.error("Model failed to load.")
        return None

# === PREPROCESS IMAGE ===
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# === UI ===
st.title("Deepfake Detector")
st.markdown("**Upload any image to check if it's AI-generated**")

file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)

    if st.button("Analyze Now", type="primary"):
        with st.spinner("Running AI..."):
            model = load_model()
            if model is None:
                st.stop()
            pred = model.predict(preprocess(img))[0][0]
            conf = pred * 100

            if pred > 0.5:
                st.error(f"**DEEPFAKE** ({conf:.1f}% confidence)")
            else:
                st.success(f"**REAL** ({100-conf:.1f}% confidence)")
                st.balloons()