# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
import urllib.request
import cv2
import tempfile
import tflite_runtime.interpreter as tflite

<<<<<<< HEAD
# YOUR TFLITE MODEL LINK
MODEL_URL = "https://drive.google.com/uc?id=1hrPLUbcWbolL8lfVgPpVAoXJ-IvU9RF6&export=download"
MODEL_PATH = "deepfake_model.tflite"
=======
# CHANGE THIS LINE AFTER UPLOADING YOUR MODEL
MODEL_URL = "https://drive.google.com/uc?id=1hrPLUbcWbolL8lfVgPpVAoXJ-IvU9RF6&export=download"
MODEL_PATH = "deepfake_model.h5"
>>>>>>> 7c6b4d4 (Add working tflite model link)

st.set_page_config(page_title="Deepfake AI", page_icon="Detective", layout="centered")

# === DOWNLOAD MODEL ===
@st.cache_resource
def get_interpreter():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# === PREPROCESS ===
def prep(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# === UI ===
st.title("Deepfake AI")
st.write("**Upload image to detect deepfake**")

interpreter = get_interpreter()
input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

def predict(arr):
    interpreter.set_tensor(input_idx, arr)
    interpreter.invoke()
    return interpreter.get_tensor(output_idx)[0][0]

# === UPLOAD IMAGE ===
file = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])
if file:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    if st.button("Check Now"):
        with st.spinner("Analyzing..."):
            p = predict(prep(img))
            if p > 0.5:
                st.error(f"**DEEPFAKE** ({p*100:.1f}%)")
            else:
                st.success(f"**REAL** ({(1-p)*100:.1f}%)")
                st.balloons()
