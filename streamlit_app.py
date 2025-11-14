import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("deepfake_model.h5")
    except:
        st.error("Model not found. Check deepfake_model.h5")
        return None

model = load_model()

st.title("Deepfake Detector")
st.write("Upload an image (JPG/PNG) to check if it's **real or fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Invalid image file.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_resized = cv2.resize(img_rgb, (224, 224))  # Adjust size if your model differs
        img_norm = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        # Predict
        try:
            prediction = model.predict(img_batch)[0][0]
            prob_fake = float(prediction)
            prob_real = 1 - prob_fake

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real", f"{prob_real:.1%}")
            with col2:
                st.metric("Fake", f"{prob_fake:.1%}")

            if prob_fake > 0.5:
                st.error("Potential Deepfake Detected!")
            else:
                st.success("Looks Authentic!")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
