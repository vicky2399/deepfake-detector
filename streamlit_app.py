import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Load TFLite model (cached)
# -----------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="deepfake_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_tflite_model()

# Prediction function
def predict(image_array):
    # Expand dims and normalize
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return float(pred)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Deepfake Detector")
st.write("Upload an image (JPG/PNG) to check if it's real or fake.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Resize to model input size (128x128)
        img_resized = cv2.resize(img_rgb, (128, 128))
        # Normalize already handled in predict()

        # Predict
        prob_fake = predict(img_resized)
        prob_real = 1 - prob_fake

        # Show metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Real", f"{prob_real:.1%}")
        with col2:
            st.metric("Fake", f"{prob_fake:.1%}")

        if prob_fake > 0.5:
            st.error("⚠️ Potential Deepfake Detected!")
        else:
            st.success("✅ Looks Authentic!")
