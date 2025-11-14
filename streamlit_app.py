import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# Load TFLite model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="deepfake_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Failed to load TFLite model: {e}")
        return None, None, None

interpreter, input_details, output_details = load_model()

# -------------------------------
# Prediction function
# -------------------------------
def predict(image_array):
    """
    image_array: RGB numpy array (H,W,3) resized to model input
    Returns: [prob_real, prob_fake]
    """
    img = np.expand_dims(image_array, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prob_fake = float(interpreter.get_tensor(output_details[0]['index'])[0])
    prob_real = 1 - prob_fake
    return prob_real, prob_fake

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Home", "About"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("Deepfake Detector")
    st.write("Upload an image (JPG/PNG) to check if it's **real or fake**.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and interpreter is not None:
        # Convert uploaded file to OpenCV image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Invalid image file.")
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

            # Preprocess: Resize to 128x128 (as per your TFLite model)
            img_resized = cv2.resize(img_rgb, (128, 128))
            img_norm = img_resized.astype(np.float32) / 255.0

            # Predict
            prob_real, prob_fake = predict(img_norm)

            # Display probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real", f"{prob_real:.1%}")
            with col2:
                st.metric("Fake", f"{prob_fake:.1%}")

            # Detection result
            if prob_fake > 0.5:
                st.error("⚠️ Potential Deepfake Detected!")
            else:
                st.success("✅ Looks Authentic!")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.title("About")
    st.write("""
        **Deepfake Detector** uses AI to predict whether an uploaded image is real or fake.

        - TensorFlow Lite (`.tflite`) model for lightweight deployment
        - Works directly on the web via Streamlit
        - Optimized for fast predictions
    """)
