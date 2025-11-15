import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️")

st.title("🕵️ Deepfake Detection App")
st.write("Upload an image, and the model will predict if it's REAL or FAKE.")

# Load TFLite Model Once
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="deepfake_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image):
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Processing... ⏳")

    img = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    if prediction > 0.5:
        st.error(f"🚨 Deepfake Detected! (Confidence: {prediction:.2f})")
    else:
        st.success(f"✔️ Real Image (Confidence: {1 - prediction:.2f})")
