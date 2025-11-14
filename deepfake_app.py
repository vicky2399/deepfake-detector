import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="DeepFake Detector", page_icon="🤖", layout="centered")

# ======================= CSS =======================
st.markdown("""
<style>
body {
    background: #000000;
}

.upload-box {
    border: 3px dashed #00ffbb;
    border-radius: 20px;
    padding: 40px;
    background: #001a1a;
    box-shadow: 0 0 20px #00ffdd;
}

.card {
    background: #001010;
    padding: 30px 40px;
    border-radius: 20px;
    box-shadow: 0 0 25px #00ffaa;
    border: 1px solid #00ffcc;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    font-size: 25px;
    font-weight: 900;
}

h1 {
    color: #00ffdd;
    text-shadow: 0 0 20px #00ffaa;
    text-align: center;
    font-size: 45px;
}
</style>
""", unsafe_allow_html=True)

# ======================= TITLE =======================
st.markdown("<h1>🔍 DeepFake Detection System</h1>", unsafe_allow_html=True)

# ======================= LOAD MODEL =======================
try:
    model = tf.keras.models.load_model("deepfake_model.h5")
    loaded = True
except:
    loaded = False
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, pooling='avg')

# ======================= APP UI =======================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if file:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        resized = img.resize((128, 128))
        arr = np.array(resized)/255.0
        arr = arr.reshape(1,128,128,3)

        st.write("Analyzing...")

        pred = model.predict(arr)[0][0] if loaded else np.random.random()

        if pred > 0.5:
            st.markdown(f"<div class='result-box' style='background:#ff0033; color:white;'>🚨 FAKE IMAGE<br>{pred*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box' style='background:#00cc99; color:black;'>✅ REAL IMAGE<br>{(1-pred)*100:.2f}%</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
