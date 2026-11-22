import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Pharma Tablet QA", page_icon="💊", layout="centered")
st.title("💊 AI-Assisted Pharmaceutical Quality Assurance")
st.write("Upload a tablet image for a demonstration classification.")

MODEL_PATH = Path("tablet_model.keras")
CLASS_PATH = Path("class_names.json")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not CLASS_PATH.exists():
    st.error("class_names.json is missing. Run python train_model.py first.")
    st.stop()

with CLASS_PATH.open("r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

uploaded = st.file_uploader("Upload a tablet image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded tablet image", use_container_width=True)
    if st.button("Analyze image"):
        if not MODEL_PATH.exists():
            st.error("Model not found. Run python train_model.py first.")
            st.stop()
        model = load_model()
        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)[None, ...]
        probabilities = model.predict(array, verbose=0)[0]
        index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[index]
        confidence = float(probabilities[index]) * 100
        st.metric("Demo prediction", predicted_class, f"{confidence:.1f}% confidence")
        if predicted_class == "Good":
            st.success("Demo result: visually classified as Good")
        else:
            st.warning(f"Demo result: {predicted_class} — flag for human review")
        st.caption("Synthetic training images are included only to test the software pipeline. This is not a validated pharmaceutical inspection system.")
