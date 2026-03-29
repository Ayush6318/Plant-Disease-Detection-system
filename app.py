import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# Load model
model = load_model("model.h5",compile=False)


# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping
labels = {v: k for k, v in class_indices.items()}

st.title("🌱 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"### 🌿 Prediction: {labels[class_index]}")
    st.write(f"### 🔍 Confidence: {confidence:.2f}")