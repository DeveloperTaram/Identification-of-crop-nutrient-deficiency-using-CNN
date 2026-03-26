import os
import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


# Page Config

st.set_page_config(page_title="Rice Deficiency Detector 🌾", layout="centered")

st.title("🌾 Rice Plant Nutrient Deficiency Classification")
st.write("Upload a rice leaf image to detect nutrient deficiency.")


# Load Model (SAFE)

@st.cache_resource
def load_model():
    try:
        working_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(working_dir, "trained_model", "resnet50_rice_plant_final.h5")

        original_dense_init = tf.keras.layers.Dense.__init__

        def patched_dense_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            original_dense_init(self, *args, **kwargs)

        tf.keras.layers.Dense.__init__ = patched_dense_init

        model = tf.keras.models.load_model(model_path, compile=False)


        tf.keras.layers.Dense.__init__ = original_dense_init

        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Load Class Names

@st.cache_data
def load_class_names():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(working_dir, "class_names.pkl"), "rb") as f:
        return pickle.load(f)

class_names = load_class_names()


# Image Preprocessing

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prediction Function

def predict_image_class(model, image, class_names):
    img = load_and_preprocess_image(image)
    predictions = model.predict(img)

    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence, predictions

# File Upload

uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🔍 Classify"):
            if model is None:
                st.error("Model not loaded properly.")
            else:
                prediction, confidence, probs = predict_image_class(model, uploaded_image, class_names)

                st.success(f"🌿 Prediction: {prediction}")
                st.info(f"Confidence: {confidence:.2f}%")


                # Show probabilities

                st.subheader("📊 All Class Probabilities")
                for i, prob in enumerate(probs[0]):
                    st.write(f"{class_names[i]}: {prob*100:.2f}%")