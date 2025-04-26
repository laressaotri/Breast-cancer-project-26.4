import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Page settings ---
st.set_page_config(page_title="Hope in Pixels", layout="wide", page_icon="🎗")

# --- Custom CSS for background color and styling ---
st.markdown("""
    <style>
    body {
        background-color: #FFF0F5;
    }
    .main {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet_model.h5")

model = load_model()

# --- Preprocessing function ---
def preprocess_image(image_pil):
    image = np.array(image_pil.convert("L"))  # Grayscale
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))
    return np.expand_dims(image, axis=0), image

# --- Grad-CAM generation ---
def generate_gradcam(model, input_tensor, original_img, last_conv_layer_name='conv5_block3_out', alpha=0.4):
    base_model = model.layers[0]  # ResNet inside Sequential
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        class_output = predictions[:, 0]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img.astype('uint8'), 1 - alpha, heatmap_color, alpha, 0)

    return overlay

# --- Logo ---
st.image("logo.png", width=250)

# --- Title and description ---
st.title("Hope in Pixels: Breast Cancer CT Classifier")
st.write("Upload a **grayscale CT scan** to get a prediction and a sharp Grad-CAM heatmap overlay.")

# --- Upload and Predict ---
uploaded_file = st.file_uploader("🗆️ Upload a CT scan image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    input_tensor, display_img = preprocess_image(image)
    prediction = model.predict(input_tensor)[0][0]
    label = "🢨🎗️ Cancer" if prediction > 0.5 else "✅ Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")

    overlay_img = generate_gradcam(model, input_tensor, display_img)

    # Display side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.image(display_img, caption="Original CT Scan", use_container_width=True)

    with col2:
        st.image(overlay_img, caption=" Grad-CAM Overlay", use_container_width=True)

else:
    st.info("Please upload a CT scan image to get started.")

# --- Footer credit ---
st.markdown("""
---
<div style='text-align: center; color: #5C2C36;'>
Created with care by **L.O, R.H, Z.E**
</div>
""", unsafe_allow_html=True)
