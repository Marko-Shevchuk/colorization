import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from skimage.color import lab2rgb
from PIL import Image

@st.cache_resource
def load_generator():
    return tf.keras.models.load_model("generator.h5", compile=False)

generator = load_generator()


def preprocess_image(image):
    image = np.array(image.convert("L").resize((224, 224)))  # Convert to grayscale
    L = image.astype(np.float32) / 255.0 * 100  # Rescale to [0, 100]
    L = (L / 50.0) - 1.0                        # Normalize to [-1, 1]
    L = np.expand_dims(L, axis=(0, -1))         # Shape: (1, 224, 224, 1)
    return L

def postprocess_output(L_input, ab_output):
    L_input = ((L_input + 1.0) * 50.0).squeeze()  # Denormalize L to [0, 100]
    ab_output = ab_output.squeeze()              # Shape: (224, 224, 2)
    ab_output = ab_output * 128.0                # Denormalize ab to [-128, 127]
    lab = np.zeros((224, 224, 3))
    lab[:, :, 0] = L_input
    lab[:, :, 1:] = ab_output
    rgb = lab2rgb(lab)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

# ---- Streamlit UI ----
st.title(" Greyscale Image Colorization (CGAN)")
st.write("Upload a grayscale image and see it colorized using a trained CGAN model.")

uploaded_file = st.file_uploader("Upload an image (grayscale or color)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    L_input = preprocess_image(image)
    ab_output = generator.predict(L_input)[0]

    colorized_image = postprocess_output(L_input, ab_output)

    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
