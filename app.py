import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from skimage.color import lab2rgb
from PIL import Image

@st.cache_resource
def load_generator():
    return ort.InferenceSession("generator.onnx")

generator = load_generator()

def preprocess_image(image):
    image = np.array(image.convert("L").resize((224, 224)))  
    L = image.astype(np.float32) / 255.0 * 100               
    L = (L / 50.0) - 1.0                                     
    L = np.expand_dims(L, axis=(0, -1))                     
    return L

def run_generator(L_input):
    input_name = generator.get_inputs()[0].name
    output = generator.run(None, {input_name: L_input.astype(np.float32)})
    return output[0]  # shape: (1, 224, 224, 2)

def postprocess_output(L_input, ab_output):
    L_input = ((L_input + 1.0) * 50.0).squeeze()             
    ab_output = ab_output.squeeze() * 128.0                 
    lab = np.zeros((224, 224, 3), dtype=np.float32)
    lab[:, :, 0] = L_input
    lab[:, :, 1:] = ab_output
    rgb = lab2rgb(lab)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


st.title("Greyscale Image Colorization (CGAN)")
st.write("Upload a grayscale image and see it colorized using a trained CGAN model (ONNX).")

uploaded_file = st.file_uploader("Upload an image (grayscale or color)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    L_input = preprocess_image(image)
    ab_output = run_generator(L_input)

    colorized_image = postprocess_output(L_input, ab_output)
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
