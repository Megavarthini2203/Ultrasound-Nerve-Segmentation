import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Page configuration
st.set_page_config(
    page_title="Ultrasound Nerve Segmentation",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Path to your local background image
image_path = r"C:\Users\megav\Downloads\pexels-pixabay-256262.jpg"
image_base64 = image_to_base64(image_path)
# Background CSS using the base64 image string
page_bg_img = f'''
<style>
body {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    height: 100vh;  # Ensure the image covers the entire page
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("Ultrasound Nerve Segmentation")

# Upload section
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

# Placeholder for model predictions
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Ultrasound Image", use_column_width=True)

    # Load the image
    img = Image.open(uploaded_file)

    # Convert to grayscale
    img_gray = img.convert("L")  # 'L' mode is for grayscale
    img_array = np.array(img_gray)

    # Ensure the image has 3 dimensions (height, width, 1) for grayscale
    img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension

    # Load your model with no compilation initially
    @st.cache_resource  # Cache the loaded model for performance
    def load_model():
        model = tf.keras.models.load_model(
            r"C:\Users\megav\Downloads\weights.h5", 
            compile=False
        )
        # Manually compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Adjust learning rate as needed
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    model = load_model()

    # Preprocess the image
    img_resized = tf.image.resize(img_array, [96, 96])  # Resize to model's expected input
    img_preprocessed = np.expand_dims(img_resized / 255.0, axis=0)  # Add batch dimension

    # Perform segmentation
    st.write("Performing segmentation...")
    prediction = model.predict(img_preprocessed)
    segmented_image = np.squeeze(prediction)  # Remove batch dimension

    # Show the segmented image
    st.image(segmented_image, caption="Segmented Image", use_column_width=True)
 # To print the first 100 characters of the base64 string for verification
