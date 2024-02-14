import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained deep learning model
model = tf.keras.models.load_model('D:/facemask/model.h5')

# Function to preprocess the input image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((250, 250))  # Resize the image to match model input size
    img = np.array(img) / 255.0    # Normalize pixel values
    return img[np.newaxis, ...]     # Add batch dimension

# Streamlit app
st.title('Deep Learning Model Deployment')
st.write('Upload an image to classify')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make predictions
    if st.button('Classify'):
        processed_image = preprocess_image(uploaded_image)
        prediction = model.predict(processed_image)
        st.write('Prediction:', prediction)
