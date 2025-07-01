import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open(r"D:\crop\Scripts\model.pkl", "rb"))

# Crop label mapping
label_map = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea',
    4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes',
    8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize',
    12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon',
    16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate',
    20: 'rice', 21: 'soybean', 22: 'sugarcane', 23: 'watermelon',
    24: 'wheat'
}

# Streamlit app layout
st.title("Crop Recommendation System 🌱")
st.markdown("Enter the agricultural inputs to predict the suitable crop.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Predict button
if st.button("Predict Crop"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    crop_name = label_map[int(prediction[0])]
    st.success(f"🌾 The recommended crop is: **{crop_name.capitalize()}**")
