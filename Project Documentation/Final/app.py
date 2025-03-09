import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model (H5 format)
model_path = r"C:\Users\venka\OneDrive\Desktop\tomato_leaf_disease_cnn.h5"
loaded_model = load_model(model_path)  # Load H5 model

# Class labels
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus'
]

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    return img_array

# Function to make predictions
def predict_image(img, model):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100  # Convert to percentage
    return class_names[class_idx], confidence, prediction[0] * 100  # Convert all probs to percentage

# Streamlit UI
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to detect possible diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        predicted_class, confidence, all_predictions = predict_image(img, loaded_model)

        st.success(f"🩺 **Prediction:** {predicted_class} \n 🎯 **Confidence:** {confidence:.2f}%")

        # Show probabilities as percentages
        st.subheader("Class Probabilities:")
        for class_name, prob in zip(class_names, all_predictions):
            st.write(f"**{class_name}:** {prob:.2f}%")
