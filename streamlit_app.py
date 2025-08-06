import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("saved_models/fruit_freshness_model.h5")  # Keep same filename

# Define class names (Update as per your model's classes)
class_names = [
    'Fresh Apple',     # class 0
    'Fresh Banana',    # class 1
    'Fresh Orange',    # class 2
    'Rotten Apple',    # class 3
    'Rotten Banana',   # class 4
    'Rotten Orange'    # class 5
]


# App title and uploader
st.title("🍎 Fruit Freshness Detection")
st.write("Upload an image of a fruit to check if it's fresh or rotten.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    st.markdown(f"### 🧠 Prediction: **{class_names[class_idx]}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
