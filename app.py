import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Fruit & Veg Identifier 🍎🥦", page_icon="logo.png", layout="centered")

# Language Selector
languages = {
    "en": "English",
    "te": "తెలుగు"
}

language = st.selectbox("🌐 Choose Language", options=list(languages.keys()), format_func=lambda x: languages[x])

# Translations
translations = {
    "en": {
        "title": "Fruit & Veggie Identifier 🍎🥦",
        "upload": "📤 Upload an Image for Classification",
        "processing": "Processing image...",
        "prediction": "Prediction",
        "accuracy": "Accuracy"
    },
    "te": {
        "title": "పండ్లు & కూరగాయల గుర్తింపు 🍎🥦",
        "upload": "📤 వర్గీకరణ కోసం ఒక చిత్రం అప్‌లోడ్ చేయండి",
        "processing": "చిత్రాన్ని ప్రాసెస్ చేస్తోంది...",
        "prediction": "అంచనా",
        "accuracy": "ఖచ్చితత"
    }
}

# Theme toggle
theme = st.toggle("🌗 Toggle Dark Mode")

# Custom CSS for animations and styling
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {{
            background-color: {'#2E2E2E' if theme else '#FFFFFF'};
            color: {'#FFFFFF' if theme else '#000000'};
            font-family: 'Poppins', sans-serif;
            transition: background-color 0.5s ease, color 0.5s ease;
        }}
        .title {{
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: {'#90EE90' if theme else '#4CAF50'};
            transition: color 0.5s ease;
        }}
        .upload-box {{
            border: 2px dashed {'#90EE90' if theme else '#4CAF50'};
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            margin-top: 20px;
            transition: border 0.5s ease;
        }}
        .prediction, .accuracy {{
            font-size: 2rem;
            font-weight: bold;
            opacity: 0;
            animation: fadeIn 1.5s ease-in-out forwards;
        }}
        .accuracy {{
            font-size: 1.5rem;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .zoom-container img {{
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .zoom-container img:hover {{
            transform: scale(1.1);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }}

        /* Gradient Progress Bar */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .title {{ font-size: 2rem; }}
            .prediction {{ font-size: 1.5rem; }}
            .accuracy {{ font-size: 1.2rem; }}
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="title">{translations[language]["title"]}</div>', unsafe_allow_html=True)

# Cache the model to prevent reloading
@st.cache_resource
def load_cached_model():
    model_path = 'Image_classify.keras' if os.path.isfile('Image_classify.keras') else r'C:\My_Project\Image_classification\Image_classify.keras'
    return load_model(model_path)

model = load_cached_model()

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

st.markdown(f'<div class="upload-box">{translations[language]["upload"]}</div>', unsafe_allow_html=True)

# Upload image
image = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False, label_visibility='collapsed')

def log_error(error_message):
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

if image:
    try:
        upload_progress = st.progress(100)

        image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="zoom-container">', unsafe_allow_html=True)
            st.image(image, caption='Uploaded Image', width=300)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner(translations[language]["processing"]):
            img_bat = tf.expand_dims(tf.image.resize(tf.image.decode_image(image.read(), channels=3), (img_height, img_width)), axis=0)
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict[0])

        with col2:
            predicted_label = data_cat[np.argmax(score)]
            accuracy = np.max(score) * 100

            st.markdown(f"<div class='prediction'>{translations[language]['prediction']}: {predicted_label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='accuracy'>{translations[language]['accuracy']}: {accuracy:.2f}%</div>", unsafe_allow_html=True)

    except Exception as e:
        log_error(str(e))
        st.error(f"Error processing image: {e}")

else:
    st.write("Please upload an image to proceed.")
