import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

DATASET = "clases"
IMG_SIZE = 224

soil_info = {
    "black": {"crop": "Cotton, Wheat", "fertilizer": "Urea + Potash"},
    "clay": {"crop": "Rice, Broccoli", "fertilizer": "Compost + Gypsum"},
    "red": {"crop": "Groundnut, Millets", "fertilizer": "NPK + Lime"},
    "sandy": {"crop": "Watermelon, Potato", "fertilizer": "Vermicompost"}
}

st.set_page_config(page_title="Soil AI", layout="wide")

# 🔥 MODEL LOAD (lightweight)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# 🔥 FEATURE EXTRACT
def extract_feature(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0)[0]
    return feat / np.linalg.norm(feat)

# 🔥 MULTI CROP
def extract_multi_features(img):
    w, h = img.size
    crops = [
        (0,0,w//2,h//2),
        (w//2,0,w,h//2),
        (0,h//2,w//2,h),
        (w//2,h//2,w,h),
        (w//4,h//4,3*w//4,3*h//4)
    ]
    return [extract_feature(img.crop(c)) for c in crops]

# 🔥 DATASET LOAD (FAST VERSION)
@st.cache_data
def load_dataset():
    features, labels = [], []
    for cls in os.listdir(DATASET):
        p = os.path.join(DATASET, cls)
        if not os.path.isdir(p): continue

        files = os.listdir(p)[:8]  # 🔥 limit images (fast)

        for f in files:
            try:
                img = Image.open(os.path.join(p,f)).convert("RGB")
                feats = extract_multi_features(img)
                for ft in feats:
                    features.append(ft)
                    labels.append(cls)
            except:
                pass
    return np.array(features), np.array(labels)

# 🔥 SAFE LOAD
try:
    features, labels = load_dataset()
except:
    features, labels = np.array([]), np.array([])

# ---------------- UI ----------------

st.markdown("""
<h1 style='text-align:center;color:#2E8B57;'>🌱 Soil Detection AI</h1>
<p style='text-align:center;color:gray;'>Smart Soil Analysis</p>
""", unsafe_allow_html=True)

left, right = st.columns([1.1,1])

with left:
    st.subheader("📤 Upload Image")
    file = st.file_uploader("", type=["jpg","jpeg","png"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=350)

with right:
    st.subheader("📊 Result")

    if file:
        if len(features) == 0:
            st.error("Dataset missing ❌")
        else:
            query = extract_multi_features(img)

            votes = []
            for q in query:
                sims = cosine_similarity([q], features)[0]
                votes.append(labels[np.argmax(sims)])

            vals, cnt = np.unique(votes, return_counts=True)
            final = vals[np.argmax(cnt)]
            conf = (np.max(cnt)/len(votes))*100
            info = soil_info[final]

            st.success(f"Soil: {final.upper()}")
            st.write(f"Confidence: {conf:.1f}%")

            st.info(f"🌱 Crop: {info['crop']}")
            st.info(f"🧪 Fertilizer: {info['fertilizer']}")

            st.progress(int(conf))

    else:
        st.write("Upload image")

st.markdown("<hr><p style='text-align:center;'>Made by Suju</p>", unsafe_allow_html=True)