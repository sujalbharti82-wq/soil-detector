import os
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Soil AI PRO MAX", layout="wide")

DATASET = "clases"

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>🌱 Soil AI PRO MAX</h1>
<p style='text-align:center;'>AI Powered Soil Analysis & Recommendation</p>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Model Settings")

use_cnn = st.sidebar.checkbox("Use CNN Model", True)
sim_weight = st.sidebar.slider("Similarity Weight", 0.0, 1.0, 0.6)
cnn_weight = 1 - sim_weight

# ---------- SEASON ----------
season = st.sidebar.selectbox("🌦 Select Season", ["Summer", "Winter", "Monsoon"])

season_crops = {
    "Summer": ["Maize 🌽", "Groundnut 🌰"],
    "Winter": ["Wheat 🌾", "Peas"],
    "Monsoon": ["Rice 🌾", "Millets"]
}

# ---------- SOIL INFO ----------
soil_info = {
    "clay": {"crops": ["Rice 🌾","Broccoli 🥦","Cabbage 🥬"], "fertilizer": "Compost + Gypsum"},
    "sandy": {"crops": ["Watermelon 🍉","Potato 🥔","Groundnut 🌰"], "fertilizer": "Vermicompost + Organic Matter"},
    "black": {"crops": ["Cotton 🌿","Wheat 🌾","Soybean 🌱"], "fertilizer": "Urea + Potash"},
    "red": {"crops": ["Millets 🌾","Groundnut 🌰","Pulses 🌱"], "fertilizer": "NPK + Lime"},
    "mixed": {"crops": ["Maize 🌽","Wheat 🌾","Vegetables 🥕"], "fertilizer": "Balanced NPK + Compost"}
}

# ---------- LOAD ----------
@st.cache_resource
def load_all():
    cnn = tf.keras.models.load_model("soil_model.h5", compile=False)
    feat_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    features = np.load("features.npy")
    labels = np.load("labels.npy")
    return cnn, feat_model, features, labels

cnn_model, feat_model, features, labels = load_all()
classes = sorted(os.listdir(DATASET))

# ---------- FEATURES ----------
def deep_feature(img):
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = feat_model.predict(arr, verbose=0)[0]
    return feat / np.linalg.norm(feat)

def color_feature(img):
    arr = np.array(img.resize((100,100)))
    return np.array([np.mean(arr[:,:,0]), np.mean(arr[:,:,1]), np.mean(arr[:,:,2])]) / 255.0

def texture_feature(img):
    arr = np.array(img.resize((100,100)).convert("L"))
    return np.array([np.std(arr), np.mean(arr)]) / 255.0

def full_feature(img):
    return np.concatenate([deep_feature(img), color_feature(img), texture_feature(img)])

# ---------- MULTI CROP ----------
def multi_crop(img):
    w, h = img.size
    crops = [
        (0,0,w//2,h//2),(w//2,0,w,h//2),
        (0,h//2,w//2,h),(w//2,h//2,w,h),
        (w//4,h//4,3*w//4,3*h//4)
    ]
    return [img.crop(c) for c in crops]

# ---------- SIM ----------
def sim_part(part):
    feat = full_feature(part)
    sims = cosine_similarity([feat], features)[0]

    scores = {}
    for i, lbl in enumerate(labels):
        scores[lbl] = scores.get(lbl,0) + sims[i]

    for k in scores:
        scores[k] /= np.sum(labels == k)

    return scores

# ---------- CNN ----------
def cnn_part(part):
    part = part.resize((224,224))
    arr = image.img_to_array(part)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = cnn_model.predict(arr, verbose=0)[0]
    return {cls: pred[i] for i, cls in enumerate(classes)}

# ---------- FINAL ----------
def final_predict(img):
    total_scores = {}
    for part in multi_crop(img):
        if np.std(np.array(part)) < 12:
            continue

        sim_scores = sim_part(part)
        cnn_scores = cnn_part(part) if use_cnn else {}

        for cls in classes:
            score = (sim_scores.get(cls,0)*sim_weight + cnn_scores.get(cls,0)*cnn_weight)
            total_scores[cls] = total_scores.get(cls,0) + score

    return sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

# ---------- CAMERA STATE ----------
if "cam_on" not in st.session_state:
    st.session_state.cam_on = False

# ---------- INPUT ----------
colA, colB = st.columns([2,1])

# LEFT → Upload
with colA:
    uploaded = st.file_uploader("📤 Upload Image")

# RIGHT → Camera + Buttons
with colB:
    st.markdown("### 📷 Camera")

    colBtn1, colBtn2 = st.columns(2)

    with colBtn1:
        if st.button("ON"):
            st.session_state.cam_on = True

    with colBtn2:
        if st.button("OFF"):
            st.session_state.cam_on = False

    if st.session_state.cam_on:
        camera = st.camera_input("Take Photo")
    else:
        st.info("Camera is OFF")
        camera = None

# ---------- IMAGE SELECT ----------
img = None
if uploaded:
    img = Image.open(uploaded).convert("RGB")
elif camera:
    img = Image.open(camera).convert("RGB")

st.markdown("---")

# ---------- MAIN ----------
col1, col2 = st.columns(2)

if img:
    results = final_predict(img)
    top1 = results[0]
    total = sum([r[1] for r in results])
    conf = (top1[1]/total)*100

    info = soil_info.get(top1[0], soil_info["mixed"])

    gray = np.array(img.convert("L"))
    brightness = np.mean(gray)
    texture = np.std(gray)

    with col1:
        st.markdown("### 📸 Input Image")
        st.image(img, use_container_width=True)

        st.markdown("### 📊 Dataset Overview")
        counts = {c: len(os.listdir(os.path.join(DATASET,c))) for c in classes}
        st.bar_chart(counts)

        report = f"Prediction: {top1[0]}\nConfidence: {conf:.2f}%"
        st.download_button("⬇ Download Report", report)

    with col2:
        st.markdown(f"# 🌍 {top1[0].upper()} SOIL")
        st.markdown("---")

        st.markdown("### 🌾 Recommended Crops")
        for c in info["crops"]:
            st.write(f"✔ {c}")

        st.markdown("### 🌦 Best Crops (Soil + Season)")
        combined = list(set(info["crops"] + season_crops[season]))
        for crop in combined:
            st.write(f"✔ {crop}")

        st.markdown("### 🧪 Fertilizer")
        st.info(info["fertilizer"])

        st.markdown("---")

        st.markdown("### 🧠 Insights")
        colX, colY = st.columns(2)
        colX.metric("Brightness", f"{brightness:.1f}")
        colY.metric("Texture", f"{texture:.1f}")

        moisture = "Wet" if brightness < 90 else "Normal" if brightness < 130 else "Dry"
        st.write(f"💧 Moisture: {moisture}")

        st.markdown("---")

        st.markdown("### 📊 Confidence")
        st.progress(int(conf))
        st.write(f"{conf:.2f}%")

        st.markdown("---")

        st.markdown("### 📋 Summary")

        summary = f"""
🌍 Soil: {top1[0].upper()}  
🌦 Season: {season}  
📊 Confidence: {conf:.2f}%
"""
        st.info(summary)
