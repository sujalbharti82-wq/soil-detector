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

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")
use_cnn = st.sidebar.toggle("Use CNN", True)
sim_weight = st.sidebar.slider("Similarity Weight", 0.0, 1.0, 0.6)
cnn_weight = 1 - sim_weight

# ---------- SOIL INFO ----------
soil_info = {
    "clay": {
        "crops": ["Rice 🌾", "Broccoli 🥦", "Cabbage 🥬"],
        "fertilizer": "Compost + Gypsum"
    },
    "sandy": {
        "crops": ["Watermelon 🍉", "Potato 🥔", "Groundnut 🌰"],
        "fertilizer": "Vermicompost + Organic Matter"
    },
    "black": {
        "crops": ["Cotton 🌿", "Wheat 🌾", "Soybean 🌱"],
        "fertilizer": "Urea + Potash"
    },
    "red": {
        "crops": ["Millets 🌾", "Groundnut 🌰", "Pulses 🌱"],
        "fertilizer": "NPK + Lime"
    },
    "mixed": {
        "crops": ["Maize 🌽", "Wheat 🌾", "Vegetables 🥕"],
        "fertilizer": "Balanced NPK + Compost"
    }
}

# ---------- LOAD ----------
@st.cache_resource
def load_all():
    cnn = tf.keras.models.load_model("soil_model.h5")
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

# 🔥 NEW (MATCH 1285)
def color_feature(img):
    arr = np.array(img.resize((100,100)))
    return np.array([
        np.mean(arr[:,:,0]),
        np.mean(arr[:,:,1]),
        np.mean(arr[:,:,2])
    ]) / 255.0

def texture_feature(img):
    arr = np.array(img.resize((100,100)).convert("L"))
    return np.array([
        np.std(arr),
        np.mean(arr)
    ]) / 255.0

def full_feature(img):
    return np.concatenate([
        deep_feature(img),
        color_feature(img),
        texture_feature(img)
    ])

# ---------- MULTI CROP ----------
def multi_crop(img):
    w, h = img.size
    crops = [
        (0,0,w//2,h//2),
        (w//2,0,w,h//2),
        (0,h//2,w//2,h),
        (w//2,h//2,w,h),
        (w//4,h//4,3*w//4,3*h//4)
    ]
    return [img.crop(c) for c in crops]

# ---------- SIM ----------
def sim_part(part):
    feat = full_feature(part)   # 🔥 FIXED

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
            s = sim_scores.get(cls,0)
            c = cnn_scores.get(cls,0)

            score = (s*sim_weight + c*cnn_weight)

            total_scores[cls] = total_scores.get(cls,0) + score

    sorted_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

# ---------- UI ----------
st.title("🌱 Soil AI PRO MAX")

col1, col2 = st.columns([1,1])

file = st.file_uploader("📤 Upload Soil Image")

if file:
    img = Image.open(file).convert("RGB")

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    results = final_predict(img)

    top1 = results[0]
    total = sum([r[1] for r in results])
    conf = (top1[1]/total)*100

    with col2:
        st.subheader("🔍 Prediction")
        st.success(f"🏆 {top1[0].upper()}")
        st.progress(int(conf))
        st.write(f"Confidence: {conf:.2f}%")

        st.markdown("### 📊 Top 3 Results")
        for cls, score in results[:3]:
            percent = (score/total)*100
            st.write(f"{cls} → {percent:.2f}%")

    # ---------- RECOMMENDATION ----------
    st.markdown("---")
    st.subheader("🌾 Crop Recommendation")

    soil_type = top1[0]
    info = soil_info.get(soil_type, soil_info["mixed"])

    st.success("Recommended Crops:")
    for crop in info["crops"]:
        st.write(f"✔ {crop}")

    st.subheader("🧪 Fertilizer")
    st.info(info["fertilizer"])

    if conf < 50:
        st.warning("⚠️ Low confidence → possible mixed soil")