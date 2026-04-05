import os
import numpy as np
from PIL import Image, ImageEnhance

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

DATASET = "clases"

print("🚀 Building SUPER AUGMENTED features...")

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ---------- FEATURES ----------
def deep_feature(img):
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr, verbose=0)[0]
    return feat / np.linalg.norm(feat)

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

# ---------- AUGMENT ----------
def augment(img):
    imgs = []

    imgs.append(img)
    imgs.append(img.rotate(15))
    imgs.append(img.rotate(-15))
    imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # brightness
    enhancer = ImageEnhance.Brightness(img)
    imgs.append(enhancer.enhance(1.2))
    imgs.append(enhancer.enhance(0.8))

    return imgs

# ---------- BUILD ----------
features, labels, colors = [], [], []

for cls in os.listdir(DATASET):
    path = os.path.join(DATASET, cls)
    if not os.path.isdir(path): continue

    print("Processing:", cls)

    for f in os.listdir(path):
        try:
            img = Image.open(os.path.join(path,f)).convert("RGB")

            # 🔥 AUGMENT
            aug_imgs = augment(img)

            for aimg in aug_imgs:
                parts = multi_crop(aimg)

                for part in parts:
                    features.append(full_feature(part))
                    labels.append(cls)
                    colors.append(color_feature(part))

        except:
            pass

features = np.array(features)
labels = np.array(labels)
colors = np.array(colors)

print("Shape:", features.shape)

np.save("features.npy", features)
np.save("labels.npy", labels)
np.save("color.npy", colors)

print("✅ SUPER DATASET READY 🚀")