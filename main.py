import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = 224
DATASET = "clases"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(DATASET,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=8,
    subset="training")

val = datagen.flow_from_directory(DATASET,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=8,
    subset="validation")

base = MobileNetV2(weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE,IMG_SIZE,3))

base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(train.num_classes, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=out)

model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.fit(train, validation_data=val, epochs=10)

model.save("soil_model.h5")

print("✅ TRAIN DONE")