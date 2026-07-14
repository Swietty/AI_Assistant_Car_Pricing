import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


# -------------------------------------------------
# Load model
# -------------------------------------------------

model = tf.keras.models.load_model(
    "../models/damage_classifier_v2.keras"
)

with open("../models/class_names.json") as f:
    classes = json.load(f)


# -------------------------------------------------
# Load image
# -------------------------------------------------

IMAGE_PATH = Path("../data/test_images/dent_02.jpg")

img = load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

x = img_to_array(img)
x = np.expand_dims(x, axis=0)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

prediction = model.predict(x, verbose=0)[0]

best_idx = np.argmax(prediction)


print("=" * 50)
print(f"Image: {IMAGE_PATH.name}")
print("=" * 50)

print("\nProbabilities:\n")

for cls, prob in sorted(
    zip(classes, prediction),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{cls:<18} {prob*100:6.2f} %")

print("\n" + "=" * 50)
print(f"Predicted class : {classes[best_idx]}")
print(f"Confidence      : {prediction[best_idx]*100:.2f} %")
print("=" * 50)