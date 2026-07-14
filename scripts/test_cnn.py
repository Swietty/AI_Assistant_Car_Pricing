from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model(
    "models/damage_cnn.keras"
)


labels = [
    "Crack",
    "Dent",
    "Flat_tire",
    "Glass_shatter",
    "Lamp_broken",
    "Scratch"
]


image = Image.open(
    r"D:\New_26\AI_Assistant_Car_Pricing\data\test_images\dent_1.jpg"
)


image = image.convert(
    "RGB"
)


image = image.resize(
    (224,224)
)


img = np.array(
    image,
    dtype=np.float32
)


img = img / 255.0


img = np.expand_dims(
    img,
    axis=0
)


prediction = model.predict(img)


print(prediction)


for i,p in enumerate(prediction[0]):

    print(
        labels[i],
        round(float(p),4)
    )