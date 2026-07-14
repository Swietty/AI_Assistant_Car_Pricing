import os
import shutil
import json

import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# PATHS
# =====================================================

CSV_PATH = "../data/raw/train.csv"
IMAGE_FOLDER = "../data/raw/images"
DATASET_FOLDER = "../data/damage_images"
MODEL_FOLDER = "../models"
MODEL_PATH = "../models/damage_classifier.keras"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# =====================================================
# LABELS
# =====================================================

label_map = {
    1: "Crack",
    2: "Scratch",
    3: "Flat_tire",
    4: "Dent",
    5: "Glass_shatter",
    6: "Lamp_broken"
}

# =====================================================
# CREATE DATASET STRUCTURE
# =====================================================

def prepare_dataset():

    print("Preparing dataset...")


    df = pd.read_csv(CSV_PATH)


    for class_name in label_map.values():

        os.makedirs(
            os.path.join(
                DATASET_FOLDER,
                class_name
            ),
            exist_ok=True
        )

    copied = 0

    for _, row in df.iterrows():

        src = os.path.join(
            IMAGE_FOLDER,
            row["filename"]
        )

        class_name = label_map[row["label"]]

        dst = os.path.join(
            DATASET_FOLDER,
            class_name,
            row["filename"]
        )

        if os.path.exists(src):

            shutil.copy2(
                src,
                dst
            )

            copied += 1

    print(
        f"Copied images: {copied}"
    )

# =====================================================
# LOAD DATA
# =====================================================

def create_datasets():


    train_ds = tf.keras.utils.image_dataset_from_directory(

        DATASET_FOLDER,

        validation_split=0.2,

        subset="training",

        seed=42,

        image_size=(224,224),

        batch_size=32
    )



    val_ds = tf.keras.utils.image_dataset_from_directory(

        DATASET_FOLDER,

        validation_split=0.2,

        subset="validation",

        seed=42,

        image_size=(224,224),

        batch_size=32
    )


    class_names = train_ds.class_names


    print(
        "CLASS ORDER:"
    )

    print(class_names)



    with open(
        "../models/class_names.json",
        "w"
    ) as f:

        json.dump(
            class_names,
            f
        )


    AUTOTUNE = tf.data.AUTOTUNE


    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        AUTOTUNE
    )


    return train_ds, val_ds, class_names



# =====================================================
# BUILD MODEL
# =====================================================


def build_model(num_classes):


    base_model = EfficientNetB0(

        weights="imagenet",

        include_top=False,

        input_shape=(224,224,3)

    )


    base_model.trainable = False



    model = models.Sequential([
      
        base_model,


        layers.GlobalAveragePooling2D(),


        layers.Dropout(
            0.4
        ),


        layers.Dense(
            128,
            activation="relu"
        ),


        layers.Dropout(
            0.3
        ),


        layers.Dense(
            num_classes,
            activation="softmax"
        )

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model




# =====================================================
# TRAIN
# =====================================================


def train():


    prepare_dataset()


    train_ds, val_ds, class_names = create_datasets()


    model = build_model(
        len(class_names)
    )


    callbacks = [


        ModelCheckpoint(

            MODEL_PATH,

            monitor="val_accuracy",

            save_best_only=True,

            mode="max"

        ),



        EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True

        ),



        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.3,

            patience=2

        )

    ]



    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=10,

        callbacks=callbacks

    )


    print(
        "Training finished"
    )


    print(
        "Best model saved:"
    )

    print(
        MODEL_PATH
    )



if __name__ == "__main__":

    train()